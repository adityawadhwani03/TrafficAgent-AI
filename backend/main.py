"""
Traffic Agent Backend — FastAPI
Endpoints:
  POST /analyze          → analyze uploaded image(s)
  POST /analyze/bdd100k  → run on local BDD100K directory
  GET  /results          → list past results
  GET  /status           → system health (ollama, yolo, chroma)
  GET  /stream/{job_id}  → SSE stream for live progress
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import chromadb
import ollama
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
STATIC_DIR   = BASE_DIR / "static"
UPLOADS_DIR  = BASE_DIR / "uploads"
RESULTS_DIR  = BASE_DIR / "results"
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────
SLM_MODEL         = "phi3:mini"
YOLO_MODEL        = "yolo11n.pt"
CONF_THRESH       = 0.35
CHROMA_COLLECTION = "traffic_rules"

# ─── Traffic Rules Knowledge Base ─────────────────────────────────────
TRAFFIC_RULES = [
    {"id": "r001", "category": "pedestrian",    "rule": "Yield to pedestrians in crosswalks at all times. When a pedestrian is detected in or entering the crosswalk, the vehicle must stop."},
    {"id": "r002", "category": "pedestrian",    "rule": "At intersections without crosswalk markings, pedestrians crossing the roadway must be yielded to if they are in the driver's half of the road or close enough to be a hazard."},
    {"id": "r003", "category": "pedestrian",    "rule": "When a pedestrian is detected on the road outside a crosswalk at night, reduce speed and be prepared to stop."},
    {"id": "r004", "category": "traffic_light", "rule": "A red traffic light requires a full stop before the stop line. Do not proceed until the light turns green."},
    {"id": "r005", "category": "traffic_light", "rule": "A yellow (amber) traffic light means prepare to stop. Only proceed if stopping safely is not possible."},
    {"id": "r006", "category": "traffic_light", "rule": "A green traffic light permits proceeding through the intersection, but you must still yield to vehicles and pedestrians already in the intersection."},
    {"id": "r007", "category": "traffic_light", "rule": "A flashing red light must be treated as a stop sign: come to a full stop, then proceed when safe."},
    {"id": "r008", "category": "stop_sign",     "rule": "At a stop sign, come to a complete stop at the stop line, crosswalk, or intersection edge. Proceed only when the way is clear."},
    {"id": "r009", "category": "stop_sign",     "rule": "At a 4-way stop, the vehicle that arrives first has right of way. If simultaneous, the vehicle on the right goes first."},
    {"id": "r010", "category": "vehicle",       "rule": "Maintain a safe following distance from the vehicle ahead. The 3-second rule applies in normal conditions; increase to 6+ seconds in rain or low visibility."},
    {"id": "r011", "category": "vehicle",       "rule": "When merging or changing lanes, ensure adequate gap between vehicles. Do not cut off other drivers."},
    {"id": "r012", "category": "vehicle",       "rule": "Emergency vehicles (ambulance, fire truck, police) with active lights/sirens must be yielded to immediately. Pull to the right and stop."},
    {"id": "r013", "category": "speed",         "rule": "Reduce speed in school zones, construction zones, and residential areas. Posted speed limits are maximums under ideal conditions."},
    {"id": "r014", "category": "speed",         "rule": "In adverse weather (rain, fog, snow) reduce speed significantly below the posted limit to maintain vehicle control."},
    {"id": "r015", "category": "cyclist",       "rule": "Cyclists must be given at least 3 feet of clearance when passing. Do not overtake a cyclist at an intersection."},
    {"id": "r016", "category": "cyclist",       "rule": "Motorcycles are entitled to a full lane. Do not share a lane with or weave around motorcycles."},
    {"id": "r017", "category": "visibility",    "rule": "At night or in low visibility, headlights must be on. High beams should be dimmed when oncoming traffic is within 500 feet."},
    {"id": "r018", "category": "visibility",    "rule": "In fog, use fog lights or low beams. Never use high beams in fog as it increases glare and reduces visibility further."},
]

# ─── Global state ─────────────────────────────────────────────────────
yolo_model    = None
rag_collection = None
job_streams: dict[str, list] = {}   # job_id → list of SSE event dicts

# ─── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Traffic Agent API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global yolo_model, rag_collection
    print("[STARTUP] Loading YOLO …")
    try:
        yolo_model = YOLO(YOLO_MODEL)
        print("[STARTUP] YOLO ready ✓")
    except Exception as e:
        print(f"[STARTUP] YOLO failed: {e}")

    print("[STARTUP] Building ChromaDB RAG store …")
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        client = chromadb.Client()
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        rag_collection = client.create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        rag_collection.add(
            ids=[r["id"] for r in TRAFFIC_RULES],
            documents=[r["rule"] for r in TRAFFIC_RULES],
            metadatas=[{"category": r["category"]} for r in TRAFFIC_RULES],
        )
        print(f"[STARTUP] ChromaDB ready — {len(TRAFFIC_RULES)} rules indexed ✓")
    except Exception as e:
        print(f"[STARTUP] ChromaDB failed: {e}")


# ─── Helpers ──────────────────────────────────────────────────────────
def detect(image_path: Path) -> list[dict]:
    if yolo_model is None:
        return []
    results = yolo_model(str(image_path), conf=CONF_THRESH, verbose=False)[0]
    out = []
    for box in results.boxes:
        out.append({
            "label":      results.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "box":        [round(v, 1) for v in box.xyxy[0].tolist()],
        })
    return out


def retrieve_rules(query: str, n: int = 3) -> list[dict]:
    if rag_collection is None:
        return []
    res = rag_collection.query(query_texts=[query], n_results=n)
    rules = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        rules.append({"rule": doc, "category": meta["category"], "relevance": round(1 - dist, 3)})
    return rules


SYSTEM_PROMPT = """You are an autonomous driving safety agent.
You receive YOLO detections and relevant traffic rules.
Respond ONLY with this JSON (no markdown, no extra text):
{
  "scene_description": "<2-3 sentence description>",
  "reasoning": "<step-by-step safety reasoning>",
  "recommended_action": "<concrete driving action>",
  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>"
}"""


def run_slm(detections: list[dict], rules: list[dict]) -> dict:
    det_text  = "\n".join(f"  - {d['label']} (conf={d['confidence']})" for d in detections) or "  - No objects detected"
    rule_text = "\n".join(f"  [{r['category']}] {r['rule']}" for r in rules)
    user_msg  = f"DETECTED OBJECTS:\n{det_text}\n\nRELEVANT TRAFFIC RULES:\n{rule_text}\n\nAnalyze and respond in JSON."
    try:
        resp = ollama.chat(
            model=SLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            options={"temperature": 0.1, "num_predict": 512},
        )
        raw = resp["message"]["content"].strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "scene_description": f"SLM error: {e}",
            "reasoning": "Could not complete reasoning.",
            "recommended_action": "Manual review required.",
            "risk_level": "UNKNOWN",
        }


def analyze_image(image_path: Path) -> dict:
    t0          = time.perf_counter()
    detections  = detect(image_path)
    labels      = ", ".join({d["label"] for d in detections}) or "empty road"
    rules       = retrieve_rules(f"Traffic scene with: {labels}")
    agent_out   = run_slm(detections, rules)
    latency_ms  = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "id":                 str(uuid.uuid4())[:8],
        "image":              image_path.name,
        "detections":         detections,
        "rules_retrieved":    rules,
        "scene_description":  agent_out.get("scene_description", ""),
        "agent_reasoning":    agent_out.get("reasoning", ""),
        "recommended_action": agent_out.get("recommended_action", ""),
        "risk_level":         agent_out.get("risk_level", "UNKNOWN"),
        "latency_ms":         latency_ms,
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_result(result: dict):
    path = RESULTS_DIR / f"{result['id']}.json"
    path.write_text(json.dumps(result, indent=2))


# ─── Routes ───────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    return HTMLResponse(index.read_text())


@app.get("/status")
async def status():
    ollama_ok = False
    try:
        ollama.list()
        ollama_ok = True
    except Exception:
        pass
    return {
        "yolo":    yolo_model is not None,
        "chroma":  rag_collection is not None,
        "ollama":  ollama_ok,
        "slm":     SLM_MODEL,
        "yolo_model": YOLO_MODEL,
    }


@app.post("/analyze")
async def analyze_upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        dest = UPLOADS_DIR / f.filename
        dest.write_bytes(await f.read())
        result = analyze_image(dest)
        save_result(result)
        results.append(result)
    return {"results": results}


@app.post("/analyze/bdd100k")
async def analyze_bdd100k(
    background_tasks: BackgroundTasks,
    directory: str = Form(...),
    max_images: int = Form(20),
):
    job_id = str(uuid.uuid4())[:8]
    job_streams[job_id] = []

    async def run():
        d = Path(directory)
        if not d.exists():
            job_streams[job_id].append({"type": "error", "message": f"Directory not found: {directory}"})
            job_streams[job_id].append({"type": "done"})
            return
        exts   = {".jpg", ".jpeg", ".png"}
        images = [p for p in sorted(d.iterdir()) if p.suffix.lower() in exts][:max_images]
        job_streams[job_id].append({"type": "start", "total": len(images)})
        for i, img in enumerate(images):
            result = await asyncio.get_event_loop().run_in_executor(None, analyze_image, img)
            save_result(result)
            job_streams[job_id].append({"type": "result", "index": i + 1, "total": len(images), "data": result})
            await asyncio.sleep(0)
        job_streams[job_id].append({"type": "done"})

    background_tasks.add_task(run)
    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream_job(job_id: str):
    async def event_gen() -> AsyncGenerator[str, None]:
        sent = 0
        while True:
            events = job_streams.get(job_id, [])
            while sent < len(events):
                ev = events[sent]
                yield f"data: {json.dumps(ev)}\n\n"
                sent += 1
                if ev.get("type") == "done":
                    return
            await asyncio.sleep(0.3)
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/results")
async def list_results():
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = [json.loads(f.read_text()) for f in files[:100]]
    return {"results": results}


@app.delete("/results")
async def clear_results():
    for f in RESULTS_DIR.glob("*.json"):
        f.unlink()
    return {"message": "Cleared"}
