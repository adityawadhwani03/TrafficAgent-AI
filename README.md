# Traffic Agent 🚦
### YOLO Detection × ChromaDB RAG × Phi-3 Mini (Local SLM)

A full-stack autonomous driving analysis application.  
Upload traffic images or run batch analysis on BDD100K — get real-time risk assessments powered entirely by local AI.

---

## Architecture

```
Image Input
    │
    ▼
[YOLO11n]  ──→  Object detections (labels + bounding boxes)
    │
    ▼
[ChromaDB RAG]  ──→  Top-3 relevant traffic rules (sentence-transformers embeddings)
    │
    ▼
[Phi-3 Mini via Ollama]  ──→  Scene reasoning + recommended action + risk level
    │
    ▼
FastAPI backend  ──→  Web dashboard (real-time SSE streaming)
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | https://python.org |
| Ollama | https://ollama.com/download |
| Git (optional) | https://git-scm.com |

---

## Quick Start

### macOS / Linux
```bash
bash run.sh
```

### Windows
```
Double-click run.bat
```
Or from Command Prompt:
```cmd
run.bat
```

The script will:
1. Create a Python virtual environment
2. Install all dependencies
3. Pull `phi3:mini` from Ollama (first run, ~2.3GB)
4. Start the server at **http://localhost:8000**

---

## Manual Setup

```bash
# 1. Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the SLM
ollama pull phi3:mini

# 4. Start the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000

---

## Using the App

### Upload Mode
- Drag & drop any JPG/PNG traffic images onto the upload zone
- Or click to browse files
- Results appear instantly in the dashboard

### BDD100K Batch Mode
1. Download BDD100K from https://bdd-data.berkeley.edu/
2. Enter the path to your val images directory (e.g. `./bdd100k/images/100k/val`)
3. Set max images (default 20)
4. Click **▶ Run Batch** — watch live progress stream

### Results Dashboard
- Click any result card to see full analysis:
  - Scene description
  - Step-by-step agent reasoning
  - Retrieved traffic rules with relevance scores
  - All YOLO detections with confidence
- Risk levels: **LOW** / **MEDIUM** / **HIGH** / **CRITICAL**
- Live log panel shows real-time processing output

---

## Configuration

Edit `backend/main.py` to change defaults:

```python
SLM_MODEL    = "phi3:mini"    # or "mistral", "llama3.2", etc.
YOLO_MODEL   = "yolo11n.pt"   # yolo11s.pt / yolo11m.pt for more accuracy
CONF_THRESH  = 0.35           # YOLO confidence threshold
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`           | Web dashboard |
| `GET`  | `/status`     | System health check |
| `POST` | `/analyze`    | Upload + analyze image(s) |
| `POST` | `/analyze/bdd100k` | Start batch job |
| `GET`  | `/stream/{job_id}` | SSE live progress |
| `GET`  | `/results`    | List all past results |
| `DELETE` | `/results`  | Clear all results |

---

## Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Object Detection | YOLOv11-nano | ~6MB | Fast, real-time detection |
| Embeddings | all-MiniLM-L6-v2 | ~80MB | Traffic rules RAG |
| SLM Reasoning | Phi-3 Mini | ~2.3GB | Scene understanding + decisions |

---

## Output Format

Results saved to `results/` as JSON:
```json
{
  "id": "a1b2c3d4",
  "image": "frame_001.jpg",
  "detections": [{"label": "car", "confidence": 0.87, "box": [...]}],
  "rules_retrieved": [{"rule": "...", "category": "vehicle", "relevance": 0.91}],
  "scene_description": "...",
  "agent_reasoning": "...",
  "recommended_action": "Maintain safe following distance and prepare to brake.",
  "risk_level": "MEDIUM",
  "latency_ms": 1847,
  "timestamp": "2026-03-16T14:23:01"
}
```
