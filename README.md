# 🚦 TrafficAgent-AI

## Real-Time Autonomous Traffic Analysis Powered by YOLO & RAG

![Status](https://img.shields.io/badge/Project%20Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Version](https://img.shields.io/badge/Version-1.0.0-blue)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v11-FF6B35?logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-FF4B4B?logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Phi--3%20Mini-black?logo=ollama&logoColor=white)
![Phi3](https://img.shields.io/badge/Phi--3-Mini%203.8B-blueviolet)

**An intelligent, autonomous traffic scene analyzer with real-time object detection, RAG-powered rule retrieval, and local AI reasoning for safety risk assessment.**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [API](#api)

---

## Features

- Real-time object detection using YOLOv11-nano
- ChromaDB RAG pipeline searches 18 traffic rules instantly
- Phi-3 Mini runs 100% locally — no internet needed for AI
- Live dashboard with risk levels: LOW / MEDIUM / HIGH / CRITICAL
- Batch processing support for BDD100K dataset
- Server Sent Events for real-time progress streaming
- Full REST API with FastAPI backend

---

## Architecture

```
Image Input
    |
    v
[YOLOv11-nano] ---------> Detects objects (car, person, bus, traffic light)
    |
    v
[ChromaDB RAG] ----------> Finds top 3 relevant traffic rules
    |
    v
[Phi-3 Mini via Ollama] --> Reasons about scene, gives action + risk level
    |
    v
[FastAPI + Dashboard] ----> Real-time results in browser
```

---

## Installation

### Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10+ | https://python.org |
| Ollama | Latest | https://ollama.com/download |
| Git | Latest | https://git-scm.com |

### Setup

**Step 1 - Clone the repo**
```bash
git clone https://github.com/adityawadhwani03/TrafficAgent-AI.git
cd TrafficAgent-AI
```

**Step 2 - Create virtual environment**
```bash
python -m venv .venv
```

**Step 3 - Activate it**

Windows:
```cmd
.venv\Scripts\activate
```
Mac/Linux:
```bash
source .venv/bin/activate
```

**Step 4 - Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 5 - Pull the AI model**
```bash
ollama pull phi3:mini
```

**Step 6 - Run the app**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Step 7 - Open in browser**
```
http://localhost:8000
```

---

## Usage

### Upload Mode
1. Drag and drop any traffic image onto the upload zone
2. Wait 15 to 30 seconds for AI analysis
3. Result card appears with risk level and recommended action
4. Click any card to see full detailed breakdown

### BDD100K Batch Mode
1. Download BDD100K from https://bdd-data.berkeley.edu
2. Enter the path to your images folder
3. Set how many images to process
4. Click RUN BATCH and watch live progress

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Web dashboard |
| GET | /status | System health check |
| POST | /analyze | Upload and analyze images |
| POST | /analyze/bdd100k | Start batch job |
| GET | /stream/{job_id} | Live SSE progress stream |
| GET | /results | List all past results |
| DELETE | /results | Clear all results |

---

## Output Format

```json
{
  "id": "a1b2c3d4",
  "image": "traffic_photo.jpg",
  "detections": [
    {"label": "car", "confidence": 0.87},
    {"label": "person", "confidence": 0.76}
  ],
  "rules_retrieved": [
    {
      "rule": "Yield to pedestrians in crosswalks",
      "category": "pedestrian",
      "relevance": 0.91
    }
  ],
  "scene_description": "A busy intersection with vehicles and a pedestrian.",
  "agent_reasoning": "Step 1: Pedestrian detected near crosswalk...",
  "recommended_action": "Slow down and yield to pedestrian in crosswalk.",
  "risk_level": "HIGH",
  "latency_ms": 18500,
  "timestamp": "2026-03-17T01:43:14"
}
```

---

## Models

| Model | Size | Purpose |
|-------|------|---------|
| YOLOv11-nano | 6 MB | Fast object detection |
| all-MiniLM-L6-v2 | 80 MB | Text embeddings for rule search |
| Phi-3 Mini | 2.3 GB | Local AI reasoning |

---

## Project Structure

```
TrafficAgent-AI/
    backend/
        __init__.py
        main.py
    static/
        index.html
    uploads/
    results/
    requirements.txt
    run.bat
    run.sh
    README.md
```

---

## Share Your App Online

```cmd
winget install Ngrok.Ngrok
ngrok config add-authtoken YOUR_TOKEN
ngrok http 8000
```

You get a public link like `https://abc123.ngrok-free.app` that anyone can open!

---

## What I Learned

- How YOLO object detection works in real traffic scenes
- How RAG pipelines work with ChromaDB vector search
- How to run local Small Language Models with Ollama
- How to build a FastAPI backend with Server Sent Events
- How to build and deploy a full-stack AI application

---

## Future Improvements

- Add video file support
- Add GPS coordinates to results
- Email alerts for CRITICAL risk detections
- Train YOLO on BDD100K for better traffic accuracy
- Support for Mistral and LLaMA models

---

## License

MIT License — free to use for learning and building!

---

## Author

**Aditya Wadhwani**

[![GitHub](https://img.shields.io/badge/GitHub-adityawadhwani03-black?logo=github)](https://github.com/adityawadhwani03)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Aditya%20Wadhwani-0077B5?logo=linkedin)](https://linkedin.com/in/adityawadhwani03)

> Built as a fresher AI project | Stack: Python, FastAPI, YOLO, ChromaDB, Ollama, Phi-3 Mini
