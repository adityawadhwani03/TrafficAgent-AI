#!/usr/bin/env bash
# ──────────────────────────────────────────────
#  Traffic Agent — launcher
#  Usage: bash run.sh
# ──────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
PORT=${PORT:-8000}

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║      Traffic Agent  v1.0             ║"
echo "  ║  YOLO × ChromaDB RAG × Phi-3 mini    ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── Check Python ──────────────────────────────
if ! command -v $PYTHON &>/dev/null; then
  echo "[ERROR] Python3 not found. Install from https://python.org"
  exit 1
fi

# ── Virtual env ───────────────────────────────
if [ ! -d ".venv" ]; then
  echo "[SETUP] Creating virtual environment…"
  $PYTHON -m venv .venv
fi
source .venv/bin/activate

# ── Dependencies ──────────────────────────────
echo "[SETUP] Installing/verifying dependencies…"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ── Ollama check ──────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo ""
  echo "[WARNING] Ollama not found."
  echo "  Install from: https://ollama.com/download"
  echo "  Then run:     ollama pull phi3:mini"
  echo ""
else
  echo "[OLLAMA] Checking phi3:mini…"
  if ollama list 2>/dev/null | grep -q "phi3:mini"; then
    echo "[OLLAMA] phi3:mini is ready ✓"
  else
    echo "[OLLAMA] Pulling phi3:mini (this may take a few minutes)…"
    ollama pull phi3:mini
  fi
fi

# ── Launch ────────────────────────────────────
echo ""
echo "[START] Launching server on http://localhost:$PORT"
echo "[START] Press Ctrl+C to stop."
echo ""

uvicorn backend.main:app --host 0.0.0.0 --port $PORT --reload
