@echo off
REM ──────────────────────────────────────────────
REM  Traffic Agent — Windows launcher
REM  Usage: double-click or run from cmd
REM ──────────────────────────────────────────────

cd /d "%~dp0"
set PORT=8000

echo.
echo   Traffic Agent  v1.0
echo   YOLO x ChromaDB RAG x Phi-3 mini
echo.

REM ── Check Python ──────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM ── Virtual env ───────────────────────────────
if not exist ".venv\" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM ── Dependencies ──────────────────────────────
echo [SETUP] Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

REM ── Ollama check ──────────────────────────────
ollama list >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Ollama not found.
    echo   Install from: https://ollama.com/download
    echo   Then run:     ollama pull phi3:mini
    echo.
) else (
    echo [OLLAMA] Pulling phi3:mini if not present...
    ollama pull phi3:mini
)

REM ── Launch ────────────────────────────────────
echo.
echo [START] Opening http://localhost:%PORT% in browser...
start http://localhost:%PORT%
echo [START] Press Ctrl+C to stop.
echo.

uvicorn backend.main:app --host 0.0.0.0 --port %PORT% --reload
pause
