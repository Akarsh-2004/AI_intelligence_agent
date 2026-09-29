@echo off
echo ============================================================
echo   Starting Ollama server (CPU mode)...
echo ============================================================

set OLLAMA_RUNNERS_DIR=C:\Users\akars\AppData\Local\Programs\Ollama\lib\ollama
set OLLAMA_MODELS=%USERPROFILE%\.ollama\models
set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_MODEL=gemma2:2b
set OLLAMA_BASE_URL=http://localhost:11434

REM Kill any stale Ollama processes first
taskkill /F /IM "ollama.exe" /T >nul 2>&1
taskkill /F /IM "ollama app.exe" /T >nul 2>&1
timeout /t 2 /nobreak >nul

REM Start ollama serve in the background (detached from this window)
start /B "OllamaServer" "C:\Users\akars\AppData\Local\Programs\Ollama\ollama.exe" serve

echo Waiting for Ollama to be ready...
timeout /t 8 /nobreak >nul

REM Quick health check
curl -s http://localhost:11434 >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting a bit more...
    timeout /t 5 /nobreak >nul
)

curl -s http://localhost:11434
echo.
echo ============================================================
echo   Ollama ready! Running pipeline...
echo ============================================================

python run_pipeline.py
