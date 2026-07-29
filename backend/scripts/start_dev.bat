@echo off
REM ============================================================
REM InspectAI dev launcher
REM
REM Starts the FastAPI backend in its own window, polls
REM /api/ready until the orchestrator (LLMs + embeddings + schema)
REM has finished loading, then starts the Angular dev server.
REM
REM Backend runs on port 8000 to match frontend\proxy.conf.json,
REM which proxies /api calls from `ng serve` to that port.
REM ============================================================

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "BACKEND_PORT=8000"
set "READY_URL=http://localhost:%BACKEND_PORT%/api/ready"
set "MAX_WAIT_SECONDS=300"

echo ============================================
echo  InspectAI - starting backend + frontend
echo ============================================
echo Repo root: %REPO_ROOT%
echo.

REM ---- start backend (activates venv if present) ----
REM PYTHONUTF8=1 forces UTF-8 stdout so startup logging with unicode
REM characters (checkmarks, etc.) doesn't crash on the cp1252 console.
set "VENV_ACTIVATE=%REPO_ROOT%\venv\Scripts\activate.bat"
if exist "%VENV_ACTIVATE%" (
    start "InspectAI Backend" cmd /k "cd /d "%REPO_ROOT%" && call "%VENV_ACTIVATE%" && set "PYTHONUTF8=1" && python -m uvicorn src.api.app:app --app-dir backend --port %BACKEND_PORT%"
) else (
    start "InspectAI Backend" cmd /k "cd /d "%REPO_ROOT%" && set "PYTHONUTF8=1" && python -m uvicorn src.api.app:app --app-dir backend --port %BACKEND_PORT%"
)

echo Backend launching in a separate window...
echo Waiting for it to finish loading (LLMs + embeddings + schema)...
echo.

set /a ELAPSED=0

:WAIT_LOOP
curl -s -o nul -w "%%{http_code}" "%READY_URL%" > "%TEMP%\inspectai_ready_code.txt" 2>nul
set /p READY_CODE=<"%TEMP%\inspectai_ready_code.txt"
if "%READY_CODE%"=="200" goto BACKEND_READY

if %ELAPSED% GEQ %MAX_WAIT_SECONDS% (
    echo.
    echo [WARN] Backend did not report ready within %MAX_WAIT_SECONDS%s - starting frontend anyway.
    echo        Check the backend window for errors.
    goto START_FRONTEND
)

timeout /t 2 /nobreak >nul
set /a ELAPSED+=2
echo   ...still loading ^(%ELAPSED%s elapsed^)
goto WAIT_LOOP

:BACKEND_READY
echo.
echo Backend is ready.

:START_FRONTEND
echo.
echo Starting frontend (ng serve)...
start "InspectAI Frontend" cmd /k "cd /d "%REPO_ROOT%\frontend" && npm start"

del "%TEMP%\inspectai_ready_code.txt" >nul 2>nul

echo.
echo Both processes launched in separate windows.
echo   Backend:  http://localhost:%BACKEND_PORT%
echo   Frontend: http://localhost:4200
