@echo off
echo ============================================================
echo  NL2SQL LlamaIndex Pipeline — Setup
echo ============================================================
echo.

REM Check Python
echo [1/4] Checking Python...
python --version 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found! Install from python.org
    pause
    exit /b 1
)
echo.

REM Check Ollama
echo [2/4] Checking Ollama...
ollama list 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Ollama not responding. Make sure it's running.
)
echo.

REM Install dependencies
echo [3/4] Installing Python packages (this may take a few minutes)...
pip install -r requirements.txt
echo.

REM Create .env if missing
echo [4/4] Checking .env file...
if not exist .env (
    copy .env.template .env
    echo.
    echo  Created .env from template.
    echo  IMPORTANT: Edit .env with your database credentials!
    echo  Open it with: notepad .env
    echo.
) else (
    echo  .env already exists.
)

echo.
echo ============================================================
echo  Setup complete! Next steps:
echo.
echo  1. Edit .env with your database credentials:
echo     notepad .env
echo.
echo  2. Make sure Ollama is running:
echo     ollama serve
echo.
echo  3. Run the pipeline:
echo     python run.py              (full mode)
echo     python run.py --sql-only   (SQL only, no execution)
echo     python run.py --test       (test DB connection)
echo ============================================================
pause
