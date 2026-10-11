@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.lock.txt
if errorlevel 1 goto failed
echo Installed. Run start.cmd and open http://localhost:8501
echo For local semantic embeddings, also install requirements-local.txt. See README.
pause
exit /b 0
:failed
echo Setup failed. Check Python 3.11+ and your package mirror/network. See README.
pause
exit /b 1
