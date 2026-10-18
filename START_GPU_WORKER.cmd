@echo off
cd /d "%~dp0"
if exist ".venv-run\Scripts\python.exe" (
  ".venv-run\Scripts\python.exe" scripts\serve_gpu_worker.py %*
) else (
  python scripts\serve_gpu_worker.py %*
)
pause
