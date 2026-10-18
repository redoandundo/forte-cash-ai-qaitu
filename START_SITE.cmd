@echo off
cd /d "%~dp0"
if exist ".venv-ml\Scripts\python.exe" (
  ".venv-ml\Scripts\python.exe" scripts\serve_site.py
) else if exist ".venv-run\Scripts\python.exe" (
  ".venv-run\Scripts\python.exe" scripts\serve_site.py
) else (
  echo Server environment missing. Run: py -3.12 scripts/setup_local.py --qwen
)
pause
