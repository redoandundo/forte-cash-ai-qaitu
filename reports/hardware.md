# Hardware and environment

- Date: 2026-10-17
- OS: Windows
- Python: 3.12.14 in `.venv-ml`
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB
- Driver: 610.62
- PyTorch challenger environment: 2.11.0+cu128
- Torchvision challenger environment: 0.26.0+cu128
- GPU compute capability reported by PyTorch: 12.0
- CUDA available to PyTorch: yes
- FFmpeg CLI: not installed; clip extraction uses OpenCV locally

The challenger training environment is isolated from the service environment in `.venv-ml`.
The older lock file documents the original baseline environment; challenger-specific direct
dependencies are declared in `requirements-ml.txt` and the exact runtime is recorded in the
challenger report.
