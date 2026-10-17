# Hardware and environment

- Date: 2026-10-17
- OS: Windows
- Python: 3.12.14 in `.venv-ml`
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- Driver: 546.17
- Driver-supported CUDA: 12.3
- PyTorch: 2.5.1+cu121
- Torchvision: 0.20.1+cu121
- CUDA available to PyTorch: yes
- FFmpeg CLI: not installed; clip extraction uses OpenCV locally

The training environment is isolated from the service environment. Exact installed packages are in `requirements-ml-lock.txt`.
