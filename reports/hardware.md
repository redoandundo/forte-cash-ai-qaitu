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
- Decoder: PyAV 16.0.1 with real PTS; a separate FFmpeg CLI is not required
- ONNX Runtime: 1.23.2 CPU, four intra-op threads

The training environment is isolated from the service environment. Direct dependencies are pinned in `requirements-ml-repair.txt`; the installed environment is recorded in `requirements-ml-lock.txt`. Install CUDA-compatible torch/torchvision before the other dependencies, using the handoff commands.
