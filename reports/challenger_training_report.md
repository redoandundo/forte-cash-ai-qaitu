# Local foundation-model challengers

Date: 2026-10-17

## Outcome

Two public pretrained challengers were downloaded, integrated, trained locally, and evaluated on
the same eight complete validation sources as the current MobileNetV3 baseline. Neither challenger
is promoted. The production ONNX model and its threshold remain unchanged.

The primary planned DINOv3 ConvNeXt-Tiny model is gated on Hugging Face. Access requires the user to
log in and explicitly agree to share contact information with Meta. The local backend and config are
ready, but the agent did not accept this agreement or authenticate on the user's behalf.

## Local environment

- Windows, Python 3.12.14;
- NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB;
- PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128;
- Transformers 5.20.1;
- CUDA available, compute capability 12.0.

Downloaded weights under ignored `models/pretrained/`:

- `MCG-NJU/videomae-base-finetuned-kinetics`, model.safetensors SHA-256
  `065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73`;
- `facebook/dinov2-small` as the public fallback for gated DINOv3, model.safetensors SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.

## Changes made

- added local-only DINO frame encoder plus trainable TCN/attention head;
- added local-only VideoMAE binary head;
- kept both foundation backbones frozen by default;
- added gradient accumulation for 8 GB GPUs;
- added explicit local weight paths to checkpoints;
- made dataset and clip manifests portable with project-relative paths;
- added a preflight failure when clip files are missing;
- made full-video evaluation load local foundation weights and use a configurable batch size.

## Complete-video results

Current MobileNetV3 baseline:

- precision 0.667, recall 1.000, macro F1 0.733;
- AUROC 0.500;
- false positives per negative minute 1.014;
- timestamp MAE 8.75 seconds.

VideoMAE-Base, frozen backbone, 769 trainable parameters:

- precision 0.571, recall 1.000, macro F1 0.564;
- AUROC 0.688;
- false positives per negative minute 1.373;
- timestamp MAE 3.50 seconds;
- peak training allocation 767 MB.

DINOv2-Small plus TCN, frozen backbone, 492,802 trainable parameters:

- precision 0.500, recall 1.000, macro F1 0.333;
- AUROC 0.125;
- false positives per negative minute 1.831;
- timestamp MAE 3.50 seconds;
- peak training allocation 338 MB.

## Interpretation

VideoMAE materially improved score ranking and temporal localization, but worsened the primary
false-positive measure and source-level macro F1. DINOv2 plus TCN localized some positive events
better but ranked negatives above positives. These are useful negative experiments: architecture
size was not the limiting factor.

The next high-value change is not another larger backbone. It is reviewed hard-negative mining and
a frozen source-level evaluation contract. The experience-store training gate remains in force:
at least 20 unique reviewed videos, including five positives and five negatives, before a promotion
claim or partial backbone unfreezing.

## Reproduction

```powershell
.\.venv-ml\Scripts\python.exe scripts\ml\train_video.py --config configs\videomae_base.yaml --model videomae_base --run-id videomae_base_aug --epochs 6
.\.venv-ml\Scripts\python.exe scripts\ml\evaluate_full_videos.py --checkpoint models\checkpoints\videomae_base_aug\best.pt --metrics models\checkpoints\videomae_base_aug\metrics.json --output reports\videomae_base_full_video.json --batch-size 1
.\.venv-ml\Scripts\python.exe scripts\ml\train_video.py --config configs\dinov2_tcn.yaml --model dinov2_small_temporal --run-id dinov2_small_tcn_aug --epochs 8
.\.venv-ml\Scripts\python.exe scripts\ml\evaluate_full_videos.py --checkpoint models\checkpoints\dinov2_small_tcn_aug\best.pt --metrics models\checkpoints\dinov2_small_tcn_aug\metrics.json --output reports\dinov2_small_tcn_full_video.json --batch-size 8
```
