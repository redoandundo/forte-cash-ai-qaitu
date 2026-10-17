# ML training report

Date: 2026-10-17

Task: binary violation detection with weak timestamp localization

## Delivered artifacts

- `models/best_model.onnx` — selected MobileNetV3-small temporal classifier.
- `models/labels.json` — stable binary label mapping.
- `models/preprocessing.json` — ONNX input contract.
- `models/thresholds.json` — threshold selected on complete validation videos.
- `models/checksums.sha256` — model checksum.
- `configs/model.yaml` — data, augmentation, training, and export configuration.
- `analysis/dataset_manifest.json` — normalized sources and saved group-aware split.
- `analysis/clip_manifest.json` — reproducible clip inventory.
- `reports/experiments.csv` — experiment summary.

## Annotation audit

The original 42 rows contain 40 physical video sources. `Х-4/Х-47` is one positive SHA-256 group with both candidate events preserved at 57 and 68 seconds. `Х-5/Х-48` is one negative SHA-256 group. The split is assigned to the SHA-256 source, so no duplicate can cross train and validation. The result contains 33 training clips and 8 validation clips. No timestamp is outside its source video.

## Clip construction and augmentation

Positive clips are eight-second windows centred on the supplied timestamp and clamped to the video boundary. Negative clips use the centre of an independently labelled negative source. Training augmentation includes small affine camera changes, brightness, contrast and gamma variation, and JPEG quality 35–90. Validation preprocessing is deterministic. X3D uses four frames from the central 40% of the clip to approximate its short `4x12` temporal receptive field.

## Experiments

| Model | Validation level | Precision | Recall | F1 / macro F1 | AUROC | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| MobileNet temporal, no augmentation | annotated clips | 0.571 | 1.000 | 0.727 / 0.564 | 0.750 | 51 MB |
| MobileNet temporal, augmentation | annotated clips | 0.571 | 1.000 | 0.727 / 0.564 | 0.625 | 51 MB |
| X3D-XS, augmentation, corrected 4x12 sampling | annotated clips | 0.571 | 1.000 | 0.727 / 0.564 | 0.688 | 114 MB |
| **Selected MobileNet temporal** | **complete videos** | **0.667** | **1.000** | **0.800 / 0.733** | **0.500** | **51 MB** |
| X3D-XS | complete videos | 0.500 | 1.000 | 0.667 / 0.333 | 0.500 | 114 MB |

The complete-video threshold is 0.534097 and was selected only on the eight validation sources by F2. The selected confusion matrix is `[[2, 2], [0, 4]]`. The full-video AUROC of 0.50 shows that the score ranking is not yet reliable despite perfect recall at the chosen threshold. These numbers are prototype evidence, not a production quality claim.

## Localization and speed

For the selected model, timestamp MAE is 8.75 seconds; 25% of positive peaks fall within ±2 seconds and 25% within ±5 seconds. Mean full-video runtime is 2.73 seconds and p95 is 6.52 seconds on the measured validation videos using CUDA for PyTorch evaluation. The exported ONNX CPU smoke test on `Х-15.mov` took 1.36 seconds, predicted `violation`, and returned 16 seconds for an event annotated near 15 seconds.

## Model choice

MobileNet temporal was selected because it matched X3D clip F1, produced better clip AUROC (0.75 versus 0.688), had better complete-video macro F1 (0.733 versus 0.333), and used less peak VRAM. X3D was faster in the measured full-video PyTorch pass but failed to separate complete negative videos reliably.

## Export validation

- Format: ONNX opset 17
- Input: `float32` tensor `video`, layout `NCTHW`, shape `N x 3 x 4 x 160 x 160`
- Output: one logit per video window
- ONNX size: 3.83 MB
- PyTorch/ONNX maximum absolute output difference: `1.0430813e-7`
- SHA-256: `db81e57802e351248e96f265bb5beb0accfd2a47bb58d134a7c96c08ef8dcd01`

## Exact reproduction

From the repository root in PowerShell:

```powershell
uv venv .venv-ml --python 3.12
uv pip install --python .venv-ml\Scripts\python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-ml\Scripts\python.exe -r requirements-ml.txt
.\.venv-ml\Scripts\python.exe scripts\ml\audit_annotations.py
.\.venv-ml\Scripts\python.exe scripts\ml\prepare_clips.py
.\.venv-ml\Scripts\python.exe scripts\ml\train_video.py --model mobilenet_v3_small_temporal --run-id mobilenet_noaug --epochs 6 --no-augmentation
.\.venv-ml\Scripts\python.exe scripts\ml\train_video.py --model mobilenet_v3_small_temporal --run-id mobilenet_aug --epochs 8
.\.venv-ml\Scripts\python.exe scripts\ml\train_video.py --config configs\x3d_xs.yaml --model x3d_xs --run-id x3d_xs_4x12 --epochs 12
.\.venv-ml\Scripts\python.exe scripts\ml\evaluate_full_videos.py --checkpoint models\checkpoints\mobilenet_noaug\best.pt --metrics models\checkpoints\mobilenet_noaug\metrics.json --output reports\mobilenet_noaug_full_video.json
.\.venv-ml\Scripts\python.exe scripts\ml\export_model.py --checkpoint models\checkpoints\mobilenet_noaug\best.pt --metrics reports\mobilenet_noaug_full_video.json
```

Run the exported model on one `.mov`:

```powershell
.\.venv-ml\Scripts\python.exe scripts\ml\infer_onnx.py "C:\path\to\video.mov"
```

## Limits and next iteration

The validation set has only eight unique sources and the current run is one deterministic holdout rather than five-fold cross-validation. Temporal localization is weak and false positives remain high at roughly 1.01 per negative video-minute. Before publication, run five group folds, add scenario grouping for visually matched positive/negative scenes, mine hard negatives, and calibrate the full-video threshold from out-of-fold predictions. The model is suitable as a reproducible hackathon baseline and integration artifact, not as an autonomous compliance decision system.
