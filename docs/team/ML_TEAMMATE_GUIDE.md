# ML teammate handoff

## Boundary

The ML area owns `analysis/`, `configs/model.yaml`, `models/`, `reports/`, `scripts/ml/`, and `src/fortebank_audit/ml/`. It does not change the public API, background jobs, web interface, legal language, original videos, or original workbook.

## Current result

Read `reports/training_report.md` first. The exported model is `models/best_model.onnx`; its preprocessing and decision threshold are adjacent JSON files. All processing is local.

## One-video command

```powershell
.\.venv-ml\Scripts\python.exe scripts\ml\infer_onnx.py "C:\path\to\video.mov"
```

## Rebuild order

1. Create `.venv-ml` and install `requirements-ml.txt` plus the CUDA PyTorch command from the training report.
2. Run `scripts/ml/audit_annotations.py`.
3. Run `scripts/ml/prepare_clips.py`.
4. Run both model experiments through `scripts/ml/train_video.py`.
5. Evaluate complete validation videos through `scripts/ml/evaluate_full_videos.py`.
6. Export and verify through `scripts/ml/export_model.py`.

Do not treat scene descriptions as official violation classes. Preserve grouping by source SHA-256 in every future split.
