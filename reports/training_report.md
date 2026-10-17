# Training report — current model

> Evidence update (2026-10-17): the `0.75` macro F1 below is the apparent result on one eight-source holdout. Five-fold nested scenario-safe OOF on all 40 sources estimates macro F1 at `0.540` with a 95% bootstrap interval of `0.386–0.693`. See [`ml/nested_oof_summary.md`](ml/nested_oof_summary.md) and [`ml/current_state_report.md`](ml/current_state_report.md).

Current model version: `mobilenet-temporal-pairsafe-v3`
Date: 2026-10-17

The detailed audit, before/after metrics, model comparison, remaining risks, and artifact contract are in [`ml/improvement_report.md`](ml/improvement_report.md).

## Result

- Pair-safe complete-video validation: 8 independent sources.
- Precision: `0.75`.
- Recall: `0.75`.
- Macro F1: `0.75`.
- AUROC: `0.6875`.
- Confusion matrix: `[[3, 1], [1, 3]]`.
- Timestamp MAE: `7.0 s`.
- False positives per negative minute: `0.4577`.
- p95 complete-video runtime: `3.76 s`.

## Artifact

- `models/best_model.onnx`
- SHA-256: `e44cddb9a9cb259ca7fb68b024c5be7be591fcc623116c3f1fc918583a6bac91`
- Input: RGB float32, `N x 3 x 4 x 160 x 160`.
- Six-second windows with two-second stride.
- Three-window weighted smoothing.
- Threshold: `0.37794335186481476`.

## Run

```powershell
python scripts/ml/infer_onnx.py "C:\path\to\video.mov"
```

The score is not a calibrated probability. Results require manual review.
