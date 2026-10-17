# ML improvement report v3

> Later evidence audit: this report compares versions on one eight-source holdout whose final threshold was selected on the same scores. Five-fold nested scenario-safe OOF now estimates macro F1 at `0.540` (`0.386–0.693` bootstrap 95% interval). See `nested_oof_summary.md` and `current_state_report.md`. The v3 artifact remains the integration baseline and is not production-promoted.

Date: 2026-10-17

## What was vulnerable

1. **Scenario leakage.** Visually matched positive and negative videos could be split between train and validation. This made validation dependent on background, furniture, and actors.
2. **Weak negative coverage.** Only the middle of each negative video was used, so the model did not learn normal actions throughout a recording.
3. **Noisy positive annotations.** Low-confidence hypotheses contributed as much as clearly visible events.
4. **Unstable maximum score.** A single noisy window could mark a long normal video as a violation.
5. **Biased checkpoint and threshold selection.** F2 alone rewarded predicting `violation` for almost every source.
6. **Too little temporal information.** The simple baseline used four frames, and the mandatory 16-frame R3D baseline had not been evaluated.

## Fixes

- Added `scenario_group` and kept all 13 contrastive pairs in one fold.
- Preserved SHA-256 duplicate grouping for `Х-4/Х-47` and `Х-5/Х-48`.
- Expanded the clip set from 41 to 79 clips: 22 positive and 57 negative windows sampled at 25%, 50%, and 75% of each negative source.
- Added per-source balancing and annotation-confidence weights: high `1.0`, medium `0.75`, low `0.5`.
- Added temporal jitter, perspective, lighting, JPEG, blur, and sensor-noise augmentation.
- Implemented R3D-18 frozen and last-stage fine-tuning experiments with 16 frames.
- Changed checkpoint selection to macro F1 with AUROC tie-breaking.
- Changed full-video threshold selection to macro F1, then recall and precision.
- Replaced raw maximum aggregation with weighted smoothing over three adjacent windows.
- Exported the selected model again and verified ONNX parity and complete-video inference.

## Before and after

| Metric on complete validation videos | Previous | Improved |
|---|---:|---:|
| Split | SHA-safe only | SHA-safe and scenario-pair-safe |
| Precision | 0.667 | **0.750** |
| Recall | 1.000 | **0.750** |
| Macro F1 | 0.733 | **0.750** |
| AUROC | 0.500 | **0.688** |
| False positives/minute | 1.014 | **0.458** |
| Timestamp MAE | 8.75 s | **7.00 s** |
| Within ±5 seconds | 25% | **50%** |
| p95 runtime | 6.52 s | **3.76 s** |

The new confusion matrix is `[[3, 1], [1, 3]]` on eight complete pair-safe validation sources.

## Model comparison

| Model | Frames | Complete-video macro F1 | AUROC | Peak VRAM | Decision |
|---|---:|---:|---:|---:|---|
| MobileNetV3 temporal v3 | 4 | **0.750** | **0.688** | 53 MB | selected |
| R3D-18 last-stage fine-tune | 16 | 0.564 | 0.438 | 702 MB | rejected |
| R3D-18 frozen | 16 | 0.333 at clip-selected threshold | 0.313 source AUROC | 405 MB | rejected |
| X3D-XS pair-safe | 4 | not promoted to full-video evaluation | 0.375 source AUROC | 116 MB | rejected |

R3D-18 and X3D learned the small training set but ranked matched negative sources above positives. The compact MobileNet generalized better and remained much cheaper.

## Selected artifact

- Model: `models/best_model.onnx`
- Version: `mobilenet-temporal-pairsafe-v3`
- SHA-256: `e44cddb9a9cb259ca7fb68b024c5be7be591fcc623116c3f1fc918583a6bac91`
- Input: `N x 3 x 4 x 160 x 160`, RGB float32
- Window: 6 seconds
- Stride: 2 seconds
- Aggregation: three-window weighted smoothing, then peak
- Threshold: `0.37794335186481476`
- ONNX parity maximum absolute error: `2.98e-8`

## Remaining risks

- Eight validation sources are too few for a stable production threshold.
- The threshold and reported metric use the same validation fold. Five-fold out-of-fold calibration remains the next required experiment.
- Scores occupy a narrow range, so the UI must not display them as calibrated probabilities.
- `Х-22` is a low-confidence label and remains the main false negative under the balanced threshold.
- `Х-35` is the main false positive and should be added to the next hard-negative mining round.
- The binary labels do not support reliable violation-type classification.

This artifact is a stronger hackathon screening model. It still requires human review and must not issue a legal conclusion.
