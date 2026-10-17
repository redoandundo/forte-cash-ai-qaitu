# ML to inference handoff

## Artifact contract

- ONNX: `models/best_model.onnx`
- Manifest: `models/model_manifest.json`
- Labels: `models/labels.json`
- Preprocessing: `models/preprocessing.json`
- Threshold: `models/thresholds.json`
- Checksum: `models/checksums.sha256`

The ONNX input is named `video` and has layout `NCTHW`, RGB float32, with shape `N x 3 x 4 x 160 x 160`. The output `logits` contains one binary logit per window. Apply sigmoid once.

## Temporal contract

1. Decode a six-second window.
2. Select four evenly spaced frames.
3. Resize to `160 x 160`.
4. Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
5. Advance by two seconds.
6. Smooth window probabilities with `[0.25, 0.5, 0.25]`, using edge padding.
7. Select the largest smoothed value and its window centre.
8. Return `violation` when the value is at least `0.37794335186481476`.

Do not use a raw maximum or the old threshold. Both caused avoidable false positives.

## Reference command

```powershell
python scripts/ml/infer_onnx.py "C:\path\to\video.mov"
```

Expected result fields:

```json
{
  "label": "violation",
  "score": 0.387,
  "threshold": 0.37794335186481476,
  "peak_time_sec": 17.0,
  "aggregation": "three_window_weighted_smoothing_then_peak"
}
```

`score` is a ranking score and is not a calibrated probability. Technical decode or inference errors must be returned as errors, never as `no_violation`.

## Verified smoke test

`Х-15.mov` produced `violation`, score `0.3871`, and peak `17.0 s` for an annotation near `15 s`. CPU ONNX runtime was about `2.01 s` on the development laptop.
