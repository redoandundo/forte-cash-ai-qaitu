# Model to inference contract

## Artifact

- model: `models/best_model.onnx`
- input: `video`, float32 RGB, `N x 3 x 4 x 160 x 160`
- output: `logits`, one binary logit per window
- activation: sigmoid exactly once
- preprocessing and temporal parameters: `models/preprocessing.json`
- threshold and aggregation: `models/thresholds.json`
- version and checksum: `models/model_manifest.json`, `models/checksums.sha256`

## Temporal inference

1. Decode the video once.
2. Evaluate six-second windows with a two-second stride.
3. Sample four evenly spaced frames per window.
4. Apply the exported RGB resize and ImageNet normalization.
5. Run ONNX in batches and smooth adjacent scores with weights `[0.25, 0.5, 0.25]`.
6. Group neighboring windows above the threshold into events.
7. Use the highest score in each group as `timecode_sec`.
8. Extract up to three JPEG frames at `peak - 1 s`, `peak`, and `peak + 1 s`.

## Interpretation boundary

The detector returns a ranking score and candidate time. It does not infer a confirmed action type, intent, guilt, or legal rule. Until a source safe multiclass model and rule mapping pass validation, inference returns the open set type `Иное визуально наблюдаемое событие` and states that the rule is not mapped.

Decode, inference, or mandatory evidence failure propagates as an error. It must not become `no_violation`.
