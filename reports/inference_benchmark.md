# ONNX inference implementation and verification

Date: 2026-10-17
Branch: `feature/video-inference`

## Outcome

The repository's existing MobileNetV3-small temporal ONNX model is now runnable
through a tested local inference pipeline. No model weights were downloaded or
retrained in this work. The implementation improves reproducibility, failure
handling, timestamp correctness, resource use, and evidence generation around
the existing model; it does not claim an accuracy gain.

## Delivered

- model, preprocessing, threshold-file checksum, and I/O contract validation before inference;
- CPU execution with optional CUDA provider fallback;
- sequential PyAV decoding with PTS-derived timestamps;
- deterministic 8-second windows, 2-second stride, and four RGB frames;
- ImageNet normalization and `NCTHW` batching matching the exported model;
- bounded frame buffering whose size depends on the batch, not video duration;
- calibrated sigmoid probability output without uncalibrated smoothing;
- deterministic thresholding and interval aggregation;
- one to three JPEG evidence frames with actual decoded timestamps;
- collision-resistant evidence runs and relative evidence paths;
- runtime, real-time factor, provider, model version, window count, and memory observations;
- a full-video evaluator that distinguishes the historical threshold-selection set from the current manifest split.

## Verification

Environment: Windows 11 build 26200, Python 3.12.14, PyAV 16.1.0,
OpenCV 4.14.0, NumPy 2.5.4, ONNX Runtime 1.31.0, and psutil 7.2.2 on
an AMD64 Family 25 CPU. ONNX Runtime exposed Azure and CPU providers; the
benchmark deliberately used CPU. Standalone `ffmpeg`/`ffprobe` executables were
not present, so decoding uses PyAV's bundled FFmpeg libraries.

`pytest -q` passes 18 portable tests, including generated Unicode-path video
decode, streamed nearest-frame sampling, model checksum/load, fixed-batch
rejection, predictor batch completeness, evidence-run isolation, window
boundaries, aggregation, and an end-to-end mock smoke test. The private
hackathon videos are used for benchmarks, not required for the unit suite.

The evaluator exactly reproduced the metrics stored in
`reports/mobilenet_noaug_full_video.json` when using its eight source IDs:

- confusion matrix: TN 2, FP 2, FN 0, TP 4;
- precision 0.667, recall 1.000, F1 0.800, F2 0.909;
- AUROC 0.500;
- false-positive videos per negative minute: 1.014;
- timestamp MAE: 8.75 seconds.

The current `analysis/dataset_manifest.json` assigns a different set of eight
sources to validation than the saved model report. Evaluation on that current
manifest split produces TN 0, FP 4, FN 0, TP 4 at the shipped threshold. This
split mismatch is now surfaced as `split_mismatch: true`; it must be resolved
before any new model-quality claim.

The sequential-PTS/OpenCV pipeline reproduces the rounded historical metrics,
but individual peak scores differ from the original OpenCV random-seek and
PyTorch-resize evaluator by up to about 0.00059 in the observed run. Because
scores lie close to the threshold, a parity fixture and threshold recalibration
on the exact production pipeline are required before promotion.

## CPU benchmarks

On this machine, after a warm local setup:

- `X-1.mov`: 25.93 seconds of video processed in 1.06 seconds, real-time factor 0.041, 9 windows;
- `Х-14.mov`: 142.53 seconds processed in 7.27 seconds, real-time factor 0.051, 68 windows.

The pipeline is substantially faster than real time on CPU for these samples.
The measured RSS delta includes ONNX Runtime allocator behavior and is an
observation, not a portable peak-memory guarantee.

Scoring is a single sequential decode pass. Evidence extraction is separate and
may decode again for each retained event; `max_events=5` bounds this cost.
Window plans and predictions scale linearly with window count, and a hard
`max_windows=100000` guard rejects pathological durations before planning.

## Honest failure examples

- `X-1.mov` is positive, but the model peaks at 6 seconds while the supplied
  event is near 22 seconds. This is a localization failure.
- `Х-14.mov` is negative, but every scored window is above the shipped
  threshold and the result is a false positive spanning almost the whole video.

These cases show why high recall on eight validation sources cannot be treated
as reliable human-action understanding.

## Safety boundary

The model detects weak visual correlations in a small supplied dataset. It does
not establish intent, identity, fraud, policy guilt, or a legal conclusion. A
positive output is a review candidate with evidence and confidence, never an
automatic sanction. Camera angle, occlusion, lighting, scene changes, unseen
cashier behavior, and label ambiguity remain outside the measured guarantee.

Smoothing and a separate `min_event_sec` filter are intentionally not enabled:
the delivered threshold was calibrated on raw eight-second window scores, and
adding them without recalibration would silently change the decision rule. The
eight-second model window already sets the minimum event support in this
baseline. This is an explicit waiver from the guide's suggested default.

## Next improvement gate

Do not tune the threshold on the same eight videos and call it progress. First:

1. freeze one canonical source-level split and version it with the model;
2. run group-aware cross-validation or out-of-fold scoring;
3. label and mine hard negative windows, especially the failure modes above;
4. compare every challenger on recall, false positives per negative minute,
   timestamp error, latency, and calibration;
5. promote only if the challenger improves the frozen evaluation without
   worsening the declared safety constraints.

The highest-value model experiment is hard-negative mining plus context-aware
temporal training. Video foundation embeddings or open-vocabulary anomaly
models are reasonable challengers only after this evaluation contract is fixed.
