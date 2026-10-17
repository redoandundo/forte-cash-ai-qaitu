# Inference -> Teamlead handoff

Статус: ready for teamlead integration review

## Версия

- Branch: `feature/video-inference`
- Team-contract sync: `3978ae9`
- Accepted-model integration: `db5198e`
- DINO/VideoMAE production revert: `212d2f2`
- ML source commit: `4a33bff`
- Model version: `mobilenet-temporal-pairsafe-v3`
- Runtime/provider: ONNX Runtime, verified on CPU

## Реализовано

- `VideoReader`: bounded sequential decode; accepted adapter reproduces rounded
  frame-index sampling explicitly declared in the manifest.
- Windowing: 6-second windows, 2-second stride, four RGB frames.
- Predictor: validates artifact checksum, nested manifest, input/output contract,
  preprocessing, threshold, aggregation, sampling, and ONNX session shape.
- Aggregation: `[0.25, 0.5, 0.25]` edge-padded smoothing before peak selection
  and thresholding.
- Evidence: 1-3 local JPEG frames per candidate event.
- Failure semantics: exceptions propagate as technical failures; no inference or
  Qwen exception is converted into `no_violation` or a positive result.
- Meta/human-review layer from `322aa28` remains internal and never rewrites
  detector output or weights.
- Qwen from `3cd8f18` is now gated by `--enable-experimental-qwen`, describes
  visible JPEG content only, emits no confidence, and cannot change verdict,
  score, event, or timecode.
- DINO/VideoMAE commit `271400f` was reverted from this branch. The ML developer
  can inspect/cherry-pick that historical commit; it must not enter production.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Observed: `28 passed`.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m fortebank_audit.inference.evaluation --selection reference
```

Observed on the accepted eight-source holdout:

- confusion: TN 3, FP 1, FN 1, TP 3;
- precision/recall/F1: 0.75 / 0.75 / 0.75;
- AUROC: 0.6875;
- timestamp MAE: 7.0 seconds;
- false positives per negative minute: 0.458;
- split mismatch: false.

The first PTS-nearest run produced F1 0.667 because one negative crossed the
threshold. Root cause was sampling mismatch with the accepted ML reference.
After declaring/reproducing `opencv_frame_index_round`, the accepted metrics and
class decisions are reproduced.

## Qwen A/B

```powershell
$env:PYTHONPATH = "src"
.\.venv-ml\Scripts\python.exe -m fortebank_audit.inference.ab_evaluation `
  --qwen-model models\pretrained\qwen3-vl-2b-instruct `
  --output reports\inference\qwen_ab_holdout_summary.json
```

Same eight sources: baseline 11.13 seconds; Qwen 59.95 seconds over six candidate
events; five valid descriptions; one schema failure; zero public-output changes.
This does not prove a quality gain because no blinded human usefulness/time
labels exist. `promotion_allowed=false` remains mandatory.

## Контракты

- Canonical manifest: `models/model_manifest.json`.
- Artifact SHA-256:
  `e44cddb9a9cb259ca7fb68b024c5be7be591fcc623116c3f1fc918583a6bac91`.
- Threshold: `0.37794335186481476`.
- Aggregation: `three_window_weighted_smoothing_then_peak`.
- Detector score is a ranking score, not calibrated probability.
- Internal verdict is `possible_violation|no_violation`; teamlead's adapter owns
  the public `violation|no_violation` mapping only after successful inference.
- Any decode/model/evidence failure must produce public `status=failed`, with
  `verdict=null` and an empty violations list.

## Ограничения и blockers

- Nested out-of-fold macro F1 is about 0.540 with a wide confidence interval;
  this is not an autonomous decision system.
- The accepted holdout still contains one false positive and one false negative.
- Qwen had one invalid-schema response out of six event descriptions.
- The public API currently remains teamlead-owned and is not wired to this
  accepted inference adapter in this feature branch.
- PR #1 currently targets `main`; per team architecture the team lead should
  retarget or selectively integrate it into `integration/ml-inference`.

## Следующий шаг тимлида

1. Review commits `db5198e` and `212d2f2`.
2. Connect `analyze_video` through an API adapter that maps exceptions only to
   `failed`, sanitizes error text, and never fabricates a verdict.
3. Keep Qwen disabled in the public path.
4. Run the API integration test and one full demo smoke test before promotion.
