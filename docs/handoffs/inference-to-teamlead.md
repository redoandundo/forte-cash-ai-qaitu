# Inference -> Teamlead handoff

Статус: ready for teamlead integration review

Integration note: inference code and tests were imported, but the source
branch's modified ONNX weight was not imported because repository policy keeps
weights outside Git. The integrated ONNX path therefore requires an explicit
canonical `--bundle`; the v3 measurements below remain source-branch evidence,
not a claim that the v3 artifact exists in this checkout.

## Версия

- Branch: `feature/video-inference`
- Team-contract sync: `3978ae9`
- Accepted-model integration: `db5198e`
- DINO/VideoMAE production revert: `212d2f2`
- ML source commit reviewed: `5e6d383` (runtime fix `5c9f095`, repair `744f5e8`)
- Source-branch fallback model: `mobilenet-temporal-pairsafe-v3` (external/not integrated)
- Optional development bundle: `baseline-repair-v2-development`
- Runtime/provider: ONNX Runtime, verified on CPU

## Реализовано

- `VideoReader`: bounded sequential decode; accepted adapter reproduces rounded
  frame-index sampling explicitly declared in the manifest.
- Windowing: 6-second windows, 2-second stride, four RGB frames.
- Predictor: validates artifact checksum, nested manifest, input/output contract,
  preprocessing, threshold, aggregation, sampling, and ONNX session shape.
- New ML bundle adapter: validates the complete canonical schema-v1 package and
  its checksum inventory. Development-only bundles require the explicit
  `--allow-development-model` flag and cannot replace the fallback silently.
- New ML sampling: real nearest PTS from stream start, tail window enabled,
  strict missing-PTS/truncated-decode failure, fixed ONNX batch size 1.
- Aggregation: `[0.25, 0.5, 0.25]` edge-padded smoothing before peak selection
  and thresholding.
- New ML postprocessing: fewer than three windows remain unsmoothed; active
  centers merge only within one stride; retain the three strongest events and
  return them chronologically.
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

Observed: `33 passed`.

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

The current v3 fallback was rerun after the bundle adapter changes and still
reproduces the same eight-source result above.

## Новый ML bundle

Expected archive outside Git:
`artifacts/ml/bundles/baseline-repair-v2.zip`, archive SHA-256
`b9a8b2c809f7a6dc72e0bf655707fb9d2e430c906e573aa2f10ed4750d0ae90b`.
Canonical model SHA-256:
`00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d`.

The archive is not present in Git or in this checkout, so its real weights could
not be executed here. Once the ML engineer transfers and extracts it, the
integration smoke is:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m fortebank_audit.inference.benchmark `
  --video "data\raw\open\videos\Х-15.mov" `
  --predictor onnx `
  --bundle artifacts\ml\bundles\baseline-repair-v2 `
  --allow-development-model
```

The opt-in is required because `metrics.json` marks the bundle
`development_only_not_promoted`. ML reports that its quality gate failed, so it
is available for technical A/B only. No ONNX model is selected implicitly in
the integration branch.

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

- Canonical manifest: `model_manifest.json` inside the explicitly selected bundle.
- Source-branch fallback artifact SHA-256 (not integrated):
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
- The repaired ML archive is intentionally outside Git and still must be handed
  to inference/integration before a real-weight end-to-end smoke can run.

## Следующий шаг тимлида

1. Review the bundle adapter commit together with `db5198e` and `212d2f2`.
2. Connect `analyze_video` through an API adapter that maps exceptions only to
   `failed`, sanitizes error text, and never fabricates a verdict.
3. Keep Qwen disabled in the public path.
4. Transfer the ML archive, verify its stated archive SHA-256, extract it, and
   run the command above. Do not promote it unless a future identical holdout
   demonstrates a measured quality gain.
5. Run the API integration test and one full demo smoke test before promotion.
