# Accepted ONNX inference verification

Date: 2026-10-17

Branch: `feature/video-inference`

ML source commit: `4a33bff`

## Outcome

The inference pipeline now consumes `mobilenet-temporal-pairsafe-v3` from the
accepted ML bundle. It validates the ONNX checksum, nested manifest, input and
output names/shapes, preprocessing, threshold, aggregation, and sampling
contract before inference. Decode, model, manifest, checksum, and evidence
errors propagate as technical failures; they are never converted to
`no_violation` or a positive finding.

The accepted decision path is:

```text
6 s window -> 4 RGB frames -> ImageNet normalization -> sigmoid
-> [0.25, 0.5, 0.25] edge-padded smoothing
-> peak -> threshold 0.37794335186481476
```

The accepted model was calibrated with rounded frame-index sampling. A first
run using nearest PTS changed one near-threshold negative into a false positive.
The production adapter therefore declares `opencv_frame_index_round` in the
manifest and reproduces that behavior explicitly instead of guessing it.

## Verification

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m fortebank_audit.inference.evaluation --selection reference
```

Observed on the same eight-source pair-safe holdout:

- confusion matrix: TN 3, FP 1, FN 1, TP 3;
- precision: 0.75;
- recall: 0.75;
- F1: 0.75;
- AUROC: 0.6875;
- false positives per negative minute: 0.458;
- timestamp MAE: 7.0 seconds;
- provider: CPU;
- split mismatch: false.

Portable inference tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/inference -q
```

The DINO/VideoMAE challenger test previously lived under `tests/inference` and
required PyTorch in the production environment. That experiment is being
reverted from this branch and handed to the ML developer through commit
`271400f`; it is not part of the accepted inference path.

## Qwen A/B

```powershell
$env:PYTHONPATH = "src"
.\.venv-ml\Scripts\python.exe -m fortebank_audit.inference.ab_evaluation `
  --qwen-model models\pretrained\qwen3-vl-2b-instruct `
  --output reports\inference\qwen_ab_holdout_summary.json
```

Observed on the same eight sources:

- baseline processing: 11.13 seconds;
- six candidate events sent to the experimental arm;
- Qwen processing: 59.95 seconds;
- five valid descriptions;
- one `VerificationOutputError`, retained as a technical experimental failure;
- zero public verdict, score, event, or timecode changes.

This run does not demonstrate a quality gain. The holdout has detector labels
and timecodes, but no blinded human labels for description usefulness or review
time. Qwen remains behind `--enable-experimental-qwen`, emits no confidence,
and cannot modify the public result. Promotion requires a measured human A/B
gain and a schema-failure policy agreed by the team lead.

## Known limits

- The detector score is a ranking score, not a calibrated probability.
- Nested out-of-fold macro F1 is 0.540 with a wide 95% interval; the model is a
  review aid, not an autonomous decision system.
- One false positive and one false negative remain on the eight-source holdout.
- The public API adapter is owned by the team lead and must map every inference
  exception to `status=failed` with no verdict.
- Generated evidence, video data, Qwen weights, and human-review memory remain
  local and ignored by Git.
