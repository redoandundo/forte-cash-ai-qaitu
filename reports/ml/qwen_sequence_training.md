# Qwen sequence adapter: completed local experiments

Date: 2026-10-18. Base revision before the work:
`693320432d56119b461a3f34360a90a257cdd76f`.
API integration is separately committed as `48c1739`.

## Result and release decision

Two complete QLoRA experiments ran locally on RTX 5050 Laptop 8 GB. The selected
adapter is **experimental, not release-ready**. It does not meet the quality gate
and is not enabled by the standard API launcher. No perfect-accuracy claim is made.

The task is binary screening of a complete video using 16 chronological uniformly
sampled frames. Human video labels supervise A/B class-token likelihoods. No
invented frame observations, regulatory labels, or legal conclusions were used.
The user's detailed descriptions remain private for subsequent grounding work;
they were not used as generated frame-level targets in these experiments.

There are 40 unique sources from 42 files, with 24 train, 7 validation and 9
diagnostic-test sources. Known duplicates and related scenario groups stay in
one split. All sources were previously inspected during development, and the
diagnostic test was evaluated for both experiments; it is not an untouched blind
test. ONNX training overlap is not excluded. Threshold remains fixed at 0.5 for
the two-token ranking score, which is not a calibrated probability.

## Measurements

- Experiment 1: 3 epochs, 72 optimizer steps, learning rate 0.0001, batch size 1.
  Total preparation/training/evaluation time 160.57 s. Test TP 4, TN 0, FP 5, FN 0:
  collapsed to the positive class.
- Experiment 2: 6 epochs, 144 example passes and **36 optimizer steps** using
  gradient accumulation 4, learning rate 0.00002. Total time 249.20 s.
  Epoch 4 selected by validation macro-F1 then recall, with earliest epoch winning
  ties. Validation TP 4, TN 0, FP 3, FN 0; test TP 3, TN 1, FP 4, FN 1.
- Base Qwen on the diagnostic test: TP 0, TN 5, FP 0, FN 4.
- Experiment 2 test macro-F1: 0.4156, recall: 0.75. These are raw classifier
  results, not ONNX + recall-guard results.
- Peak PyTorch allocated memory: 3.134 GiB. Maximum observed parameter change
  over experiment 2: 0.00063517. Adapter SHA-256:
  `9d519f34dc7a900ede428a81af63f08ee58f030483a417c1309fba8d2ef380c3`.

The early experiment-2 private log called example passes `optimizer_steps`.
The exported manifest corrects this to 36 optimizer steps and 144 example passes;
the training script now records both counters independently.

## Reproduction and distribution

```powershell
.\.venv-ml\Scripts\python.exe scripts/ml/train_qwen_sequence.py `
  --output models/checkpoints/qwen-sequence-v2 `
  --epochs 6 --learning-rate 0.00002 --accumulation 4
```

Use a new output directory on repeated runs. Training requires private
`analysis/dataset_manifest.json`, human annotations and `split_config.json` under
`data/processed/qwen_sft/`, plus the pinned local Qwen snapshot. Dependency versions
are in `requirements-qwen.txt`; the experiment used additional packages under
ignored `tmp/qwen-training-deps` without changing the working inference environment.

Only the small adapter, its configuration, aggregate manifest and license are
distributed in `models/qwen-sequence-adapter/`, at the user's explicit request.
No video, frame, annotation, per-source evaluation or private review is included.
The Apache-2.0 Qwen base weights are downloaded at the pinned revision by the setup
script and checked by SHA-256.

## API verification

The API now uses the checksum-checked ONNX pipeline instead of the legacy missing
joblib baseline. Job creation, auth, failure sanitization, upload cleanup and
single-job admission have focused tests. Public event fields match the API
contract and JPEGs are validated by the local smoke test.

Three real videos, including a 142.53 s video, completed through the API:

- ONNX mode: 1.54 s, 5.35 s, 1.55 s.
- Reloaded sequence-adapter advisory mode: 19.09 s (including first model load),
  10.95 s, 3.20 s. All responses were `done` with valid JPEG evidence.

These are technical integration checks, not accuracy checks: the negative source
videos can still produce ONNX false positives. The advisory sequence adapter does
not remove them. The semantic Qwen path remains separately selectable and retains
the existing conservative rejection rules.

The final workspace test suite passed 71 tests (including two pre-existing local
review tests outside this publication). Final results are recorded in
`docs/handoffs/qwen-training-runtime.md`. Dependency availability was checked
with pip dry-run; a fresh-machine dependency installation was not claimed.

## Next step

Review frame-level evidence and use context covering document checks, object
transitions and exit preparation. Several procedures require tiny details or
events between the 16 sampled frames. More epochs on these same weak labels are
not a substitute for that evidence. Add new independent sources for final testing
and three-way uncertainty supervision before making Qwen authoritative.
