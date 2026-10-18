# Local Qwen adapter training pilot

Date: 2026-10-18. Base repository commit: `693320432d56119b461a3f34360a90a257cdd76f`.
Training additions remain uncommitted in `feature/video-inference`; this is a
separate ML experiment, not an ONNX bundle replacement or production integration.
Existing uncommitted review preparation and inference handoff changes were preserved.

## Observed result

- Audited 42 human annotation rows against 40 checksum-verified source videos.
  No binary label conflicts; annotated times are in range.
- Three explicitly timed human descriptions became pilot examples. The other
  39 annotation rows remain pending image/target alignment. No independent visual
  verification was performed by an external model.
- Six real optimizer steps completed on RTX 5050 Laptop GPU, training 1,605,632
  language-attention LoRA parameters with a frozen NF4 base and frozen vision tower.
- Pilot training/save/first-example check took 13.51 seconds, excluding imports,
  model loading and preparation. Peak PyTorch allocated memory: 3.017 GiB.
- Maximum adapter parameter change: 0.000603754015173763. Finite losses and
  gradients were checked at every step.
- Saved adapter: 6,438,952 bytes. SHA-256:
  `0ac680473e56fe0029106e7c3f1ebab1e8aac040bfd6d4d03713c09d1584a726`.
- A fresh process successfully reloaded the adapter and performed a forward pass.
  Reload loss is not a numerical parity assertion: preparation changes frozen
  parameter dtypes in the training process.
- Three focused tests passed. `git diff --check` passed. Pytest warned that its
  cache could not be written; test execution itself succeeded.

## Local storage and reproduction

No database is required. Private annotation JSON, temporal JPEGs, split manifest
and training JSONL live under ignored `data/processed/qwen_sft/`. Weights,
processor settings, training configuration, logs and reload check live under
ignored `models/checkpoints/qwen-qlora-pilot-v1/`. No media or weights were uploaded.

Dependencies were installed separately under ignored `tmp/qwen-training-deps`:
PEFT 0.21.2 and bitsandbytes 0.50.2. Existing inference packages were not upgraded.
The pilot uses the existing Python environment with PyTorch 2.11.0+cu128 and
Transformers 5.20.1. Model loading is local-only with checksum verification.

Executed commands:

```powershell
.\.venv-ml\Scripts\python.exe -m pip install --target tmp/qwen-training-deps --no-deps peft bitsandbytes
.\.venv-ml\Scripts\python.exe scripts/ml/build_qwen_sft.py
.\.venv-ml\Scripts\python.exe scripts/ml/train_qwen_adapter.py
.\.venv-ml\Scripts\python.exe scripts/ml/train_qwen_adapter.py --verify-only
.\.venv-ml\Scripts\python.exe -m pytest tests/ml/test_qwen_sft_training.py -q
git diff --check
```

For reproduction, install the recorded dependency versions and use fresh output
directories through `--output`; existing data/checkpoints are never overwritten.
Private annotations and the base snapshot are required inputs, intentionally not
part of the public repository. Validation/test sources are rejected by the optimizer.

## Limits and next step

This is a training plumbing proof, not a trained production verifier. Targets are
human timeline supervision, not approved per-frame review. Small details may be
lost at the pilot image resolution. There are no insufficient-evidence examples
yet, and no held-out quality evaluation was performed. Training loss is not accuracy.
All videos were previously inspected during development; the diagnostic split
does not establish an untouched external test or exclude prior ONNX exposure.

Before a substantive training run, align the remaining annotations with visible
frames, review full-sequence claims and ambiguous rule interpretations, add real
insufficient-evidence examples, and populate the fixed scenario-grouped validation
and test sets. Compare raw base/adapted answers separately from the existing recall
guard. The current adapter is not enabled in API or inference.
