# Qwen visual verifier handoff

Date: 2026-10-17

## Delivered

- pinned public `Qwen/Qwen3-VL-2B-Instruct` manifest and SHA-256 verification;
- local-only downloader; model weights remain ignored by Git;
- lazy GPU backend with `device_map=auto` and low-CPU-memory loading;
- strict JSON contract for visible evidence, confidence, matched signals, and limitations;
- rejection of malformed JSON and invented signal identifiers;
- deterministic JSON and Markdown reporting;
- optional integration into the existing `meta_cli analyze` command;
- observable-signal catalogue that does not claim intent, guilt, fraud, or legal violation.

## Reference-machine smoke test

Hardware: NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM.

Input: `X-1.mov`, one first-layer candidate event, three evidence frames.

- ONNX scoring and evidence pipeline: 1.53 seconds on CPU;
- Qwen model load: 6.75 seconds;
- Qwen event verification: 8.13 seconds;
- Qwen peak allocated VRAM: 4.16 GiB;
- report output: `data/raw/reports/<run-id>.json` and `.md`;
- model stays resident when several events are checked in one run, so the load cost is paid once.

The first layer activated across the full clip and peaked at 6 seconds, while the supplied event is
near 22 seconds. The contextual layer correctly abstained because activation was diffuse. Qwen
reported visible review signals from the three generated frames, but that observation is not a
policy determination.

## Reproduce

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-vlm.txt
.\.venv-ml\Scripts\python.exe -m pip install -e . --no-deps
.\.venv-ml\Scripts\python.exe scripts\ml\download_qwen_verifier.py
.\.venv-ml\Scripts\python.exe -m pytest tests\inference -q
.\.venv-ml\Scripts\python.exe -m fortebank_audit.inference.meta_cli analyze `
  --video data\raw\open\videos\X-1.mov `
  --vlm-model models\pretrained\qwen3-vl-2b-instruct
```

## Required team decision

The repository still has no approved violation taxonomy. The committed signal catalogue contains
only visible cues. A product or legal owner must provide the mapping from those cues to a policy
rule before the report can call an event a violation. Until then the correct output remains a
review candidate with observable evidence and explicit uncertainty.
