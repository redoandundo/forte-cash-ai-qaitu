# ForteBank Video Audit

Local prototype for detecting visually observable signs of cashier-procedure violations in video. The first milestone audits the supplied dataset and extracts evidence frames around labeled timestamps.

## Current milestone

- safely extract `open.zip`;
- match videos to labels by numeric id;
- parse Russian timecode text;
- collect video metadata;
- extract five frames around each labeled event;
- explore the dataset in a reproducible notebook.

## Run

```powershell
uv sync
uv run python scripts/prepare_data.py "C:\Users\royze\Downloads\open.zip"
uv run python scripts/build_dataset_report.py
uv run python scripts/train_baseline.py
uv run jupyter lab notebooks/01_dataset_audit.ipynb
```

## Run the API

Set a private key and start the server:

```powershell
$env:FORTEBANK_API_KEY = "replace-with-a-random-key"
uv run uvicorn fortebank_audit.api:app --host 0.0.0.0 --port 8000
```

Submit a video and poll the returned job id:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/analyze `
  -H "X-API-Key: $env:FORTEBANK_API_KEY" `
  -F "file=@sample.mov"
```

Open `http://localhost:8000` for the upload interface.

## Run and verify the ONNX inference pipeline

The repository already contains the selected ONNX artifact. The inference code
checks its SHA-256 and input/output contract before processing any video, uses
container timestamps (PTS), scores in one decode pass with bounded frame
buffering, and saves evidence frames under the ignored `data/raw/evidence/`
directory. Evidence extraction may perform an additional bounded decode pass.

```powershell
uv run pytest tests/inference -q
uv run python -m fortebank_audit.inference.benchmark `
  --video data/raw/open/videos/X-1.mov `
  --predictor onnx
uv run python -m fortebank_audit.inference.evaluation --selection reference
```

Use `--selection reference` when reproducing the published threshold report.
The current dataset manifest contains a different validation split; the
evaluator reports that mismatch explicitly instead of silently mixing results.

## Run the contextual review layer

The second layer does not rewrite model weights online. It stores privacy-safe
run summaries in a local ignored SQLite database, routes uncertain or diffuse
activations to human review, and uses only reviewed examples as context for
future decisions.

```powershell
uv run python -m fortebank_audit.inference.meta_cli analyze `
  --video data/raw/open/videos/X-1.mov
uv run python -m fortebank_audit.inference.meta_cli review `
  --run-id <run-id> `
  --outcome confirmed_positive `
  --corrected-timecode-sec 22 `
  --error-type localization_error
uv run python -m fortebank_audit.inference.meta_cli plan
```

The runtime database defaults to `data/raw/meta/experience.sqlite3` and is not
committed. Repeated runs of the same physical video count as one context item,
so replaying a file cannot manufacture experience.

## Run the local visual verifier and report

Qwen3-VL is an optional second-stage verifier. It sees only the evidence frames
from candidate events, returns a validated JSON observation, and never changes
the first-layer score. The final JSON and Markdown reports are assembled by
deterministic code. They are decision-support artifacts, not findings of guilt,
fraud, intent, or a legal violation.

Weights, videos, evidence JPEGs, reports, and the review database stay outside
Git. Install the ML environment and download the pinned public model locally:

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-vlm.txt
.\.venv-ml\Scripts\python.exe -m pip install -e . --no-deps
.\.venv-ml\Scripts\python.exe scripts\ml\download_qwen_verifier.py
```

Run the full local path (ONNX candidate scorer, contextual routing, Qwen visual
verification, and deterministic report):

```powershell
.\.venv-ml\Scripts\python.exe -m fortebank_audit.inference.meta_cli analyze `
  --video data\raw\open\videos\X-1.mov `
  --vlm-model models\pretrained\qwen3-vl-2b-instruct
```

Reports are written under the ignored `data/raw/reports/` directory. The model
is loaded with `device_map=auto` and low-CPU-memory mode. On the reference RTX
5050 Laptop GPU, a measured three-frame smoke test loaded in 7.6 seconds,
generated 48 tokens in 4.2 seconds, and peaked at 4.08 GiB VRAM.

The signal catalogue in `configs/inference/observable_signals.json` contains
only visible review cues. The team lead must map those cues to an approved
business policy before any output can be called a policy violation.

The current classifier is an experimental baseline for exercising the full pipeline. Its metrics are not yet strong enough for an autonomous decision system, and its event type is intentionally generic. Every positive result means only "possible observable violation" and requires human review.

Generated data stays under `data/raw/` and `artifacts/`, which are excluded from Git.

## Next milestone

Review the extracted frames, define observable event categories, and implement the smallest local baseline that produces a verdict, timestamp, and evidence frame.
