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

The current classifier is an experimental baseline for exercising the full pipeline. Its metrics are not yet strong enough for an autonomous decision system, and its event type is intentionally generic. Every positive result means only "possible observable violation" and requires human review.

Generated data stays under `data/raw/` and `artifacts/`, which are excluded from Git.

## Next milestone

Review the extracted frames, define observable event categories, and implement the smallest local baseline that produces a verdict, timestamp, and evidence frame.
