# ForteBank Video Audit

Local prototype for detecting visually observable signs of cashier-procedure violations in video. The first milestone audits the supplied dataset and extracts evidence frames around labeled timestamps.

## Current milestone

- duplicate-safe and scenario-safe source manifests;
- MobileNet video baseline exported to ONNX;
- production API inference through cached ONNX Runtime with single-pass video decoding;
- complete-video sliding-window inference with temporal smoothing;
- five-fold nested OOF evaluation with inner-fold threshold calibration;
- SQLite learning lab containing hypotheses, experiments, metrics, error cases, and promotion gates.

## Run

```powershell
uv sync
uv run python scripts/prepare_data.py "C:\path\to\open.zip"
uv run python scripts/build_dataset_report.py
uv run python scripts/train_baseline.py
uv run jupyter lab notebooks/01_dataset_audit.ipynb
```

Run the reproducible nested OOF audit after installing `requirements-ml-lock.txt`:

```powershell
python scripts/ml/run_nested_oof.py --epochs 1
python scripts/research/import_nested_oof.py
python scripts/research/lab.py dashboard
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

The interface opens the selected file locally, places clickable markers at candidate times, and shows up to three context frames per event. `confidence` is a model ranking score, not a probability. `no_violation` means that no evaluated fragment crossed the current threshold; it does not certify the employee or recording. See `docs/contracts/api_contract.md` for the exact output semantics.

The current classifier is an integration baseline. Its honest nested OOF macro F1 is `0.540` with a bootstrap 95% interval of `0.386–0.693`; it requires human review and is not production-promoted.

The production CPU path processes the 40 supplied sources (21.66 minutes of video) in 46.67 seconds. Mean latency is 1.17 seconds, p95 is 2.74 seconds, and the longest 142.5-second source takes 5.71 seconds after upload. The API uses `models/best_model.onnx` by default.

Generated data stays under `data/raw/` and `artifacts/`, which are excluded from Git.

## Next milestone

Train a scenario-pair ranking objective with verified hard negatives, then add temporal MIL and rerun the same nested OOF evaluation.
