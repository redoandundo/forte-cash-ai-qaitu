# Reproducible local API

## Your computer hosts the website

On the already configured Windows server, double-click `START_SITE.cmd`.
It reuses `.venv-ml` (or `.venv-run`) and the existing local Qwen, enables the
ONNX -> semantic Qwen -> dossier workflow, and opens the browser when ready.
There is no download or installation on the tester's computer.

Private access links are written to `secrets/tester-access.txt` (ignored by Git).
Send a same-network link to the tester. The link fills the access key in the
browser automatically; it is removed from the address bar after opening.
Keep these links private. The first link is for the server computer itself.
The launcher chooses an available port starting at 8000 and leaves existing
servers running. Close the launcher process to stop its server.

LAN access depends on network routing and the Windows firewall. These links
are not public internet addresses. Remote internet access requires an HTTPS
reverse proxy or a private VPN configured for the host; this launcher does not
configure either or change firewall rules. Keep the host awake during testing.

The instructions below are initial installation steps for a new server only.

Requires Python 3.12, an internet connection for initial package/model installation,
and enough disk space. Standard mode uses CPU ONNX. Qwen additionally requires a
compatible NVIDIA GPU/driver; the measured device was RTX 5050 Laptop 8 GB.
After installation, video processing is local and does not require internet.

## Standard mode

From the repository root:

```powershell
py -3.12 scripts/setup_local.py
$env:FORTEBANK_API_KEY = "your-private-random-key"
.\.venv-run\Scripts\python.exe scripts/run_local.py
```

Open http://127.0.0.1:8000 . Enter the same API key in the web interface.
The checked-in ONNX artifact and manifests are used. No training data or
`baseline.joblib` is needed. On Linux use `python3.12` and `.venv-run/bin/python`.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/analyze `
  -H "X-API-Key: $env:FORTEBANK_API_KEY" -F "file=@sample.mov"
curl.exe http://127.0.0.1:8000/api/v1/jobs/JOB_ID `
  -H "X-API-Key: $env:FORTEBANK_API_KEY"
```

Only one analysis runs at a time. A concurrent submission returns HTTP 429;
retry it after the active job finishes. Jobs live in process memory, so restarting
the server loses job history. Technical failures return `failed`, not a negative
verdict. Uploads are removed after processing; JPEG evidence remains locally under
ignored `data/raw/evidence/api/`. Do not expose this development server publicly
without the team's deployment configuration.

## Optional Qwen modes

The intended two-layer architecture is ONNX candidate detection followed by
Qwen event verification and a dossier. Use `--qwen-semantic` for this workflow.
Qwen receives chronological context around each ONNX event. The API description
includes its observations with timestamps, reasoning, risk and limitations;
the response also contains JPEG evidence. The event peak timestamp still comes
from ONNX: exact semantic localization is not established by this implementation.
The newly trained binary adapter does not perform this dossier task and is not
loaded by semantic mode. Semantic mode currently uses the pinned base Qwen.

```powershell
py -3.12 scripts/setup_local.py --qwen
```

This installs CUDA 12.8 PyTorch wheels and downloads the pinned ~4.25 GB Qwen base
snapshot if the existing weights fail checksum validation. A compatible NVIDIA
driver is required. Initial downloads may take much longer than inference.

Existing semantic verifier, with its recall protection:

```powershell
.\.venv-run\Scripts\python.exe scripts/run_local.py --qwen-semantic
```

New trained whole-video adapter, **experimental/advisory only**:

```powershell
.\.venv-run\Scripts\python.exe scripts/run_local.py --qwen-sequence --allow-experimental
```

Choose one Qwen mode, not both simultaneously. The adapter's failed quality gate
is recorded in its manifest. Without explicit experimental opt-in it refuses to
load. It can append a whole-video review to an existing event description, but it
does not remove ONNX events, invent timecodes, or turn missing ONNX detections
into located events. It is not a replacement for the structured semantic verifier.

The base Qwen weights are downloaded, not stored in ordinary Git. The small
trained adapter is included in `models/qwen-sequence-adapter/`. Raw videos,
annotations, private reports and caches are not in Git.

## Local verification

```powershell
.\.venv-run\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv-run\Scripts\python.exe -m pytest tests/api tests/ml tests/inference -q
```

Training scripts require the team's private data/annotations; running the API
does not. See `reports/ml/qwen_sequence_training.md` for measured limits.
