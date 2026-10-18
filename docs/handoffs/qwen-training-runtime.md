# Qwen training and API runtime handoff

Branch: `feature/video-inference`. Commits:
- `48c1739`: API ONNX composition, authentication/error/concurrency checks.
- `65d8d52`: local QLoRA training and diagnostic evaluation.
- `cde00ef`: experimental adapter, checksums, installer and aggregate results.
- `1f01567`: event-level Qwen observations and limitations preserved in API dossier.

API changes are separate commits because API ownership belongs to the team lead.
No main/develop integration was performed. Pre-existing local review files and
edits to the older inference handoff were deliberately excluded.

## Run now

Update: Windows hosts with existing dependencies/models can double-click
`START_SITE.cmd`; `scripts/serve_site.py` selects a free port and enables semantic
Qwen, generates a persistent private access key and writes browser access links
under ignored `secrets/`. Testers need only a browser on the same reachable LAN.
No model is downloaded by this launcher. Public internet hosting is not configured.
Launcher/API checks: `pytest tests/api -q -p no:cacheprovider --basetemp tmp/pytest-site`
completed with 8 passing tests and one Starlette deprecation warning. Dossier text
is HTML-escaped before display and preserves line breaks.

Follow `docs/team/LOCAL_RUN.md`. For the intended ONNX -> Qwen -> dossier path:

```powershell
py -3.12 scripts/setup_local.py --qwen
$env:FORTEBANK_API_KEY = "your-private-random-key"
.\.venv-run\Scripts\python.exe scripts/run_local.py --qwen-semantic
```

ONNX supplies candidate intervals and JPEGs. Base Qwen verifies each candidate
using chronological frames. API description preserves observations, reasoning,
risk and limitations. Event peak timestamps still come from ONNX. The trained
whole-video binary adapter is NOT the semantic dossier model; it is optional,
experimental and fails its quality gate. It must not be represented as a trained
event-localization/dossier model. User explicitly authorized public distribution
of this specific 6.44 MB adapter after a privacy/quality warning. Videos, labels,
private reviews and the base snapshot remain outside Git.

## Observed verification

```powershell
.\.venv-ml\Scripts\python.exe -m pytest tests/api tests/inference tests/ml -q -p no:cacheprovider --basetemp tmp/pytest-release-final
```

71 passed in 5.10 s, one Starlette deprecation warning; includes two pre-existing
local review tests outside these commits. After the dossier change:

```powershell
.\.venv-ml\Scripts\python.exe -m pytest tests/api/test_dossier.py tests/inference -q -p no:cacheprovider --basetemp tmp/pytest-dossier-final
```

60 passed in 5.77 s. `git diff --check` passed. Setup/launcher/training modules
compiled successfully; dependency pip dry-run succeeded. A clean-machine full
installation was not run. Three local real-video API smoke checks passed in ONNX
and reloaded binary-adapter advisory modes, with schema/JPEG validation. These
smokes do not establish semantic dossier accuracy. See
`reports/ml/qwen_sequence_training.md` for measured times and classifier errors.

## Storage and next step

Jobs are in process memory, evidence is in ignored local files, weights are model
files; this is not a database-backed training service. Training annotations and
split configuration remain private JSON files under `data/processed/qwen_sft/`.

Next ML work: build reviewed event-context targets with grounded frame timestamps,
three-way decisions and explanations, then train/evaluate the semantic adapter.
Use new independent videos for final evaluation. Existing diagnostic results are
not a blind test; neither error-free behavior nor exact localization is proven.
