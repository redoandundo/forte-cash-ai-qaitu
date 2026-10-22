# Frontend and inference integration implementation plan

Date: 2026-10-17
Design: `docs/superpowers/specs/2026-10-17-frontend-inference-integration-design.md`
Branch: `feature/teamlead-ml-inference-integration`

## Objective

Connect the approved upload-first UI shell to the local FastAPI job lifecycle
and the integrated inference pipeline. The first runnable mode uses
`MockPredictor` for scores while retaining real video decode, temporal windows,
event aggregation and JPEG evidence extraction.

## Task 1 — Preserve and import the UI shell

Files:

- `web/index.html`
- `web/styles.css`
- `web/app.js`
- `README.md`
- `docs/superpowers/specs/2026-10-17-upload-first-ui-shell-design.md`

Steps:

1. Commit the already staged UI shell on `feature/teamlead-alt-testing` without adding unrelated untracked documents.
2. Cherry-pick that commit onto the integration feature branch.
3. Verify that only the expected five files entered the commit.

## Task 2 — Expose inference progress safely

Files:

- `src/fortebank_audit/inference/pipeline.py`
- `tests/inference/test_pipeline_smoke.py`

Steps:

1. Add an optional progress callback with stable stage identifiers.
2. Emit decode/windowing, inference, aggregation and evidence stages in monotonic order.
3. Keep the default call signature behaviour unchanged for existing callers.
4. Add a focused test that records stage order and confirms no path/frame data enters progress messages.
5. Run `pytest tests/inference -q`.

## Task 3 — Build the teamlead runtime adapter

Files:

- `src/fortebank_audit/job_runtime.py` (new)
- `tests/api/test_job_runtime.py` (new)

Steps:

1. Parse `QAITU_INFERENCE_MODE`, bundle path and development-model opt-in.
2. Build and cache either `MockPredictor` or an explicitly supplied canonical ONNX bundle.
3. Convert internal events to the required public `violation|no_violation` schema.
4. Read evidence JPEGs into base64 and reject missing/invalid evidence.
5. Produce a deterministic internal summary and runtime metadata for the UI.
6. Sanitize all public technical errors.
7. Test mock output mapping, missing evidence, invalid configuration and fail-closed ONNX selection.

## Task 4 — Replace the legacy API analyzer

Files:

- `src/fortebank_audit/api.py`
- `tests/api/test_api.py` (new)
- `pyproject.toml` only if a reproducible test dependency is missing

Steps:

1. Keep the exact required POST and GET routes and `X-API-Key` dependency.
2. Replace the old job runner with the runtime adapter.
3. Track internal stage, percent, runtime mode and summary separately from the public response.
4. Add authenticated `/internal/api/v1/jobs/{job_id}/progress` for the bundled frontend.
5. Preserve upload cleanup after success or failure.
6. Ensure public failed responses contain no absolute path or traceback.
7. Test invalid key, invalid extension, upload accepted, processing, done, failed and not-found.

## Task 5 — Connect the existing frontend

Files:

- `web/app.js`
- `web/index.html`
- `web/styles.css`

Steps:

1. Replace the demo timeout with POST upload and authenticated polling.
2. Poll the public job and internal progress endpoints until done, failed or five-minute timeout.
3. Render backend stage, percent and runtime mode.
4. Render all events and all 1-3 evidence images safely with DOM APIs.
5. Add a marker rail positioned from event timecode/video duration.
6. Seek the local preview when a marker, timecode or event card is clicked.
7. Render deterministic summary and clear/no-violation/failed states.
8. Keep demo state controls accessible only through an explicit debug query flag or remove them from the normal path.

## Task 6 — End-to-end verification

Files:

- `tests/integration/test_api_inference_frontend.py` (new if API tests do not already cover the path)
- `README.md`
- `docs/handoffs/integration-status.md`

Steps:

1. Run compile checks and focused ML, inference, API and integration suites.
2. Run the complete `pytest -q` suite and `pip check`.
3. Start the local API with mock mode and a non-secret demo key.
4. Open `http://127.0.0.1:8123/` in a real browser.
5. Upload `Х-10.mov`, observe stages, inspect event cards/evidence, and click a timecode.
6. Force one technical failure and verify the UI does not show `no_violation`.
7. Record only observed commands/results in the integration handoff.
8. Commit implementation in small ownership-aware commits.

## Expected commit sequence

1. `feat(web): preserve upload-first demo shell`
2. `feat(inference): expose pipeline progress stages`
3. `feat(api): connect jobs to inference runtime`
4. `feat(web): render live inference jobs`
5. `docs: record frontend inference smoke`

## Safety checks before every commit

- `git diff --check`
- explicit `git add` paths only
- no videos, JPEGs, weights, secrets, `.env`, cache or absolute user paths
- technical errors remain `failed`
- existing user changes outside the named files remain untouched
