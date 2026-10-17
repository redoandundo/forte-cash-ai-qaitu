# Frontend and inference integration design

Date: 2026-10-17
Status: approved direction, pending written-spec review
Owner: teamlead
Target branch: `feature/teamlead-ml-inference-integration`

## Goal

Turn the existing upload-first UI shell into one honest end-to-end demo:

1. the user selects a local `.mov` and sees a browser preview;
2. the frontend uploads it to the local API;
3. the UI shows observable processing stages;
4. the backend runs the real video reader, temporal windowing, aggregation and JPEG extraction;
5. the result screen shows verdict, clickable timecodes, event evidence and a concise deterministic summary;
6. technical failures remain `failed` and never become `no_violation`.

The first integrated run uses `MockPredictor` because the canonical
`baseline-repair-v2` bundle is not available. The UI must clearly label this as
demo prediction. Switching to ONNX later is a backend configuration change,
not a frontend rewrite.

## Chosen approach

Keep the current UI shell at `http://127.0.0.1:8123/` and replace its timer-based
state simulation with real API calls. The mock mode is not a browser fake: it
uploads and decodes the selected video, creates temporal windows, aggregates
events and writes evidence JPEGs. Only the model score source is mocked.

Rejected alternatives:

- keeping the existing browser-only state switcher would not test API or inference integration;
- blocking all work until the real bundle arrives would waste the remaining hackathon time and leave the product path unverified.

## Architecture

```text
Browser UI
  -> POST /api/v1/analyze (.mov, X-API-Key)
  -> GET /api/v1/jobs/{job_id}
  -> GET /internal/api/v1/jobs/{job_id}/progress
       |
       v
In-memory job service
  -> runtime selection: mock | onnx bundle
  -> VideoReader and deterministic temporal windows
  -> Predictor
  -> postprocessing and event aggregation
  -> 1-3 JPEG evidence frames
  -> public API adapter and deterministic summary
```

The required public endpoints and field names remain unchanged. Processing
details live on a separate internal progress endpoint so the jury contract is
not polluted with UI-only fields.

## Runtime selection

The backend reads configuration once at startup:

- `QAITU_INFERENCE_MODE=mock` enables the hackathon integration demo;
- `QAITU_INFERENCE_MODE=onnx` requires `QAITU_MODEL_BUNDLE` to point to a valid canonical bundle;
- development-only bundles additionally require an explicit opt-in setting;
- an invalid or missing ONNX bundle fails startup or the job explicitly; it never falls back silently to mock.

The loaded predictor is reused between jobs. The result includes an internal
runtime label so the frontend can show `Demo predictor` or the actual model
version without changing the public verdict schema.

## Processing stages

The internal progress record uses monotonic stages:

1. `upload_received` — upload saved and validated;
2. `video_decode` — metadata and frames are being read;
3. `temporal_windows` — time windows and sampling are prepared;
4. `model_inference` — the selected predictor scores windows;
5. `aggregation` — scores are smoothed and candidate events are merged;
6. `evidence` — 1-3 JPEG frames are extracted per event;
7. `summary` — the API result and UI summary are assembled;
8. `done` or `failed`.

An optional progress callback is added to the inference pipeline. It carries
only stage identifiers and contains no video, frame or absolute path. This is a
minimal cross-owner change and is committed separately from the API/frontend
adapter.

## Public result mapping

Internal inference values are adapted as follows:

- `possible_violation` -> public `violation`;
- `no_violation` -> public `no_violation`;
- each internal event becomes one `violations[]` item;
- the event peak becomes `timecode_sec`;
- each evidence JPEG is read locally and returned as base64;
- any missing mandatory evidence is a technical failure;
- exception text is sanitized before it reaches the public response.

Until verified event categories exist, the public type is the cautious
`Визуально нетипичное действие`. The description states only that the temporal
detector marked an episode for manual review. It does not claim theft, legal
violation, identity or loss.

## Frontend behaviour

The existing upload and workspace layout is preserved. Its behaviour changes:

- `Начать проверку` submits the selected file with the entered `X-API-Key`;
- the state switcher is removed from the normal demo path and may remain only behind an explicit debug flag;
- the progress card polls the internal progress endpoint and highlights the actual current stage;
- the required public job endpoint determines `processing`, `done` or `failed`;
- a successful result renders all events, not one hard-coded card;
- every event renders its timecode, description and 1-3 base64 JPEGs;
- clicking a timecode seeks the local browser preview to that second;
- a marker rail displays one red marker per event;
- `no_violation` says only that signs were not detected in this recording;
- `failed` shows a technical error and no verdict;
- the model badge clearly distinguishes `Demo predictor` from ONNX.

The first release does not generate persistent video clips. “Нарезка” is the
event list with 1-3 evidence frames plus direct playback at the corresponding
timecode. This avoids duplicating source videos and is sufficient for the
required contract.

## Summary

The summary is deterministic and derived only from the structured result. It
contains:

- result wording;
- number of candidate episodes;
- earliest and strongest available event timecodes;
- reminder that a person must review evidence;
- runtime mode and elapsed processing time when available.

It does not invent event categories, legal rules or facts that are not present
in the inference result.

## Error and data handling

- The API accepts only `.mov` up to the configured size limit.
- Uploads are deleted after success or failure.
- Evidence exists only for the job result lifecycle and is never committed.
- Video, evidence and descriptions are not sent to external AI APIs.
- API keys remain in environment/session storage and are not committed.
- Browser preview object URLs are revoked when the file changes or the page closes.
- A job timeout remains five minutes.

## Verification

Automated checks:

- API key rejection and valid upload acceptance;
- exact public response schema for processing, done and failed;
- mock end-to-end run on a generated Unicode `.mov`;
- progress stages are monotonic;
- event mapping returns 1-3 valid JPEG base64 values;
- missing evidence and predictor/decode failures return `failed`;
- timecode and evidence timestamps remain aligned;
- existing ML and inference suites continue to pass.

Browser smoke:

- open the existing shell through the API server;
- upload a real local `.mov`;
- observe stage changes;
- receive the result;
- click every event and confirm player seek;
- inspect all evidence images and summary;
- repeat with a forced failure.

## Commit boundaries

1. Import the already prepared UI shell without altering its visual direction.
2. Add the minimal inference progress callback and focused tests.
3. Add the teamlead-owned runtime/API adapter and contract tests.
4. Replace the frontend demo timer with API polling and dynamic result rendering.
5. Add run instructions and record the observed end-to-end smoke.

## Definition of done

- The existing frontend performs a real local upload/job flow.
- Mock mode processes the selected video through decode, windows, aggregation and evidence extraction.
- The UI displays real job stages, clickable event timecodes, 1-3 JPEG frames and a deterministic summary.
- The public API still matches `docs/contracts/api_contract.md`.
- Failures remain failures.
- Full automated tests pass.
- One real local `.mov` completes through the browser in under 300 seconds.
- The limitation “model scores are mocked until the external bundle is supplied” is visible and documented.

## Out of scope for this slice

- claiming real-model quality before the bundle is delivered;
- a second verifier or Qwen in the public path;
- generated video clip files;
- legal-rule generation;
- employee identity or face recognition;
- persistent shift history and weekly dashboards.
