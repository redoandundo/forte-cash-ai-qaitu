# ML/Inference Integration Design

## Goal

Create a reviewable feature branch that combines the current team-owned ML and
inference implementations without touching the public API or the unfinished
frontend work. The result must prove canonical bundle compatibility in code and
tests while preserving the checked-in v3 model as the default fallback.

This integration must not claim that `baseline-repair-v2-development` is
production-ready. Its quality gate failed, and its real ZIP archive is not
available in the shared checkout.

## Inputs

- Team-lead base: `origin/feature/teamlead-core@fd23a1b`.
- ML source: `origin/feature/ml-model@5e6d383`.
- Inference source: `origin/feature/video-inference@2f53ce1`.
- Expected external archive:
  `artifacts/ml/bundles/baseline-repair-v2.zip`.
- Expected archive SHA-256:
  `b9a8b2c809f7a6dc72e0bf655707fb9d2e430c906e573aa2f10ed4750d0ae90b`.

## Integration Strategy

Use a selective integration rather than merging either feature branch in full.
The feature branches contain overlapping historical ML files and shared-root
changes, so a full merge would obscure ownership and make conflicts difficult
to audit.

The integration branch will:

1. take the repaired ML implementation, ML configs, ML tests, dependencies,
   reports, and ML handoff from `feature/ml-model`;
2. take the current inference implementation, inference tests/configuration,
   reports, and inference handoff from `feature/video-inference`;
3. preserve team-lead-owned contracts and API files from the team-lead base;
4. resolve overlapping ML paths in favour of the repaired ML branch;
5. retain the v3 fallback model and its complete sidecars until a promoted
   replacement exists;
6. keep the repaired development bundle external to Git and require explicit
   opt-in when it is used.

## Runtime Architecture

The default path remains:

```text
checked-in v3 bundle
  -> opencv_frame_index_round sampling
  -> dynamic-batch ONNX predictor
  -> accepted v3 aggregation
```

The optional development path is:

```text
baseline-repair-v2 bundle directory
  -> schema/checksum/release-status validation
  -> PTS-nearest sampling from stream start
  -> six-second windows, two-second stride, tail included
  -> fixed-batch-1 ONNX predictor
  -> sigmoid exactly once
  -> [0.25, 0.5, 0.25] smoothing
  -> adjacent-center event merge, maximum three events
  -> 1-3 JPEG evidence frames
```

The optional path must reject development bundles unless the caller supplies
`--allow-development-model`. A technical error propagates as a failure and is
never converted into `no_violation`.

## Contract Ownership

- `docs/contracts/**` stays identical to the team-lead base unless the team lead
  approves a separate contract change.
- The canonical ML manifest remains schema version 1 with nested `input` and
  `output` objects.
- Bundle-specific preprocessing, thresholds, labels, metrics, and checksum
  inventory travel together and are never mixed with v3 sidecars.
- Public API mapping is intentionally outside this integration commit. The API
  adapter must later map internal `possible_violation` semantics to the exact
  public verdict contract.

## Error Handling

- Missing, corrupt, path-escaping, or checksum-mismatched bundle files fail
  before inference.
- A development-only release fails without explicit opt-in.
- Missing/non-monotonic PTS, truncated decode, invalid ONNX shape, non-finite
  output, and evidence extraction failure remain technical failures.
- The v3 fallback remains usable if the optional development archive is absent.

## Verification

The feature branch is ready for review when all of the following pass:

1. `git diff --check`.
2. Python compilation for `src` and `scripts`.
3. ML focused tests.
4. Inference focused tests.
5. Existing project tests that do not require private videos or the missing
   external archive.
6. A synthetic canonical fixed-batch ONNX integration test covering bundle
   validation, PTS/tail windowing, inference, postprocessing, events, and JPEG.
7. A fallback smoke test proving the checked-in v3 path still loads.

Real-weight verification is a separate acceptance step. Once the archive is
transferred, the team lead must verify its archive SHA-256, extract it without
changing internal paths, run the documented command with
`--allow-development-model`, and record observed results. No quality or parity
claim from the ML report is upgraded to independently verified before that run.

## Commit Boundaries

The design document is committed first. Implementation follows in separate,
auditable commits:

1. integrate ML-owned files;
2. integrate inference-owned files and resolve compatibility conflicts;
3. add or adjust integration verification and handoff evidence.

Each commit stages explicit paths and records only commands actually executed.

## Out of Scope

- Promoting `baseline-repair-v2-development` over v3.
- Retraining or changing model architecture.
- Committing model weights, private videos, detailed labels, generated evidence,
  or local caches.
- Wiring or modifying the public API and frontend.
- Pushing, opening a pull request, or merging into `integration/ml-inference`
  without a separate user request.
