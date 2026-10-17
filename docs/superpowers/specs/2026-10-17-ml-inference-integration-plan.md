# ML/Inference Integration Implementation Plan

## 1. Integrate repaired ML ownership

- Import the latest ML runtime, configs, scripts, focused tests, reports,
  dependency pins, and ML handoff from `origin/feature/ml-model@5e6d383`.
- Preserve team-lead contracts and the checked-in v3 fallback model.
- Verify the ML-only diff and commit explicit paths.

## 2. Integrate current inference ownership

- Import the latest inference runtime, configs, focused tests, reports, model
  fallback sidecars, and inference handoff from
  `origin/feature/video-inference@2f53ce1`.
- Resolve overlapping ML files in favour of the repaired ML branch.
- Preserve API/frontend ownership and commit explicit paths.

## 3. Verify the combined tree

- Add a repository integration test for a synthetic canonical fixed-batch ONNX
  bundle and synthetic Unicode video.
- Cover explicit development opt-in, PTS/tail sampling, one sigmoid,
  postprocessing, events, and JPEG evidence.
- Run `git diff --check`, compilation, ML tests, inference tests, the new
  integration test, and the broader test suite that does not require private
  inputs.
- Record actual results in the integration handoff and commit them.

## 4. External artifact acceptance

- Do not fabricate or commit the missing ML archive.
- Give the ML engineer the exact requested filename and SHA-256.
- After transfer, verify and smoke-test the real bundle in a follow-up commit.
