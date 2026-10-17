# ML/Inference integration status

Статус: API/frontend/mock end-to-end verified; real ML artifact acceptance blocked

## Входные версии

- Integration branch: `feature/teamlead-ml-inference-integration`
- ML source: `origin/feature/ml-model@5e6d383`
- ML integration commit: `f549302`
- Inference source: `origin/feature/video-inference@2f53ce1`
- Inference integration commit: `63cb5ba`
- Cross-component test commit: `9185e26`
- Inference progress callback commit: `5abf369`
- API/runtime integration commit: `88e49dc`
- Live frontend integration commit: `e5bf43e`
- Expected ML bundle: `baseline-repair-v2.zip`
- Expected archive SHA-256: `b9a8b2c809f7a6dc72e0bf655707fb9d2e430c906e573aa2f10ed4750d0ae90b`
- The expected archive is intentionally not stored in Git and was not available locally during integration.

## Выполненные команды

```powershell
python -m compileall -q src scripts
python -m pytest tests/ml -q
python -m pytest tests/inference -q
python -m pytest tests/integration/test_ml_inference_bundle.py -q
python -m pytest -q
python -m pip check
node --check web/app.js
git diff --check
```

Observed results:

- ML: `32 passed`.
- Inference: `32 passed`.
- Synthetic cross-component ONNX smoke: `1 passed`.
- API/runtime: `7 passed`.
- Full available suite: `73 passed`.
- Dependency check: `No broken requirements found`.
- JavaScript syntax check: passed.
- Diff whitespace check: passed.

## Проверки

- Contract validation: passed on a synthetic canonical fixed-batch bundle; a development bundle is rejected unless explicit opt-in is set.
- ONNX smoke test: passed with a real ONNX Runtime CPU session and fixed input `[1, 3, 4, 160, 160]`.
- PTS/tail behaviour: passed; a nine-second synthetic video produced three deterministic windows.
- Short positive video: passed on the synthetic nine-second fixture; result was `possible_violation` with one merged event.
- Long negative video: not run with the missing real ML bundle.
- Unicode filename: passed with `касса-тест.mov`.
- CPU fallback: passed with `CPUExecutionProvider` forced by `prefer_cuda=False`.
- Evidence validation: passed; three JPEG files were produced and verified on disk.
- Real `baseline-repair-v2` quality/latency: not verified because the archive was unavailable.
- Source-branch v3 ONNX weight: deliberately not imported; repository policy requires model weights to be transferred outside Git. ONNX execution fails closed unless an explicit canonical bundle is supplied.
- Browser smoke on local `Х-10.mov`: passed at `http://127.0.0.1:8123`; upload returned `202`, public/progress polling returned `200`, one event at 10 seconds and three JPEG frames rendered, timecode navigation worked, browser console had zero errors.
- Cleanup after browser smoke: zero uploaded MOV files and zero generated evidence files remained in the configured temporary roots.
- Mock-mode boundary: decode, temporal windows, aggregation, evidence and API/UI are real; only predictor scores are deterministic demo values and the UI labels them `Demo model`.

## Производительность

- Provider in integration smoke: CPU.
- Processing time/realtime factor/RAM: not treated as a benchmark because the test uses a synthetic `ReduceMean` ONNX graph rather than the real ML artifact.
- GPU compatibility: code supports provider selection, but no CUDA benchmark was run in this integration environment.

## Решение

- Code compatibility between the current ML contract and inference adapter: yes.
- Ready for code review: yes.
- Ready for a local technical demo in explicit mock mode: yes.
- Ready for `develop`: no, pending exact real-bundle acceptance and team review.
- Blockers:
  1. The exact `baseline-repair-v2.zip` artifact is missing locally.
  2. The ML release record remains development-only/not promoted; reported quality gates are not sufficient for production approval.
  3. Representative long-negative and positive acceptance videos have not been run through the exact bundle.
- Required fixes and owners:
  1. ML engineer: transfer the exact archive outside Git without repacking it, preserving the expected SHA-256; if rebuilt, publish the new archive SHA, model SHA, manifest, preprocessing, thresholds, labels, metrics, handoff and checksum inventory together.
  2. Teamlead: verify the received archive checksum, run the integration test plus representative acceptance videos, and record measured CPU/GPU latency.
  3. ML engineer and teamlead: treat the current threshold/model as development-only until agreed quality gates pass; technical failures must remain errors, never `no_violation`.
