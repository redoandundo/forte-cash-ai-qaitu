# ML/Inference integration status

Дата: 2026-10-18.

Статус: точный `baseline-audited-v4.zip` принят, проверен и подключён к полному
API/UI пути. Технический end-to-end на реальных видео подтверждён. Модель остаётся
development-only и не проходит quality gate.

## Входные версии

- Integration branch: `feature/teamlead-ml-inference-integration`.
- ML handoff: `origin/feature/ml-model@f891b213b8de5f923c136e4f228db08122553fe6`.
- ML code/artifact source: `aba56d31030aa9830fe2770e4aa9d54d2879ef24`.
- Inference source fix: `origin/feature/video-inference@405f06e`.
- Integrated decoder/Unicode fix: `875c09b`.
- Evidence/diagnostics integration: `da658c9`.
- API/runtime/UI integration: `806d10d`.
- Expected bundle: `baseline-audited-v4-development`.
- Expected archive: `baseline-audited-v4.zip`, 7,425,647 bytes.
- Expected archive SHA-256:
  `c387bdea9bd80a124c5587fccfcec3243d2550658ee09ca6834acaa9e54a1241`.
- Expected canonical `model.onnx` SHA-256:
  `00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d`.
- Release status: `development_only_not_promoted`; `quality_passed=false`.

Принятый архив имел точный размер 7,425,647 bytes и ожидаемый archive SHA-256.
После распаковки `model.onnx` имел ожидаемый SHA-256; случайные веса и
репозиторный historical `models/best_model.onnx` не использовались.

## Подключение

- `QAITU_INFERENCE_MODE` теперь обязателен: неявного default mock нет.
- ONNX mode требует `QAITU_MODEL_BUNDLE`; development bundle требует отдельный
  `QAITU_ALLOW_DEVELOPMENT_MODEL=true`.
- `QAITU_EXPECTED_MODEL_SHA256` позволяет зафиксировать именно согласованные
  веса поверх внутренней проверки полного `SHA256SUMS.txt`.
- FastAPI создаёт и проверяет runtime при startup. Ошибка bundle/model завершает
  startup и не переключает сервис на mock.
- `get_runtime()` кэширует predictor один раз на процесс.
- `QAITU_MAX_CONCURRENT_JOBS` ограничивает параллельный доступ к одной session;
  значение для demo — `1`.
- `/health` показывает mode, predictor class, model version, model SHA-256,
  release status и ONNX Runtime providers без локальных путей.
- Внутренний progress endpoint сохраняет полный window timeline и для событий
  `start_sec`, `end_sec`, `peak_time_sec`, `peak_score`, supporting count/indices.
  Публичная схема API не расширена этими данными.
- Технические decode/model/evidence ошибки дают `failed`, `verdict=null` и пустой
  список violations.

## Исправленные дефекты

- Unicode JPEG output: `cv2.imencode` и `Path.write_bytes`; JPEG повторно
  декодируется в регрессионном тесте.
- Unknown codec: отсутствие `stream.codec_context` проверяется до чтения
  metadata/decode и даёт `UnsupportedVideoError`.
- Evidence: до девяти кандидатов выбираются внутри границ события, сначала
  сохраняется ближайший к peak кадр, затем временно удалённые и визуально
  отличимые кадры. При одном уникальном кадре возвращается один.
- UI показывает десятые доли секунды для события и JPEG, поэтому разные
  timestamp больше не превращаются в подписи `09/10/10`.
- UI получает identity из `/health` до запуска и явно показывает `ONNX` либо
  `Demo model`.

## Выполненные проверки

```powershell
python -m compileall -q src scripts
python -m pytest tests/ml -q
python -m pytest tests/inference -q
python -m pytest tests/api tests/integration -q
python -m pytest -q
python -m pip check
node --check web/app.js
git diff --check
```

Фактический результат:

- ML tests: `32 passed`.
- Inference tests: `37 passed`.
- API + integration tests: `15 passed`, одно предупреждение о будущем изменении
  compatibility shim в `fastapi.testclient`.
- Полный доступный набор: `84 passed`, то же одно предупреждение.
- Dependency check: `No broken requirements found`.
- Python compile, JavaScript syntax и whitespace checks: passed.

## End-to-end matrix

| Сценарий | Ожидание | Фактический результат |
|---|---|---|
| Canonical synthetic ONNX, 9 s, Unicode input/output | `done`, ONNX identity, event, valid JPEG | passed через реальный HTTP API и UI; 1 event at 3.0 s, 1 уникальный JPEG |
| Видимый runtime до загрузки | ONNX/model/provider видимы | passed: `ONNX`, `integration-development`, `CPUExecutionProvider` |
| File picker | выбор по кнопке, не только drag/drop | passed в headed Chromium через native file chooser |
| Повторная загрузка | две независимые задачи | passed; обе `done` |
| Два одновременных запроса при limit=1 | без падения/смешения evidence | passed; обе задачи `done`, общий wall time 0.879 s на synthetic fixture |
| Unicode evidence directory | JPEG существует и декодируется | passed |
| Визуально одинаковые кадры | не создавать три дубликата | passed; возвращён один уникальный JPEG |
| Повреждённый MOV | `failed`, не `no_violation` | passed через API |
| Неизвестный codec | typed failure, не native crash | passed через API и отдельный subprocess regression test |
| Неверный expected model SHA | runtime startup/load rejected | passed |
| Повреждённый/отсутствующий model file | bundle/model load rejected | passed в bundle tests |
| CPU fallback | `CPUExecutionProvider` | passed |
| Browser console | no errors | passed: 0 errors, 0 warnings |
| Exact v4 identity | правильные bundle/version/SHA/provider | passed через `/health`: `OnnxPredictor`, `baseline-audited-v4-development`, exact SHA, `CPUExecutionProvider` |
| 42 локальных размеченных видео | все завершаются технически | passed: 42 `done`, 0 `failed`, все созданные JPEG декодируются |
| True positive через API/UI | событие и уникальные evidence | passed на `Х-11`: peak 19.0 s, JPEG 16.0/19.0/22.0 s |
| True negative | `no_violation` | passed на `Х-3` |
| Known false positive | ошибка качества остаётся видимой | confirmed на `Х-12`: peak 37.0 s |
| Known false negative | ошибка качества остаётся видимой | confirmed на `X-1`: размеченный момент 22 s, API вернул `no_violation` |
| Unicode filename | загрузка и обработка без ошибки | passed через API и headed browser на `Х-11.mov` |
| Повреждённый Unicode MOV | `failed`, не `no_violation` | passed через exact-v4 API |
| Integrated temporal shift `0/+4/-4` | `19/23/15 s` | passed через exact-v4 API; JPEG также сдвинулись `16/19/22 -> 20/23/26 -> 12/15/18` |

Synthetic fixture проверяет wiring, а не качество распознавания. Он не является
`baseline-audited-v4` и не используется в demo после проверки.

## Производительность

Машина текущей интеграции: Windows 11, Intel Core 5 120U, 12 logical CPUs,
ONNX Runtime 1.23.2. Доступен `CPUExecutionProvider`; `nvidia-smi` и CUDA provider
в этом окружении недоступны.

- Synthetic ONNX cold import + runtime creation: 1.1461 s.
- Повторный cached runtime lookup: 0.000002 s; тот же объект.
- Первый полный HTTP upload-to-result, 9 s fixture: 0.271 s; pipeline 0.178 s.
- Повторный запрос: 0.140 s; pipeline 0.081 s.
- Две ограниченно-параллельные задачи: обе `done`, общий wall 0.879 s.

Exact-v4 полный HTTP upload-to-result был измерен на 42 локальных видео:

- максимальная проверенная длительность видео: 142.533 s;
- максимальный wall time: 33.371 s;
- technical failures: 0;
- runtime provider: `CPUExecutionProvider`;
- ограничение 300 s выполнено для этого набора и этой машины, но не является
  гарантией для произвольного размера видео или другого оборудования.

## Temporal shift

Temporal shift повторён через текущий HTTP API на `Х-11`: исходный peak 19.0 s,
held-frame prefix +4 s дал 23.0 s, удаление первых 4 s дало 15.0 s. Оба результата
`done/violation`; evidence timestamps сдвинулись согласованно. Это подтверждает,
что интегрированный pipeline реагирует на положение содержимого, а не выдаёт
фиксированный пик около десятой секунды.

## Решение и блокеры

- Code compatibility: verified.
- Synthetic real-ONNX API/UI path: verified.
- Exact `baseline-audited-v4` API/UI path: verified.
- Ready to show as a technical hackathon demo: **yes**, на `127.0.0.1:8123`.
- Ready for production or autonomous legal conclusions: **no**.
- Quality promotion: **no**; baseline metrics остаются precision 0.500,
  recall 0.619, positive F1 0.553, macro F1 0.458, FP events/min 1.033,
  event recall ±5 s 0.364, peak MAE 7.133 s на reused development dataset.

`scripts/start_onnx_demo.ps1` проверил archive/model SHA-256, установил exact
bundle без молчаливой перезаписи и запустил API в явном ONNX-режиме. Перед
презентацией `/health` должен снова подтвердить exact SHA, `OnnxPredictor` и
активный provider; `8124` остаётся старым mock-прототипом и не используется.
