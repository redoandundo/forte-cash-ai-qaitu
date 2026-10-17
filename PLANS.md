# План проекта Qaitu

## Фаза 0 — Repository foundation

Статус: in progress

- Общие инструкции Codex.
- GitHub branch model.
- Team context, goals и contracts.
- Безопасный `.gitignore`.
- Handoff templates.

Готово, когда новый участник может открыть репозиторий и понять задачу, свою ветку, свою директорию и формат результата без устного объяснения.

## Фаза 1 — Data и contract freeze

Статус: partially complete

- Dataset manifest.
- Проверка меток и дубликатов.
- Preliminary scene families.
- Model/inference contract.
- API contract.
- Требования frontend и публичного URL.

Готово после подтверждения бинарной цели и фиксирования model I/O.

## Фаза 2 — Параллельные ML и inference

Статус: planned

ML-инженер:

- duplicate-safe split;
- R3D-18 baseline;
- X3D-XS experiment при возможности;
- metrics, export и handoff.

Inference-инженер:

- streaming reader;
- temporal windows;
- MockPredictor и OnnxPredictor;
- aggregation, evidence и benchmark.

Готово, когда обе feature-ветки имеют воспроизводимые tests и handoffs.

## Фаза 3 — ML/inference integration

Статус: planned

- Объединение в `integration/ml-inference`.
- Проверка checksum и manifest.
- ONNX smoke test.
- End-to-end запуск на коротком и длинном видео.
- CPU/GPU benchmark.

Готово, когда интеграционный результат воспроизводим и технические ошибки отделены от verdict.

## Фаза 4 — API и demo

Статус: planned

- `/api/v1/analyze`.
- Job status и артефакты.
- `X-API-Key` и точная JSON-схема жюри.
- Frontend загрузки и результата.
- Публичный URL сервиса и API.
- Финальные integration tests.
- Demo-сценарий Datathon26.
- Merge `develop -> main` и release tag.

Готово после smoke test целевой demo-сборки.
