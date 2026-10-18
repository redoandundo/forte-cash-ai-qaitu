# Qaitu — ForteBank Cash AI

Командный проект Qaitu для Datathon26. Мы решаем кейс ForteBank: создаём полностью локальный сервис, который анализирует `.mov` кассовой зоны, находит визуально наблюдаемые признаки возможного нарушения, возвращает тип события, таймкод, описание, нормативное обоснование и 1–3 JPEG-доказательства.

Система помогает человеку проводить проверку и не является окончательным юридическим или дисциплинарным заключением.

## Текущий прототип

В репозитории уже находятся:

- FastAPI backend с `X-API-Key`;
- `POST /api/v1/analyze` и `GET /api/v1/jobs/{job_id}`;
- простой web-интерфейс загрузки;
- локальный video analyzer;
- экспериментальный ONNX baseline;
- ML training/evaluation/export scripts;
- конфигурации, отчёты и model metadata.

Текущий классификатор нужен для проверки полного технического пути. Его метрики ещё не считаются качеством финального решения.

Подготовлено подключение локального Qwen через существующий интерфейс inference:
[команды приёмки и запуска](docs/team/QWEN_INTEGRATION_RUNBOOK.md),
[контракт API-адаптера](docs/contracts/semantic_integration_contract.md).
По умолчанию Qwen выключен; улучшенные веса и совместимый inference-код ещё
должны пройти интеграцию и проверку. Ручное применение правил доступно отдельно.

## Команда

| Роль | Зона ответственности |
|---|---|
| Тимлид | архитектура, API, jobs, GitHub, frontend, интеграция и demo |
| ML-инженер | данные, R3D-18/X3D, обучение, оценка, export и model handoff |
| Inference-инженер | декодирование, temporal windows, ONNX Runtime, aggregation, таймкоды, evidence и benchmark |

Полный контекст: [`knowledge/TEAM_CONTEXT.md`](knowledge/TEAM_CONTEXT.md).

## Быстрый старт Codex

1. Прочитать [`AGENTS.md`](AGENTS.md).
2. Прочитать [`GOALS.md`](GOALS.md) и [`PLANS.md`](PLANS.md).
3. Прочитать [`knowledge/PROJECT_CONTEXT.md`](knowledge/PROJECT_CONTEXT.md) и [`knowledge/current_state.yaml`](knowledge/current_state.yaml).
4. Прочитать contracts, handoffs и руководство своей роли.
5. Проиндексировать текущую ветку через `codebase-memory-mcp`.

## Установка и запуск прототипа

### Публичное техническое демо

Нужны интеграционное Python-окружение с ONNX Runtime, проверенный bundle v4 и
`cloudflared` из официального релиза Cloudflare. Модель выполняется локально;
HTTPS-трафик проходит через Cloudflare Tunnel. Это временный адрес для показа,
а не сервис с гарантированным временем доступности. Qwen пока выключен.

```powershell
./scripts/start_public_demo.ps1 -PythonExecutable python -CloudflaredExecutable cloudflared
```

Скрипт проверяет SHA модели и доступ без ключа, создаёт случайный ключ в
`secrets/public-demo-api-key.txt`, запускает API и туннель скрытыми процессами.
Ключ передаётся жюри отдельно от URL и вводится в «Ключ доступа к сервису».
Секрет не добавляется в Git. Адрес и PID находятся в ignored-файле
`tmp/public-demo/session.json`. При повторном запуске URL изменяется.
Компьютер должен оставаться включённым и подключённым к сети.

```powershell
$demo = Get-Content tmp/public-demo/session.json -Raw | ConvertFrom-Json
python scripts/smoke_public_api.py --base-url $demo.url --key-file secrets/public-demo-api-key.txt --output tmp/public-demo/smoke.json
./scripts/stop_public_demo.ps1
```

Smoke создаёт искусственное видео, проверяет HTTPS, ключ, схему, JPEG,
повреждённый файл и длительность запроса. Датасет и реальные кадры через
публичный туннель для этой проверки не отправляются. Остановка туннеля
немедленно делает временный адрес недоступным. Jobs и ручные сопоставления
хранятся в памяти до перезапуска API.

### Сопоставление нормативного правила

После анализа открыть у эпизода «Сопоставить правило после просмотра».
Проверяющий выбирает правило и подтверждает все условия по видео. Неполный
контекст не позволяет сопоставить правило. Подтверждение добавляет в карточку
и `description` публичного результата источник и отметку «Ручная проверка»;
автоматический verdict, тип события, таймкод и JPEG не изменяются.
Кнопка «Снять сопоставление» возвращает исходное описание.

Каталог: НБРК № 231, пункт 20 (приём/пересчёт и ранее принятые наличные),
пункт 22 (уход до завершения обслуживания). Источник и ограничения —
`legal/README.md`; реализация — `src/fortebank_audit/rules.py`.
Бинарный score не назначает правило. Автоматическое сопоставление по наблюдениям
Qwen не подключено и не объявляется проверенным.

Внутренние endpoints с тем же `X-API-Key`:

- `GET /internal/api/v1/rules` — каталог с обязательными наблюдениями.
- `PUT /internal/api/v1/jobs/{job_id}/events/{event_index}/rule-review` —
  `{ "rule_id": "CASH-04", "confirmed_facts": ["cashier_leaves_workplace", "client_service_in_progress", "full_context_visible"] }`.
- Для снятия: `{ "rule_id": null, "confirmed_facts": [] }`.

`event_index` начинается с 1. Запросы без ключа отклоняются; неизвестные/неполные
условия дают 422, отсутствующий эпизод — 404, незавершённый анализ — 409.
Обязательная внешняя JSON-схема жюри не меняется.

### Сквозное локальное демо с ONNX

Финальное демо запускается только с явно выбранным ONNX bundle. Если режим,
bundle, development opt-in или checksum заданы неверно, API завершается при
старте и не переключается на mock.

```powershell
$env:PYTHONPATH = "src"
$env:FORTEBANK_API_KEY = "demo-secret"
$env:QAITU_INFERENCE_MODE = "onnx"
$env:QAITU_MODEL_BUNDLE = "artifacts\ml\bundles\baseline-audited-v4"
$env:QAITU_ALLOW_DEVELOPMENT_MODEL = "true"
$env:QAITU_EXPECTED_MODEL_SHA256 = "00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d"
$env:QAITU_PREFER_CUDA = "true"
$env:QAITU_MAX_CONCURRENT_JOBS = "1"
uv run uvicorn fortebank_audit.api:app --host 127.0.0.1 --port 8123
```

Откройте `http://127.0.0.1:8123`, выберите локальный `.mov` и нажмите «Начать проверку». Интерфейс покажет этапы pipeline, резюме, таймкоды и 1–3 JPEG для каждого эпизода. Загруженный файл и временные evidence-файлы удаляются после анализа.

Для Windows можно одной командой проверить приватно переданный архив, установить
bundle и запустить тот же ONNX-режим:

```powershell
.\scripts\start_onnx_demo.ps1 `
  -ArchivePath "C:\path\to\baseline-audited-v4.zip" `
  -PythonExecutable ".\.venv\Scripts\python.exe"
```

Скрипт не подменяет отсутствующую модель: при неверном SHA-256, повреждённом
архиве или несовместимом bundle запуск завершается ошибкой до обработки видео.

Перед распаковкой приватно переданного архива проверьте его отдельно:

```powershell
(Get-FileHash .\baseline-audited-v4.zip -Algorithm SHA256).Hash.ToLower()
# c387bdea9bd80a124c5587fccfcec3243d2550658ee09ca6834acaa9e54a1241
```

`baseline-audited-v4-development` не прошёл quality gate и разрешается только
для технического hackathon-demo. Mock остаётся доступен исключительно при явном
`QAITU_INFERENCE_MODE=mock` для тестов интерфейса.

### Полный прототип

```powershell
uv sync
uv run python scripts/prepare_data.py "<PATH_TO_OPEN_ZIP>"
uv run python scripts/build_dataset_report.py
uv run python scripts/train_baseline.py
```

Запуск API:

```powershell
$env:FORTEBANK_API_KEY = "replace-with-a-random-key"
$env:QAITU_INFERENCE_MODE = "onnx"
$env:QAITU_MODEL_BUNDLE = "artifacts\ml\bundles\baseline-audited-v4"
$env:QAITU_ALLOW_DEVELOPMENT_MODEL = "true"
$env:QAITU_EXPECTED_MODEL_SHA256 = "00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d"
uv run uvicorn fortebank_audit.api:app --host 0.0.0.0 --port 8000
```

Web-интерфейс: `http://localhost:8000`.

Запрос:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/analyze `
  -H "X-API-Key: $env:FORTEBANK_API_KEY" `
  -F "file=@sample.mov"
```

ML-воспроизведение описано в [`reports/training_report.md`](reports/training_report.md) и [`docs/team/ML_TEAMMATE_GUIDE.md`](docs/team/ML_TEAMMATE_GUIDE.md).

## Ветки

```text
main
└── develop
    ├── feature/teamlead-core
    ├── feature/ml-model
    ├── feature/video-inference
    └── integration/ml-inference
```

Полные правила: [`docs/team/GITHUB_TEAM_ARCHITECTURE.md`](docs/team/GITHUB_TEAM_ARCHITECTURE.md).

## Требования Datathon26

- Frontend и публичный API обязательны.
- API защищается заголовком `X-API-Key`.
- Публичные статусы: `processing`, `done`, `failed`.
- Verdict: `violation` или `no_violation`.
- Результат должен появиться не позднее 5 минут.
- Видео, кадры и их описания нельзя отправлять в облачные AI-сервисы.
- Распознавание или идентификация лиц запрещены.
- После загрузки весов обработка не должна требовать интернета.

Безопасная выжимка ТЗ: [`docs/requirements/DATATHON26_CASE_REQUIREMENTS.md`](docs/requirements/DATATHON26_CASE_REQUIREMENTS.md). Точная API-схема: [`docs/contracts/api_contract.md`](docs/contracts/api_contract.md).

## Данные и публичный GitHub

Исходный PDF, `open.zip`, `.mov`, labels, clips, storyboards и подробные dataset-derived reviews остаются локальными. В публичный GitHub отправляются только код, безопасные contracts, конфигурации, воспроизводимые отчёты и командный контекст.
