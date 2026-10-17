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

### Сквозное локальное демо

По умолчанию API запускает реальный video pipeline с детерминированным demo-predictor: декодирование `.mov`, временные окна, агрегация, таймкоды и JPEG настоящие, а scores пока тестовые.

```powershell
$env:PYTHONPATH = "src"
$env:FORTEBANK_API_KEY = "demo-secret"
$env:QAITU_INFERENCE_MODE = "mock"
uv run uvicorn fortebank_audit.api:app --host 127.0.0.1 --port 8123
```

Откройте `http://127.0.0.1:8123`, выберите локальный `.mov` и нажмите «Начать проверку». Интерфейс покажет этапы pipeline, резюме, таймкоды и 1–3 JPEG для каждого эпизода. Загруженный файл и временные evidence-файлы удаляются после анализа.

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
$env:QAITU_MODEL_BUNDLE = "artifacts\ml\bundles\baseline-repair-v2"
$env:QAITU_ALLOW_DEVELOPMENT_MODEL = "true"
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
