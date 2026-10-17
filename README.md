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

```powershell
uv sync
uv run python scripts/prepare_data.py "<PATH_TO_OPEN_ZIP>"
uv run python scripts/build_dataset_report.py
uv run python scripts/train_baseline.py
```

Запуск API:

```powershell
$env:FORTEBANK_API_KEY = "replace-with-a-random-key"
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

## Run and verify the ONNX inference pipeline

The repository already contains the selected ONNX artifact. The inference code
checks its SHA-256 and input/output contract before processing any video, uses
container timestamps (PTS), scores in one decode pass with bounded frame
buffering, and saves evidence frames under the ignored `data/raw/evidence/`
directory. Evidence extraction may perform an additional bounded decode pass.

```powershell
uv run pytest tests/inference -q
uv run python -m fortebank_audit.inference.benchmark `
  --video data/raw/open/videos/X-1.mov `
  --predictor onnx
uv run python -m fortebank_audit.inference.evaluation --selection reference
```

Use `--selection reference` when reproducing the published threshold report.
The current dataset manifest contains a different validation split; the
evaluator reports that mismatch explicitly instead of silently mixing results.

## Run the contextual review layer

The second layer does not rewrite model weights online. It stores privacy-safe
run summaries in a local ignored SQLite database, routes uncertain or diffuse
activations to human review, and uses only reviewed examples as context for
future decisions.

```powershell
uv run python -m fortebank_audit.inference.meta_cli analyze `
  --video data/raw/open/videos/X-1.mov
uv run python -m fortebank_audit.inference.meta_cli review `
  --run-id <run-id> `
  --outcome confirmed_positive `
  --corrected-timecode-sec 22 `
  --error-type localization_error
uv run python -m fortebank_audit.inference.meta_cli plan
```

The runtime database defaults to `data/raw/meta/experience.sqlite3` and is not
committed. Repeated runs of the same physical video count as one context item,
so replaying a file cannot manufacture experience.

## Run the local visual verifier and report

Qwen3-VL is an experimental, feature-flagged evidence-description tool. It sees
only the JPEG frames from existing candidate events and can describe visible
actions, blur, or occlusion for a human reviewer. It never changes the public
verdict, detector score, or timecode, and it does not emit a confidence value.

Weights, videos, evidence JPEGs, reports, and the review database stay outside
Git. Install the ML environment and download the pinned public model locally:

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-vlm.txt
.\.venv-ml\Scripts\python.exe -m pip install -e . --no-deps
.\.venv-ml\Scripts\python.exe scripts\ml\download_qwen_verifier.py
```

Run the full local path (ONNX candidate scorer, contextual routing, Qwen visual
verification, and deterministic report):

```powershell
.\.venv-ml\Scripts\python.exe -m fortebank_audit.inference.meta_cli analyze `
  --video data\raw\open\videos\X-1.mov `
  --enable-experimental-qwen `
  --vlm-model models\pretrained\qwen3-vl-2b-instruct
```

Reports are written under the ignored `data/raw/reports/` directory. The model
is loaded with `device_map=auto` and low-CPU-memory mode. On the reference RTX
5050 Laptop GPU, a measured three-frame smoke test loaded in 7.6 seconds,
generated 48 tokens in 4.2 seconds, and peaked at 4.08 GiB VRAM.

The signal catalogue in `configs/inference/observable_signals.json` contains
only visible review cues. The team lead must map those cues to an approved
business policy before any output can be called a policy violation.

The same accepted holdout can be run with the isolated Qwen arm using:

```powershell
.\.venv-ml\Scripts\python.exe -m fortebank_audit.inference.ab_evaluation `
  --qwen-model models\pretrained\qwen3-vl-2b-instruct
```

This A/B measures runtime, schema reliability, and verifies that detector
outputs remain identical. It cannot establish a quality gain until a blinded
human review provides labels for description usefulness or review-time savings.

The current classifier is an experimental baseline for exercising the full pipeline. Its metrics are not yet strong enough for an autonomous decision system, and its event type is intentionally generic. Every positive result means only "possible observable violation" and requires human review.

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
