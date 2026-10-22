# Полная GitHub-архитектура команды ForteBank Video Compliance

Версия: 1.0

Дата: 17 октября 2026 года

Владелец: тимлид

## 1. Что строим

Команда состоит из трех участников:

| Участник | Ответственность | Основная ветка |
|---|---|---|
| Тимлид | архитектура, API, job-система, конфигурация, финальная интеграция и demo | `feature/teamlead-core` |
| ML-инженер | данные, split, обучение, оценка, экспорт и model handoff | `feature/ml-model` |
| Inference-инженер | чтение видео, окна, ONNX Runtime, aggregation, таймкоды, evidence и benchmark | `feature/video-inference` |

Общая цель — полностью локальный сервис:

```text
.mov
  -> decode и timestamps
  -> temporal windows
  -> локальная видео-модель
  -> scores
  -> aggregation
  -> verdict + timecode + JPEG evidence
  -> API result
```

Исходные видео, кадры и содержимое банковского кейса не отправляются во внешние AI API.

## 2. Полная схема веток

```text
main                         стабильная демонстрационная версия
│
└── develop                  проверенная сборка следующей версии
    │
    ├── feature/teamlead-core
    │     API, jobs, config, storage adapters, final composition
    │
    ├── feature/ml-model
    │     train, eval, export, manifest, ML tests
    │
    ├── feature/video-inference
    │     decode, windows, predictor, aggregation, evidence
    │
    └── integration/ml-inference
          временное соединение двух feature-веток
```

### `main`

- Только стабильный релиз или рабочая demo-версия.
- Прямые push и force-push запрещены.
- Изменения приходят только PR из `develop`.
- Каждый merge отмечается tag, например `v0.1.0-demo`.

### `develop`

- Общая проверенная версия команды.
- В нее не складывают незавершенные эксперименты.
- Прямые push запрещены.
- Изменения попадают через PR после тестов.

### `feature/teamlead-core`

- Публичный API `/api/v1/analyze`.
- Job lifecycle и статусы.
- Общая конфигурация.
- Безопасная работа с путями и артефактами.
- Финальный composition root, который подключает inference.

### `feature/ml-model`

- Dataset manifest и split без утечки дубликатов.
- R3D-18 baseline.
- X3D-XS experiment при стабильном окружении.
- Metrics и error analysis.
- Export ONNX/TorchScript.
- `model_manifest.json`, preprocessing, thresholds и checksum.

### `feature/video-inference`

- `VideoReader` и реальные timestamps.
- `WindowGenerator`.
- `MockPredictor` и `OnnxPredictor`.
- Temporal aggregation.
- JPEG evidence.
- Benchmark CPU/GPU.
- Никакого обучения и публичного API.

### `integration/ml-inference`

- Создает тимлид от актуального `develop`.
- Сюда направляются Draft PR из двух feature-веток.
- Здесь проверяется реальная модель внутри pipeline.
- Здесь не проводят новые независимые эксперименты.
- После успешного end-to-end теста создается PR в `develop`.
- После merge ветка удаляется и при следующем цикле создается заново.

## 3. Почему inference не является дочерней веткой ML

Git не хранит постоянную иерархию веток как папки. Если создать `feature/video-inference` от незавершенной `feature/ml-model`, второй участник получит всю нестабильность первого и будет зависеть от его переписывания истории.

Правильная зависимость:

```text
feature/ml-model -----------┐
                            ├── integration/ml-inference
feature/video-inference ----┘
```

Оба участника стартуют от `develop` и договариваются через контракт. Общая integration-ветка дает второму участнику файлы первого, не превращая их рабочие ветки в одну.

## 4. Модели и ответственность

### 4.1. Что использует ML-инженер

| Компонент | Статус | Зачем |
|---|---|---|
| R3D-18 pretrained | обязательно | простой воспроизводимый video baseline |
| X3D-XS pretrained | опционально | более легкий кандидат для локального inference |
| RT-DETR-R18 | только после решения | детекция людей или крупных объектов при подтвержденной необходимости |
| ByteTrack | только вместе с детектором | связывание объектов между кадрами |
| SmolVLM2-500M Video | research-only | проверка гипотез, не основной классификатор |

Первый production milestone — бинарное обнаружение возможного нарушения и временная локализация. Многоклассовая юридическая классификация не обучается на неподтвержденных типах.

### 4.2. Что использует inference-инженер

На первом спринте второй участник не скачивает pretrained R3D, X3D, RT-DETR, ByteTrack или SmolVLM.

Он устанавливает:

- FFmpeg/ffprobe;
- Python 3.11;
- PyAV или OpenCV;
- NumPy;
- ONNX Runtime CPU либо GPU;
- pytest;
- psutil.

И использует:

1. `MockPredictor` — пока модель еще обучается.
2. Маленький искусственный ONNX fixture — только для теста адаптера.
3. Экспортированный `model.onnx` от ML-инженера — после handoff.

Если победит R3D-18, pipeline запускает экспорт R3D-18. Если победит X3D-XS, pipeline запускает X3D-XS. Вся разница должна находиться в manifest и preprocessing, а не в hardcoded логике pipeline.

### 4.3. Что именно передает ML-инженер

```text
model.onnx или model.ts
model_manifest.json
preprocessing.json
thresholds.json
labels.json
metrics.json
SHA256SUMS.txt
docs/handoffs/ml-to-inference.md
```

Минимальный manifest:

```json
{
  "model_name": "r3d18_binary",
  "model_version": "0.1.0",
  "artifact": "model.onnx",
  "sha256": "<64 hex characters>",
  "runtime": "onnxruntime",
  "input_name": "video",
  "input_layout": "NCTHW",
  "input_shape": [1, 3, 16, 112, 112],
  "dtype": "float32",
  "color": "RGB",
  "window_sec": 6.0,
  "stride_sec": 1.0,
  "output_name": "violation_score",
  "output_semantics": "probability_of_possible_violation",
  "threshold_file": "thresholds.json",
  "preprocessing_file": "preprocessing.json"
}
```

Pipeline обязан валидировать manifest и checksum до первого длинного запуска.

## 5. Где хранится модель

Обычный Git содержит только:

- исходный код;
- configs;
- manifest;
- preprocessing;
- thresholds;
- labels;
- metrics;
- SHA-256;
- инструкции воспроизведения.

Большие веса хранятся одним из способов:

1. приватный Git LFS;
2. приватный GitHub Release;
3. локальное командное хранилище с доступом только команды.

Вес не должен публиковаться в открытом репозитории без отдельного решения. Видео и evidence никогда не загружаются в GitHub.

Локальное расположение стандартизируется переменной:

```text
MODEL_DIR=<absolute local path>
```

Приложение не должно зависеть от пути конкретного пользователя.

## 6. База знаний для Codex

Это не SQLite и не бинарная база. Это текстовая, версионируемая база знаний:

```text
AGENTS.md
GOALS.md
PLANS.md
PROMPTS.md
knowledge/
├── PROJECT_CONTEXT.md
├── ARCHITECTURE.md
├── GLOSSARY.md
├── current_state.yaml
└── components.yaml
docs/
├── contracts/
│   ├── model_inference_contract.md
│   ├── model_manifest.schema.json
│   └── api_contract.md
├── decisions/
│   └── ADR-*.md
└── handoffs/
    ├── ml-to-inference.md
    ├── inference-to-teamlead.md
    └── integration-status.md
```

### Владение контекстом

| Файл | Кто обновляет |
|---|---|
| `AGENTS.md` | тимлид через reviewed PR |
| `GOALS.md`, `PLANS.md` | тимлид |
| `knowledge/current_state.yaml` | тимлид |
| `docs/contracts/*` | изменение предлагает участник, принимает тимлид |
| `docs/handoffs/ml-to-inference.md` | ML-инженер |
| `docs/handoffs/inference-to-teamlead.md` | inference-инженер |
| `docs/decisions/ADR-*` | автор решения, принимает тимлид |

Так центральный status-файл не будет конфликтовать в трех ветках.

### Что читает каждый Codex перед задачей

1. Корневой `AGENTS.md`.
2. `GOALS.md` и актуальную фазу в `PLANS.md`.
3. `knowledge/PROJECT_CONTEXT.md`.
4. `knowledge/current_state.yaml`.
5. Контракты, связанные с задачей.
6. Последний handoff upstream-компонента.
7. Фактический код через локальный `codebase-memory-mcp`.

Граф `codebase-memory-mcp` строится локально из текущей ветки. Его кэш не коммитится.

## 7. Рекомендуемое дерево репозитория

```text
.
├── .github/
│   ├── CODEOWNERS
│   ├── pull_request_template.md
│   └── workflows/
│       ├── tests.yml
│       └── contracts.yml
├── AGENTS.md
├── GOALS.md
├── PLANS.md
├── PROMPTS.md
├── README.md
├── .gitignore
├── pyproject.toml
├── knowledge/
├── docs/
│   ├── contracts/
│   ├── decisions/
│   ├── handoffs/
│   └── team/
├── src/
│   ├── api/
│   ├── core/
│   ├── jobs/
│   ├── ml/
│   └── inference/
├── configs/
│   ├── app/
│   ├── ml/
│   └── inference/
├── tests/
│   ├── api/
│   ├── ml/
│   ├── inference/
│   └── integration/
├── artifacts/
│   ├── README.md
│   └── manifests/
├── reports/
│   ├── ml/
│   ├── inference/
│   └── integration/
└── scripts/
```

Не создавать пустые сложные frameworks. Директории добавляются, когда в них появляется реальный контракт, код или инструкция.

## 8. Зоны изменения файлов

| Путь | Основной владелец |
|---|---|
| `src/ml/**` | ML-инженер |
| `configs/ml/**` | ML-инженер |
| `tests/ml/**` | ML-инженер |
| `reports/ml/**` | ML-инженер |
| `src/inference/**` | inference-инженер |
| `configs/inference/**` | inference-инженер |
| `tests/inference/**` | inference-инженер |
| `reports/inference/**` | inference-инженер |
| `src/api/**`, `src/jobs/**`, `src/core/**` | тимлид |
| `tests/api/**`, `tests/integration/**` | тимлид с участниками |
| `docs/contracts/**` | тимлид после review |
| `.github/**`, `AGENTS.md`, `PLANS.md` | тимлид |

Участник может предложить изменение чужой зоны, но обязан выделить его отдельным commit и описать в PR.

## 9. Инициализация пустого GitHub-репозитория

Это выполняет только тимлид после добавления базовых файлов.

```powershell
git status --short
git branch -M main
git remote add origin <GITHUB_REPOSITORY_URL>
git add -- .gitignore README.md AGENTS.md GOALS.md PLANS.md PROMPTS.md knowledge docs .github
git diff --cached --check
git commit -m "chore: initialize team architecture"
git push -u origin main

git switch -c develop
git push -u origin develop

git switch -c integration/ml-inference
git push -u origin integration/ml-inference

git switch develop
```

Если `origin` уже существует, `git remote add` не выполнять. Проверить:

```powershell
git remote -v
```

Нельзя использовать `git add .`, пока в рабочей папке находятся dataset, videos, legal dumps или временные файлы.

## 10. Настройки GitHub

### `main`

- default branch;
- require pull request;
- require status checks;
- запрет force push;
- запрет удаления;
- минимум один review, если это не блокирует хакатонный цикл.

### `develop`

- require pull request;
- require tests;
- запрет force push;
- merge разрешает тимлид.

### `integration/ml-inference`

- прямой force push запрещен;
- feature-ветки подключаются через Draft PR;
- финальный merge или закрытие выполняет тимлид.

### Merge policy

- Feature -> integration: обычный merge commit или squash по решению тимлида.
- Integration -> develop: merge commit, чтобы сохранить границу интеграции.
- Develop -> main: merge commit и release tag.
- Никогда не rebase опубликованную общую integration-ветку.

## 11. Создание рабочих веток

ML-инженер:

```powershell
git clone <GITHUB_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
git fetch origin --prune
git switch develop
git pull --ff-only origin develop
git switch -c feature/ml-model
git push -u origin feature/ml-model
```

Inference-инженер:

```powershell
git clone <GITHUB_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
git fetch origin --prune
git switch develop
git pull --ff-only origin develop
git switch -c feature/video-inference
git push -u origin feature/video-inference
```

Тимлид:

```powershell
git fetch origin --prune
git switch develop
git pull --ff-only origin develop
git switch -c feature/teamlead-core
git push -u origin feature/teamlead-core
```

## 12. Рабочий цикл участника

Перед работой:

```powershell
git status --short
git fetch origin --prune
git switch <OWN_FEATURE_BRANCH>
git merge --no-edit origin/develop
```

Если worktree грязный, сначала определить владельца изменений. Не использовать `reset --hard` и не удалять чужие файлы.

После небольшого завершенного этапа:

```powershell
git status --short
git diff --check
git add -- <ONLY_OWN_EXPLICIT_PATHS>
git diff --cached --check
git commit -m "<type>(<scope>): <result>"
git push
```

Примеры commits:

```text
feat(ml): add duplicate-safe dataset split
feat(ml): export r3d18 model manifest
feat(inference): add streaming video reader
feat(inference): add ONNX predictor adapter
test(inference): cover temporal aggregation boundaries
docs(handoff): publish ML artifact contract
```

## 13. Как соединяются ML и inference

### Этап A — параллельная работа

ML-инженер обучает baseline. Inference-инженер строит pipeline с `MockPredictor` по согласованному контракту.

### Этап B — ML handoff

ML-инженер:

1. Экспортирует модель.
2. Проверяет parity PyTorch -> ONNX.
3. Создает manifest и checksum.
4. Обновляет `docs/handoffs/ml-to-inference.md`.
5. Push своей ветки.
6. Открывает Draft PR в `integration/ml-inference`.

### Этап C — получение файлов вторым участником

После того как тимлид принял ML handoff в integration:

```powershell
git fetch origin --prune
git switch feature/video-inference
git merge --no-edit origin/integration/ml-inference
```

Так второй участник получает код, manifest и handoff первого. Сам вес модели загружается из согласованного приватного источника и проверяется по SHA-256.

### Этап D — end-to-end

Inference-инженер подключает ONNX, запускает smoke test и benchmark, затем открывает Draft PR:

```text
feature/video-inference -> integration/ml-inference
```

### Этап E — продвижение

После прохождения integration tests тимлид открывает:

```text
integration/ml-inference -> develop
```

После проверки API и demo:

```text
develop -> main
```

## 14. Правила model handoff

Handoff считается готовым только при наличии:

- commit SHA;
- model version;
- artifact location;
- SHA-256;
- manifest;
- preprocessing;
- thresholds;
- labels;
- input/output shapes;
- PyTorch/ONNX parity result;
- smoke-test command;
- known limitations.

Inference-инженер не должен угадывать normalization, color order, layout или threshold.

При несовпадении checksum, shape или manifest pipeline обязан завершиться технической ошибкой. Нельзя превращать ошибку модели в `no_violation`.

## 15. Точный master-prompt для Codex второго участника

Перед отправкой заменить только:

- `<GITHUB_REPOSITORY_URL>`;
- `<REPOSITORY_DIRECTORY>`;
- при необходимости GitHub username.

```text
Ты работаешь как inference-инженер проекта ForteBank Video Compliance. Твоя задача — создать и вести ветку feature/video-inference, реализовать локальный видео inference pipeline и подготовить его к соединению с моделью ML-инженера через integration/ml-inference.

Репозиторий:
<GITHUB_REPOSITORY_URL>

Локальная папка:
<REPOSITORY_DIRECTORY>

ВАЖНЫЕ ГРАНИЦЫ

- Ты не обучаешь модель.
- Ты не меняешь архитектуру ML-модели.
- Ты не реализуешь публичный API и job queue.
- Ты не коммитишь исходные видео, clips, JPEG evidence, веса, .venv, caches, secrets или абсолютные пути пользователя.
- Ты не отправляешь видео, кадры или модель во внешние сервисы.
- Ты не выполняешь force push, reset --hard или удаление чужих файлов.
- Ты не используешь git add .; добавляй только явные принадлежащие тебе пути.
- Ты не коммитишь напрямую в main, develop или integration/ml-inference.
- Ты можешь push только feature/video-inference.
- Не меняй git user.name или git user.email. Если identity отсутствует, останови commit и сообщи человеку точную проблему.

ПЕРВЫЙ ЭТАП — READ-ONLY PREFLIGHT

1. Проверь git status, current branch, remotes и доступные remote branches.
2. Если есть неожиданные незакоммиченные изменения, не изменяй их и сообщи владельцу.
3. Убедись, что origin/develop существует. Если его нет, остановись: базовую архитектуру должен сначала опубликовать тимлид.
4. Прочитай полностью:
   - AGENTS.md;
   - GOALS.md;
   - PLANS.md;
   - knowledge/PROJECT_CONTEXT.md;
   - knowledge/current_state.yaml;
   - docs/contracts/model_inference_contract.md;
   - docs/contracts/model_manifest.schema.json;
   - docs/handoffs/ml-to-inference.md, если он существует;
   - docs/team/INFERENCE_TEAMMATE_GUIDE.md.
5. Если codebase-memory-mcp доступен, проиндексируй текущую ветку и используй graph tools для поиска кода и зависимостей.
6. Проверь Python, ffmpeg, ffprobe и ONNX Runtime providers.
7. Кратко сообщи найденное состояние, предполагаемые файлы изменений и риски. Затем продолжай в рамках этого prompt.

СОЗДАНИЕ ВЕТКИ

Выполни безопасно:

git fetch origin --prune
git switch develop
git pull --ff-only origin develop

Если локальная feature/video-inference уже существует, переключись на нее. Если существует только origin/feature/video-inference, создай tracking branch. Если ветки нигде нет, создай:

git switch -c feature/video-inference

Опубликуй только эту ветку:

git push -u origin feature/video-inference

Не создавай main, develop или integration-ветку: это зона тимлида.

МОДЕЛЬНАЯ ПОЛИТИКА

В первые два часа не скачивай R3D-18, X3D, RT-DETR, ByteTrack или SmolVLM. Создай детерминированный MockPredictor и adapter OnnxPredictor.

Реальную модель ты получишь от ML-инженера как:
- model.onnx или model.ts;
- model_manifest.json;
- preprocessing.json;
- thresholds.json;
- labels.json;
- SHA256SUMS.txt.

Pipeline обязан определять preprocessing и shapes из этих файлов. Ничего не угадывай и не hardcode под конкретную архитектуру. Если модель или config отсутствует, MockPredictor должен работать, а реальный OnnxPredictor должен завершаться понятной ошибкой.

ТВОЯ ЗОНА ФАЙЛОВ

- src/inference/**
- configs/inference/**
- tests/inference/**
- reports/inference/**
- docs/handoffs/inference-to-teamlead.md

Не меняй src/ml, src/api, src/jobs, AGENTS.md, PLANS.md, .github или общие contracts без отдельного согласования. Если обнаружена проблема в контракте, зафиксируй предложение в handoff и сообщи тимлиду.

ОБЯЗАТЕЛЬНАЯ СТРУКТУРА

src/inference/
  contracts.py
  video_reader.py
  windowing.py
  preprocessing.py
  predictor.py
  aggregation.py
  evidence.py
  pipeline.py
  benchmark.py

tests/inference/
  test_video_reader.py
  test_windowing.py
  test_predictor.py
  test_aggregation.py
  test_evidence.py
  test_pipeline.py

reports/inference/
  inference_benchmark.md

ДВУХЧАСОВОЙ РЕЗУЛЬТАТ

00:00-00:10: preflight, branch, environment report.
00:10-00:30: streaming VideoReader, metadata, PTS timestamps, Unicode paths.
00:30-00:50: WindowGenerator, uniform sampling, short/exact/tail tests.
00:50-01:10: Predictor protocol, MockPredictor, OnnxPredictor с CUDA/CPU fallback.
01:10-01:30: smoothing, threshold, merge events, peak time.
01:30-01:45: 1-3 JPEG evidence около peak без выхода за duration.
01:45-02:00: analyze_video, pytest, benchmark и handoff.

Definition of Done:

1. .mov читается streaming без загрузки целого видео в RAM.
2. Используются реальные timestamps; frame_index/FPS только документированный fallback.
3. Windowing покрывает short, exact, overlap и tail cases.
4. MockPredictor детерминирован.
5. OnnxPredictor валидирует manifest, input shape и providers.
6. Aggregator возвращает events и peak_time_sec.
7. EvidenceExtractor сохраняет читаемые JPEG в ignored runtime folder.
8. analyze_video работает с MockPredictor на реальном видео.
9. pytest проходит.
10. Benchmark содержит duration, processing time, realtime factor, RAM/VRAM и provider.
11. Handoff содержит точную команду воспроизведения и список ожидаемых ML artifacts.

GIT-ДИСЦИПЛИНА

Делай небольшие проверяемые commits только после тестов. Рекомендуемые границы:

1. feat(inference): scaffold video pipeline contracts
2. feat(inference): add streaming reader and temporal windows
3. feat(inference): add mock and ONNX predictors
4. feat(inference): add event aggregation and evidence
5. test(inference): verify pipeline and benchmark handoff

Перед каждым commit:

git status --short
git diff --check
git add -- <только явные пути твоей зоны>
git diff --cached --check

После commit push только feature/video-inference. Не merge свою ветку самостоятельно.

ИНТЕГРАЦИЯ С ML

Пока ML artifact отсутствует, работай с MockPredictor.

Когда тимлид сообщит, что ML handoff принят в integration/ml-inference:

git fetch origin --prune
git switch feature/video-inference
git merge --no-edit origin/integration/ml-inference

Не разрешай содержательные конфликты в ML-коде самостоятельно. Если конфликт относится к чужой зоне, остановись и сообщи тимлиду.

После получения модели:

1. Проверь SHA-256.
2. Проверь manifest schema.
3. Выполни ONNX smoke test на одном окне.
4. Выполни end-to-end test на коротком .mov.
5. Сравни CPU/GPU providers.
6. Запиши фактические результаты в reports/inference/inference_benchmark.md.
7. Обнови docs/handoffs/inference-to-teamlead.md.

ЗАВЕРШЕНИЕ

Push feature/video-inference. Если GitHub CLI авторизован, создай Draft PR:

feature/video-inference -> integration/ml-inference

Название:

feat(inference): add local video analysis pipeline

Не выполняй merge PR.

В финальном сообщении укажи:
- branch;
- commit SHA;
- список измененных файлов;
- число прошедших тестов;
- benchmark;
- использованный provider;
- что работает с MockPredictor;
- что проверено с реальным ONNX;
- blockers;
- ссылку на Draft PR;
- точные следующие действия тимлида.

Продолжай до проверяемого результата, а не останавливайся после создания плана.
```

## 16. Что должно быть в `.gitignore`

```gitignore
.venv/
venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.idea/
.vscode/
.env
.env.*
*.log
tmp/
outputs/
runs/
evidence/
checkpoints/
data/open/videos/
analysis/clips/
*.mov
*.mp4
*.avi
*.mkv
*.pt
*.pth
*.ckpt
*.onnx
*.engine
*.plan
```

Если маленький ONNX fixture нужен тестам, добавить точечное исключение, например:

```gitignore
!tests/fixtures/tiny_identity.onnx
```

Только после проверки, что fixture не содержит реальных данных и имеет небольшой размер.

## 17. Минимальный PR checklist

```markdown
## Результат

Кратко описать измеримый результат.

## Зона изменений

- [ ] Изменялись только принадлежащие автору директории
- [ ] Изменения общих contracts выделены отдельно

## Проверки

- [ ] Unit tests
- [ ] Integration/smoke test
- [ ] git diff --check
- [ ] Нет видео, весов, evidence, secrets и абсолютных путей

## Контракты

- [ ] Input/output совместимы
- [ ] Manifest/schema обновлены при необходимости
- [ ] Breaking changes явно указаны

## Handoff

- [ ] Есть команда воспроизведения
- [ ] Есть фактические результаты
- [ ] Известные ограничения перечислены
```

## 18. Конфликты и аварийные ситуации

- Не использовать `git reset --hard` для исправления конфликта.
- Не делать force push в опубликованную общую ветку.
- Не принимать конфликт в чужой зоне наугад.
- Сначала сохранить `git status` и список конфликтных файлов.
- Владелец соответствующего компонента объясняет правильный вариант.
- После разрешения повторно запускаются unit и integration tests.
- Техническая ошибка не считается отрицательным результатом модели.

## 19. Полный маршрут релиза

```text
1. Тимлид публикует main, develop, integration/ml-inference.
2. ML-инженер создает feature/ml-model.
3. Inference-инженер создает feature/video-inference.
4. Оба работают параллельно по общему model contract.
5. ML открывает Draft PR в integration.
6. Тимлид принимает проверенный ML handoff.
7. Inference подтягивает integration, подключает artifact.
8. Inference открывает Draft PR в integration.
9. Команда запускает end-to-end tests.
10. Тимлид открывает integration -> develop.
11. API подключается и проверяется в develop.
12. Тимлид открывает develop -> main.
13. Создается tag v0.1.0-demo.
```

## 20. Финальная проверка тимлида

- [ ] Default branch — `main`.
- [ ] `main` и `develop` защищены.
- [ ] `integration/ml-inference` создана от `develop`.
- [ ] У каждого участника отдельная feature-ветка.
- [ ] В репозитории присутствует `AGENTS.md`.
- [ ] Контракт ML/inference зафиксирован до начала интеграции.
- [ ] Второй участник стартует с MockPredictor.
- [ ] Первый участник передает ONNX/TorchScript с manifest и checksum.
- [ ] RT-DETR/ByteTrack не добавлены без error analysis.
- [ ] SmolVLM не находится в основном execution path.
- [ ] Видео и веса исключены из обычного Git.
- [ ] Каждый PR содержит тесты, команды и handoff.
- [ ] Merge в `main` выполняется только после demo smoke test.
