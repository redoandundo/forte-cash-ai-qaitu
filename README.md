# Qaitu — ForteBank Cash AI

Командный проект Qaitu для мероприятия Datathon26. Мы решаем кейс ForteBank: создаём полностью локальную систему анализа видео кассовой зоны, которая обнаруживает признаки возможного нарушения, указывает временной интервал и сохраняет проверяемые JPEG-доказательства.

Система является инструментом поддержки проверки. Она не выносит юридическое решение и не должна утверждать то, что невозможно надёжно установить по видео.

Официальные требования проекта в безопасном для репозитория виде изложены в [`docs/requirements/DATATHON26_CASE_REQUIREMENTS.md`](docs/requirements/DATATHON26_CASE_REQUIREMENTS.md). Исходный PDF и датасет в публичный GitHub не загружаются.

## Команда

| Роль | Зона ответственности |
|---|---|
| Тимлид | архитектура, API, jobs, GitHub, интеграция и финальная demo-сборка |
| ML-инженер | данные, R3D-18/X3D, обучение, оценка, экспорт модели и model handoff |
| Inference-инженер | декодирование видео, окна, ONNX Runtime, таймкоды, aggregation, evidence и benchmark |

Полный контекст команды: [`knowledge/TEAM_CONTEXT.md`](knowledge/TEAM_CONTEXT.md).

## С чего начать Codex

1. Прочитать [`AGENTS.md`](AGENTS.md).
2. Прочитать [`GOALS.md`](GOALS.md) и актуальную фазу в [`PLANS.md`](PLANS.md).
3. Прочитать [`knowledge/PROJECT_CONTEXT.md`](knowledge/PROJECT_CONTEXT.md) и [`knowledge/current_state.yaml`](knowledge/current_state.yaml).
4. Прочитать относящиеся к задаче contracts и handoffs.
5. Проиндексировать текущую ветку через `codebase-memory-mcp`, если он доступен.

## Ветки

```text
main
└── develop
    ├── feature/teamlead-core
    ├── feature/ml-model
    ├── feature/video-inference
    └── integration/ml-inference
```

Полная схема работы: [`docs/team/GITHUB_TEAM_ARCHITECTURE.md`](docs/team/GITHUB_TEAM_ARCHITECTURE.md).

## Данные и безопасность

- Видео, кадры, модельные веса, evidence, секреты и локальные кэши не коммитятся.
- Dataset labels, подробные разборы отдельных видео и исходный PDF кейса остаются локальными.
- Обработка содержимого видео выполняется локально.
- В Git хранятся код, contracts, manifests, метрики, конфигурация и воспроизводимые инструкции.
- Перед использованием модели проверяются manifest, preprocessing и SHA-256.

## Текущий статус

Проект находится на этапе формирования общей архитектуры и параллельного старта ML и inference-компонентов. Актуальное состояние хранится в [`knowledge/current_state.yaml`](knowledge/current_state.yaml).
