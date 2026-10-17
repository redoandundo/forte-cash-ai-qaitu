# Повторно используемые prompts

## Ориентация в репозитории

```text
Прочитай AGENTS.md, GOALS.md, PLANS.md, knowledge/TEAM_CONTEXT.md, knowledge/PROJECT_CONTEXT.md и knowledge/current_state.yaml. Затем прочитай contracts и handoffs, связанные с моей ролью. Выполни read-only аудит текущей ветки и git status. Если доступен codebase-memory-mcp, проверь актуальность индекса и используй graph tools для знакомства с кодом. Сообщи: текущую фазу, мою зону файлов, upstream/downstream зависимости, риски и ближайший проверяемый результат. Не изменяй файлы до этого отчёта.
```

## ML-инженер

Использовать полный prompt из `docs/team/ML_TEAMMATE_GUIDE.md`.

## Inference-инженер

Использовать полный prompt из раздела 15 `docs/team/GITHUB_TEAM_ARCHITECTURE.md` и техническое руководство `docs/team/INFERENCE_TEAMMATE_GUIDE.md`.

## Проверка handoff

```text
Проверь последний handoff моей upstream-команды против соответствующего contract. Ничего не изменяй. Убедись, что указаны commit SHA, версия, команды, фактические тесты, artifacts, checksum, ограничения и следующий шаг. Отдельно перечисли отсутствующие или противоречивые данные. Не считай artifact готовым только по заявлению автора.
```

## Подготовка integration PR

```text
Прочитай AGENTS.md, актуальную фазу PLANS.md, contracts и handoffs обеих сторон. Проверь diff feature-ветки относительно integration/ml-inference, ownership файлов, отсутствие запрещённых artifacts и breaking changes. Запусти доступные focused tests и git diff --check. Подготовь описание Draft PR с наблюдаемыми результатами, рисками и точными следующими действиями. Не выполняй merge.
```
