# Qaitu ML Learning Lab

Это полностью локальное пространство непрерывного улучшения модели. Оно хранит доказательства, гипотезы, эксперименты, метрики, ошибки, reward-события и этапы развития в SQLite.

## Состав

- `learning_lab.sqlite3` — основная база.
- `schema.sql` — воспроизводимая схема.
- `seed/` — проверенные источники, гипотезы, циклы и начальные эксперименты.
- `reward_policy.json` — формула награды и обязательные защитные условия.
- `META_RESEARCH.md` — исследование аналогов и рекомендуемой архитектуры.
- `NEXT_STAGE.md` — автоматически выбранный следующий этап.
- `dashboard.html` — локальная обзорная панель.
- `scripts/research/lab.py` — CLI управления лабораторией.

## Основная идея

Система награждает только улучшение на независимых источниках. Train accuracy, размер модели и красивый demo сами по себе награды не дают.

Награда учитывает:

- macro F1;
- AUROC;
- recall;
- попадание таймкода в ±5 секунд;
- timestamp MAE;
- ложные срабатывания в минуту;
- p95 latency;
- объём независимой validation-выборки.

Любая утечка сценариев или ошибка ONNX parity блокирует продвижение модели независимо от метрик.
Reward-v2 также требует пять outer folds, независимую калибровку threshold и нижнюю границу bootstrap macro F1 не ниже `0.50`.

## Команды

```powershell
python scripts/research/lab.py init
python scripts/research/import_nested_oof.py
python scripts/research/lab.py reward --baseline mobilenet-v1 --candidate mobilenet-balanced-v3
python scripts/research/lab.py recommend
python scripts/research/lab.py dashboard
python scripts/research/lab.py summary
```

## Текущий результат

- Локальный holdout v3: macro F1 `0.750` на 8 источниках.
- Честная nested OOF оценка: macro F1 `0.540` на 40 источниках, 95% interval `0.386–0.693`.
- Reward-v2 не продвигает v3 в production: нижняя граница CI и recall не проходят gates.
- C1 truth layer завершён; следующий этап — pair ranking, hard negatives и temporal MIL.

После каждого нового эксперимента нужно записать config, dataset version, split version, полные метрики и error cases, затем сравнить кандидата с текущей выбранной моделью.
