# Текущий отчёт по улучшению ML-системы

Дата: 17 октября 2026 года

## Короткий вывод

Версия v3 стала лучше как инженерный baseline: она быстрее, даёт меньше ложных тревог на исходном holdout и точнее локализует часть событий. После прошлого запроса веса основной ONNX-модели не менялись. Улучшился контур доказательств: теперь качество проверено nested OOF на всех 40 уникальных источниках, порог выбирается на отдельном calibration fold, а ошибки автоматически попадают в learning lab.

Честная текущая оценка способности переноситься на новый сценарий — `macro F1 0.540`, 95% bootstrap interval `0.386–0.693`. Поэтому прежние `0.750` на восьми видео сохраняются как локальный holdout результат, но больше не считаются production оценкой.

## Что реально улучшилось от v1 к v3 на прежнем holdout

| Метрика | v1 | v3 | Изменение |
|---|---:|---:|---:|
| Precision | 0.667 | 0.750 | +0.083 |
| Recall | 1.000 | 0.750 | −0.250 |
| Macro F1 | 0.733 | 0.750 | +0.017 / +2.3% |
| AUROC | 0.500 | 0.688 | +0.188 |
| False positives/minute | 1.014 | 0.458 | −54.8% |
| Timestamp MAE | 8.75 s | 7.00 s | −20.0% |
| Попадание в ±5 s | 25% | 50% | +25 п.п. |
| p95 runtime | 6.52 s | 3.76 s | −42.3% |

Это сравнение показывает impact исправлений pipeline, но обе версии не имеют одинаковой nested OOF оценки. Нельзя утверждать, что generalization macro F1 вырос на 2.3%: для v1 такой честной оценки нет.

## Новая проверка на всех данных

Сделано пять внешних scenario-safe folds. В каждом запуске:

1. внешний fold полностью исключён из обучения;
2. другой fold выбирает threshold;
3. checkpoint сохраняется на заранее фиксированном первом epoch;
4. внешний fold оценивается только один раз;
5. все 40 источников встречаются во внешней проверке ровно один раз.

| Nested OOF метрика | Результат |
|---|---:|
| Источники | 40 |
| Precision | 0.560 |
| Recall | 0.667 |
| Macro F1 | 0.540 |
| Macro F1 95% interval | 0.386–0.693 |
| Confusion matrix | `[[8, 11], [7, 14]]` |
| False positives/minute | 0.874 |
| Timestamp MAE | 8.48 s |
| Попадание в ±5 s | 52.4% |

Nested OOF обучает каждый fold на 24 источниках, потому что ещё 8 используются для внутренней калибровки. Финальная v3 обучена на 32, поэтому оценка может быть немного консервативной. Разрыв между folds остаётся большим: macro F1 от `0.365` до `0.619`.

## Impact

- Система стала пригодна как локальный инструмент предварительного отбора и показа evidence frame.
- На исходном holdout число ложных тревог в минуту сократилось примерно вдвое, а p95 latency — на 42%.
- Теперь у команды есть защита от ложного прогресса: модель не получит promotion без пяти outer folds, независимого threshold, доверительного интервала и ONNX parity.
- Learning lab хранит 3 эксперимента, 45 метрик, 31 error case, 7 гипотез и 5 learning cycles.
- Текущий ONNX остаётся интеграционным baseline. Он не прошёл production gate.

## Что обнаружено и исправлено сейчас

| Уязвимость | Исправление |
|---|---|
| Threshold и отчёт выбирались на одних восьми видео | Добавлена nested OOF оценка с отдельным calibration fold |
| Лучший epoch мог выбираться по внешнему fold | Добавлен `checkpoint-policy=last-epoch` |
| Пять independently trained моделей имели несовместимые шкалы score | Финальные решения считаются порогом своего inner calibration fold |
| Folds были несбалансированы до 2/6 | Построен детерминированный group-safe план: четыре folds 4/4, один 3/5 |
| Reward-v1 продвигал результат без confidence interval | Reward-v2 требует 5 OOF folds, threshold без leakage и CI lower bound ≥ 0.50 |
| Config всё ещё объявлял F2 | Decision metric синхронизирована с macro F1 / AUROC |
| Full-video evaluator по умолчанию использовал окно 8 s вместо 6 s | Window теперь берётся из checkpoint, fallback 6 s |
| Ошибки оставались ручным списком из трёх роликов | Все classification и localization ошибки импортируются в SQLite learning lab |

## Главные error clusters

- `11` false positives; все относятся к high-confidence negative labels.
- `9 из 11` false positives находятся в matched contrastive pairs. Модель часто узнаёт сцену, но не различает действие внутри пары.
- `7` false negatives; `6 из 7` имеют medium/low annotation confidence. Здесь сначала нужна проверка разметки и интервала.
- `10 из 21` положительных источников имеют localization error больше 5 секунд.
- Fold 4 дал macro F1 `0.365`; это признак domain/scenario shift и малого числа обучающих сценариев.

## Пределы улучшения

Жёсткий математический потолок сейчас неизвестен. Практический потолок текущего clip classifier близок: 40 источников, бинарные weak labels, только один timestamp, узкий диапазон score и действия с маленькими объектами. Замена MobileNet на более тяжёлый backbone уже не помогла R3D/X3D и сама по себе не решит эти ограничения.

Основной следующий прирост должен прийти из качества supervision:

1. pairwise ranking положительного и отрицательного видео одного сценария;
2. hard-negative clips вокруг 11 false-positive peaks;
3. ручная проверка шести uncertain false negatives;
4. temporal MIL со sparsity и smoothness для полного видео;
5. позже — отдельные zone, door и hand/table признаки там, где это подтверждает error analysis.

## Текущий этап

- `C0 Data and evaluation trust` — завершён.
- `C1 Out-of-fold truth layer` — завершён: 5 nested folds, 40 outer predictions, bootstrap interval и error curriculum записаны.
- `C2 Temporal anomaly learning` — текущий этап.
- Первый эксперимент C2: `H7 pair ranking + hard-negative curriculum`.
- Promotion target: macro F1 не ниже `0.640`, paired false positives меньше `5`, recall не ниже `0.67`; затем повтор той же nested OOF процедуры.

## Принятое решение

Не менять `models/best_model.onnx` результатами текущего аудита. v3 остаётся стабильным интеграционным baseline. Следующую модель обучать с pair-ranking и temporal MIL, а замену ONNX разрешить только после прохождения reward-v2 gates.
