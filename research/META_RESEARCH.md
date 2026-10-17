# Мета-исследование развития ForteBank Video Compliance AI

Дата: 17 октября 2026 года

## Вывод

Для текущих 42 роликов невыгодно бесконечно менять backbone. Основной прирост должен прийти из пяти связанных контуров:

1. достоверная оценка без leakage;
2. weakly supervised обучение на полном видео;
3. обучение представлений на локальном неразмеченном видео;
4. накопление проверенных ошибок и active learning;
5. отдельные интерпретируемые детекторы для зон, людей, двери и состояния стола.

Текущая MobileNet остаётся быстрым production baseline. Следующая исследовательская модель должна быть temporal MIL, а не ещё один обычный clip classifier.

## Ближайшие аналоги

### Weakly supervised video anomaly detection

[UCF-Crime / Real-world Anomaly Detection in Surveillance Videos](https://arxiv.org/abs/1801.04264) формулирует видео как bag временных сегментов и учит anomaly ranking по video-level метке. Это подходит нашему случаю: нарушение занимает малую часть ролика, а большая часть положительного видео выглядит нормально.

[RTFM](https://openaccess.thecvf.com/content/ICCV2021/html/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.html) усиливает MIL через различимость magnitude признаков, temporal context, sparsity и smoothness. Авторы отдельно рассматривают тонкие аномалии, которые мало отличаются от нормальных действий. Это наиболее прямой источник для следующего прототипа.

[Real-Time Weakly Supervised Video Anomaly Detection](https://openaccess.thecvf.com/content/WACV2024/html/Karim_Real-Time_Weakly_Supervised_Video_Anomaly_Detection_WACV_2024_paper.html) показывает end-to-end подход с оценкой задержки. Практический урок: качество нужно измерять вместе с decision period и latency, а не после обучения.

[Uncertainty-aware pseudo labels](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.html) подтверждают пользу явного моделирования неопределённости weak labels. Наша схема `high/medium/low` является упрощённой первой версией этой идеи.

### Self-supervised обучение

[VideoMAE V2](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html) показывает, что masked video pretraining строит переносимые представления без ручных классов. Для проекта это означает возможность обучиться ракурсу кассы, двери, рукам и движению на локальных видео. Полную модель нельзя сразу запускать на 4 ГБ VRAM: сначала нужен маленький encoder или заранее извлечённые frozen features.

### Адаптация к новым нарушениям

[Meta-learning для unseen abnormal activities](https://openaccess.thecvf.com/content/ACCV2020/html/Park_Learning_to_Adapt_to_Unseen_Abnormal_Activities_under_Weak_Supervision_ACCV_2020_paper.html) рассматривает адаптацию к новым аномалиям при video-level binary labels. Этот подход станет полезным после подтверждения нескольких реальных семейств нарушений; сейчас семантика классов ещё недостаточно надёжна.

### Существующие checkout-системы

[NVIDIA Smart Self-Checkout](https://docs.nvidia.com/mms/text/Smart_self_checkout_Overview.html) использует embeddings и базу известных объектов, а оператор может добавить новый пример с ground truth. Для нас аналогом должна стать локальная память проверенных сцен: embeddings, объективное наблюдение, решение проверяющего и связь с ошибочным кластером.

[NVIDIA Retail Loss Prevention](https://www.nvidia.com/en-us/ai-data-science/ai-workflows/retail-loss-prevention/) сочетает few-shot, self-supervised и active learning. Это подтверждает системный принцип: checkout AI развивается через feedback loop, а не только через разовое обучение классификатора.

[DeepStream reference application](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html) разделяет decode, primary detector, secondary classifiers, tracking и output metadata. Для ForteBank полезно то же разделение: video classifier не должен одновременно угадывать человека, деньги, дверь, юридическое правило и итоговый текст.

### Учёт экспериментов

[MLflow Tracking](https://www.mlflow.org/docs/latest/tracking/) хранит runs, параметры, метрики, модели и artifacts. Текущая `learning_lab.sqlite3` реализует минимальную локальную версию этих сущностей без новой тяжёлой зависимости. При росте команды схему можно мигрировать в локальный MLflow server с SQLite backend.

## Мета-архитектура

```text
официальные метки + reviewer feedback + источники исследований
                         ↓
               local evidence database
                         ↓
        hypotheses → experiments → error clusters
                         ↓
        scenario-safe OOF evaluator + reward policy
                         ↓
         promote / reject / request more evidence
                         ↓
         next learning cycle and active sampling
```

### Слой 1 — Truth

- SHA-256 источника;
- scenario group;
- объективное наблюдение;
- интерпретация правила отдельно;
- reviewer status;
- confidence разметки;
- история изменения метки.

### Слой 2 — Representation

- текущий MobileNet baseline;
- temporal MIL features;
- позже локальное self-supervised pretraining;
- embeddings проверенных сцен для поиска похожих ошибок.

### Слой 3 — Temporal reasoning

- bag из всех snippet полного видео;
- ranking loss между normal и anomaly bags;
- smoothness и sparsity;
- event proposal как устойчивый временной интервал;
- оценка IoU/MAE таймкода.

### Слой 4 — Structured evidence

- person tracks;
- zone crossing;
- door state;
- hand/table ROI;
- состояние ценностей только при достаточной визуальной уверенности.

### Слой 5 — Feedback and reward

- low-margin clips отправляются на проверку;
- повторяющиеся false positives формируют hard-negative curriculum;
- новая модель сравнивается с текущей на одинаковых OOF predictions;
- reward уменьшается при маленькой выборке;
- leakage, parity failure или ухудшение FP/min блокируют promotion.

## Формула награды

Сырой reward складывается из изменений macro F1, AUROC, recall, ±5-second hit rate, timestamp MAE, FP/min и p95 latency. Затем он умножается на:

```text
confidence = min(1, sqrt(validation_sources / 40)) * min(1, oof_folds / 5)
```

Поэтому holdout без outer-fold predictions не получает evidence reward. Модель нельзя продвинуть при scenario leakage, threshold-selection leakage, ONNX parity выше `1e-4`, ухудшении macro F1 или FP/min, recall ниже `0.75` либо нижней границе bootstrap macro F1 ниже `0.50`.

## Следующий цикл

### Этап A — OOF truth layer

1. Построить пять folds по `scenario_group`.
2. Проверить наличие обоих классов в каждом fold.
3. Обучить MobileNet на четырёх folds и получить predictions пятого.
4. Объединить только out-of-fold predictions.
5. Выбрать единый threshold.
6. Рассчитать bootstrap confidence intervals.
7. Записать все ошибки в `error_cases`.

### Этап B — Temporal MIL

1. Один раз извлечь признаки всех окон.
2. Представить полное видео как bag snippets.
3. Реализовать ranking, smoothness и sparsity losses.
4. Сравнить с v3 на тех же OOF folds.
5. Продвинуть модель только при положительном confidence-adjusted reward.

### Этап C — Active learning

Приоритет проверки:

1. clips около threshold;
2. повторяющиеся false positives;
3. disagreement MobileNet и MIL;
4. новые embedding-кластеры;
5. low-confidence исходные метки.

## Чего пока не делать

- Не обучать многоклассовые нарушения без подтверждённых классов.
- Не добавлять тяжёлый VLM в основной inference path.
- Не вознаграждать train accuracy.
- Не выбирать threshold отдельно на каждом test fold.
- Не считать все окна положительного видео нарушением.
- Не добавлять object detector до появления проверяемой box/zone-разметки.
- Не отправлять банковские видео во внешние сервисы.
