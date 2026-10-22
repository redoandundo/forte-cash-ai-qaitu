# Передача ML участнику inference

## Статус и границы

Ветка: `feature/ml-model`. Пакет `baseline-repair-v2-development` предназначен для технической интеграции; обе проверенные архитектуры пока не проходят quality gate. Резервный v3 сохранён отдельно без изменения весов. Отчёт: [training_report.md](../../reports/training_report.md).

Изменения относятся к ML, его тестам, конфигурации и отчётам. Общие contracts, API и frontend принадлежат их владельцам. Для интеграции переносить новые ML-коммиты после `3b605c1`, без merge смешанной ветки `codex/model-output-integration`.

Веса и подробные данные находятся вне обычного Git. В старой истории веса уже есть; эта работа историю не переписывает.

## Зафиксированная поставка

- Код: `5c9f0958f6d1c4d56353665c4da221dd1c59a0be` (включает основной ремонт `744f5e8`).
- Архив: `artifacts/ml/bundles/baseline-repair-v2.zip`, около 7,4 МБ.
- Archive SHA-256: `b9a8b2c809f7a6dc72e0bf655707fb9d2e430c906e573aa2f10ed4750d0ae90b`.
- Canonical model SHA-256: `00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d`.
- Проверено 9 файлов архива по SHA256SUMS; 32 ML-теста прошли.
- Реестр: [release_record.json](../../reports/ml/release_record.json).

## Окружение

Python 3.12. Проверено на Windows, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, RTX 3050 Ti 4 ГБ. ONNX Runtime установлен только CPU, отдельная GPU-версия ORT не нужна.

```powershell
uv venv --python 3.12 .venv-ml
uv pip install --python .venv-ml/Scripts/python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-ml/Scripts/python.exe -r requirements-ml-repair.txt
```

Для CPU обучения установить те же версии torch/torchvision с официального индекса `https://download.pytorch.org/whl/cpu`. Не смешивать GPU и CPU пакеты onnxruntime в одном окружении.

Задайте переменные окружения для локальных входов: `QAITU_DATASET_MANIFEST` — существующий нормализованный manifest, `QAITU_VIDEO_ROOT` — каталог исходных видео. Они и индивидуальная разметка не коммитятся.

## Аудит, окна и эксперимент

```powershell
$py = ".venv-ml/Scripts/python.exe"
$run = "artifacts/ml/repair-20261017"
& $py scripts/ml/audit_training_sources.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --output "$run/data_audit.json"
& $py scripts/ml/prepare_clips.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --output "$run/windows.json"
& $py scripts/ml/run_repair_experiment.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --output "$run"
```

Команды не перезаписывают имеющиеся результаты; для новой итерации задайте новый `$run`. `--resume` у experiment повторно проверяет lineage завершённых folds. Если доступны только workbook и видео, `audit_annotations.py --data-root <каталог open> --output <новый локальный manifest>` формирует аудит; новый аудит может менять confidence и группировку, поэтому это новый эксперимент.

`prepare_clips.py` сохраняет cache RGB-окон с реальными PTS в NPZ, без повторного сжатия исходных видео. Dataset использует те же функции выбора окон. Подтверждённые исправления разметки передаются через `--review` во все команды данных. Review JSON индексируется source_id; требует `review_status: confirmed`, `reviewer`; интервалы содержат `start_sec`, `end_sec`, `review_status: confirmed`. `events_complete: true` разрешается только после проверки всех событий записи.

## Финальный checkpoint и калибровка

```powershell
& $py scripts/ml/train_video.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan "$run/frozen_plan.json" --variant baseline --output "$run/final_baseline"
& $py scripts/ml/evaluate_full_videos.py --mode calibrate --checkpoint "$run/final_baseline/best.pt" --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan "$run/frozen_plan.json" --fold 4 --output "$run/final_baseline/calibration.json"
```

Fold 4 зарезервирован для калибровки. Оценивать финальный checkpoint как независимый тест на остальных четырёх folds нельзя: он на них обучен. Для разработки используется nested OOF. `--mode evaluate --calibration <файл>` отклонит пересечение с train/calibration.

## Экспорт и smoke

Положительный и отрицательный файлы для parity задаются в `QAITU_POSITIVE_VIDEO` и `QAITU_NEGATIVE_VIDEO`.

```powershell
$codeCommit = git rev-parse HEAD
$bundle = "artifacts/ml/bundles/baseline-repair-v2"
& $py scripts/ml/export_model.py --checkpoint "$run/final_baseline/best.pt" --calibration "$run/final_baseline/calibration.json" --metrics "$run/baseline/summary.json" --output-dir "$bundle" --model-version baseline-repair-v2-development --code-commit "$codeCommit" --parity-video "$env:QAITU_POSITIVE_VIDEO" "$env:QAITU_NEGATIVE_VIDEO"
& $py scripts/ml/infer_onnx.py "$env:QAITU_POSITIVE_VIDEO" --bundle "$bundle"
& $py scripts/ml/benchmark_bundle.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --bundle "$bundle" --output "$run/benchmark.json"
& $py -m pytest tests/ml -q
```

В bundle входят `model.onnx`, `model_manifest.json`, `preprocessing.json`, `thresholds.json`, `labels.json`, `metrics.json`, `SHA256SUMS.txt`, `ml-to-inference.md` и совместимый динамический граф с manifest. Код commit записывается в bundle. Ошибочный export не публикует частичный каталог и не изменяет fallback.

## Два представления одного checkpoint

| Получатель | Граф | Manifest |
|---|---|---|
| Каноническая schema v1 | `model.onnx`, batch 1 | `model_manifest.json`, вложенные input/output |
| Существующий OnnxPredictor | `compatibility/model_dynamic.onnx`, dynamic batch | `compatibility/inference/model_manifest.json`, плоский формат |

Snapshot официальной schema хранится только в ML-конфигурации: `configs/ml/model_manifest.v1.schema.json`, источник — `feature/teamlead-core@fd23a1b`, `docs/contracts/model_manifest.schema.json`. Общая schema не изменялась. Статический граф соответствует требованию целочисленных dimensions, динамический сохраняет совместимость текущего адаптера.

Не перемещайте файлы совместимости отдельно: существующий OnnxPredictor вычисляет корень как `manifest.parents[2]`. Передавайте абсолютные пути к графу и manifest после раскрытия каталога bundle; сохраняйте внутренние относительные пути.

Для проверки конкретного кода inference:

```powershell
& $py scripts/ml/verify_inference_adapter.py --reference-package artifacts/reference_inference --reference-commit 271400fdbc51f516c9dc9c3d1c10742496f63b03 --bundle "$bundle" --video "$env:QAITU_POSITIVE_VIDEO" --output "$run/adapter_verification.json"
```

`reference_inference` — локальная копия `__init__.py`, `contracts.py`, `preprocessing.py`, `predictor.py` из указанного commit; инструмент не загружает код из сети и не меняет его.

## Обязательные настройки владельцу inference

- Читать все sidecars из одного bundle. `labels.json`: 0 = no_violation, 1 = possible_violation.
- Вход float32 RGB NCTHW; baseline [N,3,4,160,160]. Resize OpenCV INTER_LINEAR stretch; ImageNet mean/std.
- Окно 6 с, stride 2 с, endpoint=False, nearest PTS, время от stream start. `include_tail_window=True`.
- Выход графа — **logit**. Sigmoid ровно один раз.
- Сглаживание [0.25,0.5,0.25], края повторяются; менее трёх окон — без сглаживания. Затем сравнение с порогом, объединение активных центров с расстоянием ≤ stride, максимум три сильнейших события.
- Событие: `start_sec, end_sec, peak_time_sec, peak_score`. Score — показатель ранжирования.
- Старые правила агрегации и новый threshold не смешивать. Совпадение OnnxPredictor проверено на уровне тензоров; весь pipeline после включения этих правил ещё требует проверки.
- JPEG и технический статус failed — ответственность inference/API. ML CLI выбрасывает исключение при технической ошибке, а не возвращает no_violation.
- При смене preprocessing, sampling, smoothing или checkpoint нужна новая калибровка. Pipeline fingerprint переносится между LF/CRLF checkout.

## Критерии принятия

Ориентироваться на [сводный отчёт](../../reports/training_report.md), а не только на успешный smoke. Сейчас качество не проходит: macro F1 < 0,64, recall < 0,75, ложные события > 0,60/мин, MAE > 5 с, event recall < 70%. Проверка полного API, JPEG и новых длинных видео остаётся интеграционной задачей. Новые веса автоматически не становятся основной моделью.

## Совмещение веток

Совместимость графа с OnnxPredictor подтверждена. Полный Git merge с `feature/video-inference@271400f` отдельно проверен через `git merge-tree`: он имеет 9 конфликтующих путей, потому что эта ветка содержит также изменения ML.

Конфликты: `.gitignore`, `analysis/clip_manifest.json`, `analysis/dataset_manifest.json`, `reports/hardware.md`, `requirements-ml.txt`, `scripts/ml/evaluate_full_videos.py`, `scripts/ml/prepare_clips.py`, `scripts/ml/train_video.py`, `src/fortebank_audit/ml/models.py`. Рабочие ветки этой проверкой не изменялись.

Порядок для владельца integration:

1. Сохранить общие contracts/API из ветки тимлида.
2. Взять актуальные ML-код, ML-зависимости, тесты и handoff из `feature/ml-model`.
3. Взять модуль `src/fortebank_audit/inference/` и относящиеся к inference конфигурации/тесты из ветки inference. Находящиеся там дополнительные ML-challenger модели и их тесты не входят в проверенный здесь рецепт из двух вариантов.
4. Объединить правила `.gitignore`; manifests исходных данных оставить вне Git. Заполненный ML handoff заменяет шаблон из ветки тимлида.
5. Применить настройки tail/smoothing из этого handoff и выполнить полные integration tests.

В ML-ветку API и inference не переносились. Автоматический merge и end-to-end совместимость всего сервиса этой поставкой не подтверждаются.
