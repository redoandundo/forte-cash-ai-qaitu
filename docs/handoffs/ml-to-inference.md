# ML → inference: проверенная development-поставка

## Дополнение 2026-10-18: R3D и проверка CPU-переноса

Код `0b0ba91`, ветка `codex/ml-scorecard-improvements`, поверх `57a5e4e`.
Отчёт и команды: [r3d_scorecard.md](../../reports/ml/r3d_scorecard.md).
Агрегированные метрики и SHA: [r3d_scorecard_metrics.json](../../reports/ml/r3d_scorecard_metrics.json).

Исправлены BatchNorm и Kinetics-нормировка R3D-18. Пять OOF folds дали recall
.952, но macro F1 .387 и 18 FP: кандидат отклонён. Активная поставка остаётся v4.
Дополнительный CPU stress с неизменными CUDA thresholds выявил одно изменение
вердикта около порога; при переносе backend проверять численную чувствительность.

Локальный `artifacts/ml/bundles/r3d-scorecard-fold0` и одноимённый ZIP — только
comparison artifact: `[1,3,16,112,112]`, ImageNet input, Kinetics conversion внутри
графа, окно 6 s / stride 2 s. Threshold=0 делает его непригодным для полезных
вердиктов; это не автоматическая замена baseline. Schema/checksum и parity
проверены: max logit error `5.066394805908203e-7` на 12 реальных окнах.
Adapter `2f53ce1` проверен на positive/negative видео, событиях и JPEG;
повреждённое видео отклонено. Полный набор после объединения: 74 passed.
Новые веса, evidence и подробные labels остаются вне Git.

## Дополнение 2026-10-18: scorecard

Код оценки и экспериментов: commit `20209ec` в `feature/ml-model`.

Проверка критериев обнаружения/классификации/устойчивости:
[scorecard_followup.md](../../reports/ml/scorecard_followup.md).
Проведены два запуска fine-tuning и 200 full-video source-condition оценок
на пяти OOF folds. Выигрыша кандидата нет, v4 и его SHA сохранены.
Обнаружена чувствительность к сжатию и повороту; другой реальный ракурс не проверен.
Калибровка теперь явно предупреждает о малом числе примеров и all-positive решении.
Эти предупреждения не меняют threshold или API. Типы событий требуют подтверждённой
разметки; критерий классификации конкретных типов пока не выполнен.

Дата: 2026-10-17. Ветка: `feature/ml-model`.
Код: `aba56d31030aa9830fe2770e4aa9d54d2879ef24`.
Полный структурированный отчёт: [diagnosis_report.md](../../reports/ml/diagnosis_report.md).
Метрики: [diagnostic_metrics.json](../../reports/ml/diagnostic_metrics.json).
Архив, SHA и фактические финальные проверки: [audited_release_record.json](../../reports/ml/audited_release_record.json).

## Решение

`baseline-audited-v4-development` — воспроизводимый **development-only** bundle.
Quality gate не пройден. Новые веса не приняты: исправленная нормировка loss
повысила recall, но ухудшила macro F1 и FP/min. Веса baseline-repair-v2 сохранены;
это диагностированная поставка с проверенными метаданными, не улучшенная модель.
Прежний fallback не заменять автоматически.

v4 включает guard отсутствующего decoder перед PyAV decode: обнаруженный
native crash заменён контролируемой ошибкой. После изменения fingerprint
калибровка и OOF пересчитаны; scores поддерживаемых видео не изменились.
Предыдущий локальный v3 архив сохранён, но текущая поставка — v4.

- Baseline nested OOF: precision 0.500, recall 0.619, positive F1 0.553,
  macro F1 0.458, FP events/min 1.033, event recall ±5 s 0.364, peak MAE 7.133 s.
- Confusion `[TN,FP; FN,TP]`: `[6,13; 8,13]`.
- Кандидат: macro F1 0.413, recall 0.667, FP/min 1.191 — отклонён.
- Five-fold split проверен по source/SHA/scenario. Calibration отделена от
  outer fold. Это повторно используемый development dataset, не новый test.
- Точки неоднозначного дубликата не подтверждены как два события. Без него
  baseline event recall 0.400, MAE 5.061 s. Полных подтверждённых интервалов нет.

## Mock и реальная модель

Настроенный MockPredictor игнорирует содержание и повторяет
`[0.1,0.2,0.8,0.9,0.3]`. С window=8 и stride=2 первый максимум — 10 s.
На том же ролике ONNX дал 19 s; разные окна имеют разные тензоры.
После добавления/удаления 4 s пик ONNX смещается на 23/15 s точно.
Нельзя исправлять mock настройкой порога реальной модели.

## Bundle и I/O

Локальный каталог: `artifacts/ml/bundles/baseline-audited-v4`.
Архив: `artifacts/ml/bundles/baseline-audited-v4.zip`.
Archive SHA-256: `c387bdea9bd80a124c5587fccfcec3243d2550658ee09ca6834acaa9e54a1241`.
ONNX SHA-256: `00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d`.
Вес и архив передаются приватно, вне Git. SHA архива — в release record.

Содержимое: model.onnx, model_manifest.json, preprocessing.json, thresholds.json,
labels.json, metrics.json, SHA256SUMS.txt, ml-to-inference.md,
compatibility/model_dynamic.onnx, compatibility/inference/model_manifest.json.

- Canonical schema-v1: вложенные input/output, fixed batch `[1,3,4,160,160]`.
- RGB float32, NCTHW, OpenCV linear stretch, ImageNet normalization.
- Window 6 s, stride 2 s, endpoint-false uniform sampling по nearest PTS,
  начало timeline = stream start; обязательный tail window.
- Output `logits`, sigmoid **ровно один раз**.
- Threshold `0.45335930958390236`, отдельный calibration fold 4 (8 источников).
  Его precision/recall/F1 не являются независимым test.
- Smoothing `[0.25,0.5,0.25]`, replicate edges; <3 окон — identity.
- Active centers объединяются в пределах stride, максимум 3 сильнейших события,
  итоговый порядок хронологический. Score — ranking score, не вероятность закона.
- Compatibility graph имеет dynamic batch; старый flat-manifest adapter может
  читать его из compatibility/inference. **Актуальный adapter 2f53ce1 должен
  использовать canonical bundle напрямую.**

Pipeline fingerprint и checkpoint SHA записаны раздельно; новый `scoring_sha256`
связывает их с threshold и postprocessing. Все файлы проверяются по inventory.
Parity: 16 реальных окон, fixed N=1 / dynamic N=16, максимальная ошибка logit
`3.5762786865234375e-7`, tolerance `1e-4`, threshold decisions совпали.

## Запуск из ML checkout

Использовать Python 3.12 и существующий `requirements-ml-repair.txt` плюс
согласованный PyTorch 2.5.1 / torchvision 0.20.1. Проверено CUDA 12.1,
RTX 3050 Ti 4 ГБ; ONNX Runtime 1.23.2 CPU. Для полного adapter smoke дополнительно:

```powershell
uv pip install --python .venv-ml/Scripts/python.exe -r configs/ml/integration-smoke-requirements.txt
$env:PYTHONPATH = "src"
$py = ".venv-ml/Scripts/python.exe"
$bundle = "artifacts/ml/bundles/baseline-audited-v4"
& $py scripts/ml/infer_onnx.py "$env:QAITU_POSITIVE_VIDEO" --bundle "$bundle"
& $py -m pytest tests/ml -q
```

`QAITU_POSITIVE_VIDEO` и `QAITU_NEGATIVE_VIDEO` задаёт пользователь полным
локальным путём. Нужны видео кассовой сцены: произвольное видео можно технически
прочитать, но модель не обучена понимать любой сюжет.

Для полного внутреннего timeline и shift test (новый каталог на каждый запуск):

```powershell
& $py -m fortebank_audit.ml.diagnostics --video "$env:QAITU_POSITIVE_VIDEO" --bundle "$bundle" --output artifacts/ml/repro-diagnostics --shift-seconds 4
```

Выбирать положительное видео с действием после 4 s; trim удаляет первые 4 s.
Команда не скрывает неуспешный shift и завершается ненулевым кодом при провале.
Для сравнения с настоящим MockPredictor добавить `--reference-package` с локальной
копией inference package из указанного ниже commit, которую создаёт adapter smoke.

```powershell
& $py -m fortebank_audit.ml.adapter_smoke --reference-commit 2f53ce18a91bcddae5135f2cac0a03aa36c7e7d4 --video "$env:QAITU_POSITIVE_VIDEO" "$env:QAITU_NEGATIVE_VIDEO" --bundle "$bundle" --output artifacts/ml/repro-adapter --evidence-root "$env:QAITU_EVIDENCE_ROOT"
```

Этот SHA должен присутствовать локально после `git fetch origin`. Инструмент
копирует выбранные файлы из Git в ignored output, не меняет inference checkout
и не скачивает модели. До исправления Unicode writer `QAITU_EVIDENCE_ROOT`
должен указывать на локальный каталог с латинским абсолютным путём.

## Воспроизведение обучения и оценки

`QAITU_DATASET_MANIFEST` — локальный нормализованный manifest,
`QAITU_VIDEO_ROOT` — каталог исходных видео. Они не публикуются.
Чтобы заново получить baseline checkpoints и frozen folds:

```powershell
$run = "artifacts/ml/repro-baseline"
& $py scripts/ml/audit_training_sources.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --output "$run/data_audit.json"
& $py scripts/ml/run_repair_experiment.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --variants baseline --output "$run"
& $py scripts/ml/train_video.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan "$run/frozen_plan.json" --variant baseline --output "$run/final_baseline"
& $py scripts/ml/evaluate_full_videos.py --mode calibrate --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan "$run/frozen_plan.json" --fold 4 --checkpoint "$run/final_baseline/best.pt" --output "$run/final_calibration.json"
```

Диагностическая OOF-оценка с группами, positive F1 и анализом ошибок:

```powershell
& $py -m fortebank_audit.ml.experiments --stage rescore --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan "$run/frozen_plan.json" --checkpoints "$run/baseline" --output artifacts/ml/repro-baseline-details
& $py -m fortebank_audit.ml.experiments --stage development --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan "$run/frozen_plan.json" --config configs/ml/weighted_objective.yaml --output artifacts/ml/repro-weighted-dev
& $py -m fortebank_audit.ml.experiments --stage oof --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan "$run/frozen_plan.json" --config configs/ml/weighted_objective.yaml --output artifacts/ml/repro-weighted-oof
```

Остальные development ablations используют отдельные configs
`weighted_objective_30.yaml`, `weighted_hard_negatives.yaml`,
`weighted_objective_8frames.yaml` и новые output directories.

Экспортировать в **новый** каталог, не перезаписывая поставку:

```powershell
$exportCommit = git rev-parse HEAD
& $py scripts/ml/export_model.py --checkpoint "$run/final_baseline/best.pt" --calibration "$run/final_calibration.json" --metrics "$run/baseline/summary.json" --output-dir artifacts/ml/bundles/baseline-reproduced --model-version baseline-reproduced-development --code-commit "$exportCommit" --parity-video "$env:QAITU_POSITIVE_VIDEO" "$env:QAITU_NEGATIVE_VIDEO"
& $py scripts/ml/benchmark_bundle.py --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --bundle "$bundle" --output artifacts/ml/repro-cpu-benchmark.json
```

## Inference-инженеру

Подключать bundle adapter из `feature/video-inference@2f53ce1`. Реальный smoke
с весами выполнен этим ML checkout через `ml.adapter_smoke`; итог в release record.
В его checkout команда после приватного переноса и проверки SHA архива:

```powershell
$env:PYTHONPATH = "src"
python -m fortebank_audit.inference.benchmark --video "$env:QAITU_POSITIVE_VIDEO" --predictor onnx --bundle artifacts/ml/bundles/baseline-audited-v4 --allow-development-model --evidence-root "$env:QAITU_EVIDENCE_ROOT"
```

Исправить отдельно `cv2.imwrite` для Unicode evidence root. Рекомендуемый путь:
`cv2.imencode` + `Path.write_bytes` и регрессионный тест. До этого ASCII output
позволяет выполнить demo; Unicode input уже проверен. Не превращать JPEG failure
в `no_violation`. Визуальную различимость JPEG и UI-округление ML-ветка не меняет.

Отдельно проверен unknown-codec fixture: текущий `VideoReader.metadata()`
завершается `AttributeError` из-за отсутствующего decoder, без verdict.
Владельцу нужно проверять `stream.codec_context is None` до чтения width/decode
и выдавать `UnsupportedVideoError`. ML sampler уже содержит такой guard.

Внутренние события: start_sec, end_sec, peak_time_sec, peak_score,
`supporting_windows` (число), `supporting_window_indices`. Полный timeline:
CLI diagnostics либо `ml.runtime.predict_video(..., include_window_scores=True)`.
Не публиковать подробные window traces и dataset-derived отчёты в Git.

## Тимлиду: подключение API

В доступном `feature/teamlead-core@fd23a1b` API всё ещё использует joblib-путь;
переменная `FORTEBANK_MODEL_PATH` сама по себе не переключает его на новый ONNX.
Нужна интеграция composition root с `load_bundle_contract`, `OnnxPredictor` и
`analyze_video` из 2f53ce1. При загрузке development bundle передать явный
`allow_development_model=True`; не делать его незаметной production-подменой.
Логировать predictor class, model_version и проверенный SHA при старте API.

`possible_violation` переводится в публичное `violation` только после успешного
анализа. Любой decode/model/evidence failure возвращает `failed`. Проверить API,
повторную загрузку, frontend и время целого запроса. Полный ML CPU-прогон 40 видео
успешен, max 10.300 s, p95 4.482 s; это не общий API SLA. Предыдущий прогон тех же весов достигал 24.941 s при другой нагрузке компьютера; ускорение модели не заявляется. Live backend и API этой
задачей не переключались. Тип действия/нарушения и юридический вывод не выдумывать.
