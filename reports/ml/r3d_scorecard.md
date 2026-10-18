# R3D-18: проверка критериев и совместимости

Дата: 2026-10-18. Ветка: `codex/ml-scorecard-improvements`, основана на
`feature/ml-model@57a5e4e`. Код обучения/экспорта: `0b0ba91`.
Полные агрегированные результаты: [r3d_scorecard_metrics.json](r3d_scorecard_metrics.json).

## Решение

Сохранить `baseline-audited-v4-development`. Исправленный R3D-18 повышает recall,
но увеличивает ложные тревоги и ухудшает macro F1. Quality gate не пройден;
новый bundle предназначен только для воспроизведения и проверки интеграции.
Улучшение качества поставляемых весов и максимальный балл жюри не заявляются.

## Что исправлено

- У замороженного R3D-18 больше не меняется статистика BatchNorm при `train()`.
  Добавлено контролируемое размораживание последних stages, совместимое с
  существующим training loop. В измеренном baseline stages заморожены.
- Новый `r3d_18_kinetics` преобразует ImageNet-normalized вход в нормировку
  pretrained Kinetics **внутри графа**. Имена входа/выхода и внешний preprocessing
  сохранены; старое имя `r3d_18` сохраняет прежнюю семантику inference.
- Нормировка берётся из установленного torchvision; её назначение описано в
  [официальной документации R3D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html).
  Spatial sampling остаётся full-frame stretch, а не стандартным center crop.
- Дополнительный `stress_audit` проверяет provenance checkpoint и calibration,
  использует уже зафиксированные thresholds и строит доверительные интервалы
  с пересэмплированием **целых scenario groups**. В отличие от существующего
  `robustness`, этот CLI не пересчитывает calibration при смене backend.
- Scoring отклоняет некорректный batch size; transform получает копию кадров.
  Backend калибровки и её диагностические предупреждения сохраняются в bundle.
  Новые командные изменения `20209ec`/`57a5e4e` сохранены при объединении.

## Обнаружение и временной сигнал

40 уникальных источников, 27 scenario groups, пять прежних frozen folds.
В каждом прогоне три folds используются для обучения, один для calibration,
один для outer evaluation. SHA/source/scenario не пересекаются.
В R3D используются 16 кадров, 112x112, окно 6 s, stride 2 s, шесть заранее
заданных эпох, frozen pretrained backbone и source/class-weighted BCE.
Threshold выбирается только по calibration; правило smoothing не менялось.
Это повторно используемый development dataset, не новый внешний test.

| CUDA OOF | Precision | Recall | Positive F1 | Macro F1 | FP events/min | Event recall ±5 s | Peak MAE s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Существующий MobileNet | .500 | .619 | .553 | .458 | 1.033 | .364 | 7.133 |
| Исправленный R3D-18 | .526 | .952 | .678 | .387 | 1.827 | .500 | 8.047 |

Confusion `[TN,FP; FN,TP]`: baseline `[[6,13],[8,13]]`, R3D `[[1,18],[1,20]]`.
У R3D остаются 18 ложноположительных видео и одно ложноотрицательное.
Без неоднозначного дубликата event recall=.550, MAE=5.944 s; это метрики по
точкам, не подтверждённым полным интервалам. Уменьшение FN не оправдывает
автоматическую замену модели при таком числе FP.

На development calibration fold (3 positive / 4 negative) получается threshold=0:
все семь видео положительные. Recall >= .75 здесь требует всех трёх попаданий.
Этот результат явно отмечается как вырожденная калибровка, а не успех detection.

## Устойчивость и численная чувствительность

Baseline дополнительно проверен на CPU (batch=2) с теми же
**CUDA-calibrated thresholds** (CUDA scoring batch=8).
Это проверка переноса backend, поэтому clean результат отличается от CUDA OOF.
Одно отрицательное видео переходит через threshold: score .4184977785 (CUDA)
против .4185324162 (CPU), threshold .4185245261. Максимальная разница max-score
на 40 видео — .0000956133. Причина конкретного расхождения на уровне GPU kernels
не установлена; это измеренная чувствительность near-threshold, не decode error.

| CPU condition | Macro F1 | Recall | FP events/min | Изменённых вердиктов / 40 |
|---|---:|---:|---:|---:|
| Clean | .427 | .619 | 1.112 | 0 |
| Яркость x0.65 | .520 | .762 | 1.112 | 6 |
| Яркость x1.35 | .469 | .810 | 1.271 | 7 |
| Смещение 5% вправо/вниз | .549 | .571 | .715 | 6 |
| JPEG quality 40 | .345 | .810 | 1.827 | 8 |

Порог не подстраивался под преобразования. JPEG применяется после resize;
смещение использует reflected border. Это синтетические проверки, а не
доказательство устойчивости к новой камере. Интервалы bootstrap условны на
уже обученных OOF-моделях и не учитывают неопределённость выбора архитектуры.
Другие perturbations команды приведены в [scorecard_followup.md](scorecard_followup.md).
Полный stress-suite R3D не запускался: кандидат уже не прошёл clean quality gate.

## Классификация

Достоверных event-type labels и подтверждённой taxonomy нет. Модель остаётся
бинарной `no_violation / possible_violation`; multiclass accuracy не заявляется.
Для критерия 3 нужны локально подтверждённые типы и интервалы действий;
план согласования и разметки сохранён в командном scorecard follow-up.

## Воспроизведение

Использовать Python 3.12, существующие ML requirements, PyTorch 2.5.1+cu121 /
torchvision 0.20.1. В этой проверке использовалась RTX 3050 Ti 4 GB.
Manifest, видео и frozen plan остаются приватными. Значения QAITU_* задаются
локальными путями; output должен быть новым каталогом.

```powershell
$env:PYTHONPATH = 'src'
$py = '.venv-ml/Scripts/python.exe'
$plan = "$env:QAITU_FROZEN_PLAN"
& $py -m fortebank_audit.ml.experiments --stage oof --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan $plan --config configs/ml/r3d18_scorecard.yaml --output artifacts/ml/r3d-oof-repro
& $py -m fortebank_audit.ml.stress_audit --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan $plan --runs "$env:QAITU_BASELINE_RUNS" --calibrations "$env:QAITU_BASELINE_CALIBRATIONS" --output artifacts/ml/baseline-stress-repro
& $py -m pytest tests/ml -q
```

Для воспроизведения именно CPU stress перед его командой установить
`$env:CUDA_VISIBLE_DEVICES = '-1'`; затем удалить эту переменную перед GPU-обучением.
`QAITU_BASELINE_RUNS` содержит outer_N/training/best.pt, а
`QAITU_BASELINE_CALIBRATIONS` — outer_N/calibration.json исходной CUDA-оценки.

## Экспорт и интеграция

Bundle: `artifacts/ml/bundles/r3d-scorecard-fold0`. Это checkpoint заранее
выбранного outer fold 0, а не final-модель, обученная на всех доступных folds.
Его train folds: 2/3/4, calibration fold: 1. Именование и статус намеренно
не позволяют считать его продвинутой заменой текущей поставки.

Внешний input: RGB float32 NCTHW `[1,3,16,112,112]`, ImageNet normalization.
Внутри R3D-графа выполняется перевод в Kinetics normalization.
Выход: один `logits`, sigmoid один раз; window=6, stride=2, обязательный tail.
Демонстрационный threshold=0 помечает все видео положительными; использовать
его для полезных пользовательских вердиктов нельзя. Canonical и dynamic graphs
нужны для воспроизводимой технической проверки новой архитектуры.

Команды экспорта и adapter smoke:

```powershell
$run = 'artifacts/ml/r3d-oof-repro/outer_0'
$bundle = 'artifacts/ml/bundles/r3d-scorecard-repro'
& $py scripts/ml/export_model.py --checkpoint "$run/training/best.pt" --calibration "$run/calibration.json" --metrics artifacts/ml/r3d-oof-repro/summary.json --output-dir $bundle --model-version r3d-scorecard-fold0-development --code-commit 0b0ba91 --parity-video "$env:QAITU_POSITIVE_VIDEO" "$env:QAITU_NEGATIVE_VIDEO"
& $py -m fortebank_audit.ml.adapter_smoke --reference-commit 2f53ce18a91bcddae5135f2cac0a03aa36c7e7d4 --video "$env:QAITU_NEGATIVE_VIDEO" "$env:QAITU_POSITIVE_VIDEO" --bundle $bundle --output artifacts/ml/r3d-adapter-repro --evidence-root "$env:QAITU_EVIDENCE_ROOT"
```

Веса, индивидуальные scores, evidence и исходные labels не добавляются в Git.
API, frontend и чужой inference checkout не меняются. Общий API SLA не измерялся.

## Фактически выполненные проверки

- Исходный набор: 52 passed; после собственных изменений: 65 passed;
  после объединения с `57a5e4e`: **74 passed in 18.75 s**.
- Пять R3D OOF прогонов и отдельный development-прогон; прежние epochs и folds
  не подбирались по outer-результатам. Всего 200 дополнительных CPU baseline
  source-condition оценок с зафиксированными порогами.
- Существующий v4 bundle загружается и проходит schema/checksum validation.
- ONNX fixed N=1 и dynamic N=12: 12 реальных окон, max logit error
  `5.066394805908203e-7` при tolerance `1e-4`; решения по порогу совпали.
- Реальный adapter `2f53ce1`: positive и negative calibration-видео;
  max window score difference `4.330307362687691e-8`, timestamp difference=0.
  JPEG открываются; corrupted video отклонён. Время pipeline: 5.75 и 2.92 s
  для этих двух примеров. Это технический smoke, не качество или общий SLA.
- `git diff --check` пройден. Все исходные командные изменения сохранены.

ONNX SHA-256: `178da5deebbd87f687e02c31c9af90caff96b445e22fb53122ccb9ab14dbc4fa`.
SHA архива и агрегированный release record находятся в JSON рядом с отчётом.
