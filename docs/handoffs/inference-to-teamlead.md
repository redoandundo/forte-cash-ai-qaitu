# Inference -> Teamlead handoff

Статус: полный локальный прогон 42/42 завершён; recall-защита прошла, но Qwen
остаётся за feature flag из-за слабой семантической дискриминации positive-кандидатов.

## Версия и изменения

- Branch: `feature/video-inference`.
- Implementation: `93a1c0c` (`feat(inference): add Qwen semantic event verification`).
- Contract tests: `1a9d02d`; startup/count tests: `0320c78`.
- Full-dataset recall guard and evaluator: `f1b7769`.
- ONNX: `mobilenet-temporal-pairsafe-v3`; его threshold, smoothing, sampling и
  `max_events` не менялись.
- Qwen backend создаётся один раз через `get_qwen_backend()` в
  `src/fortebank_audit/inference/vlm_verifier.py` и кэшируется на процесс.
- Вызов после ONNX находится в
  `src/fortebank_audit/inference/semantic_pipeline.py`:
  `analyze_video_with_semantic_verification()` -> `apply_semantic_verification()`.
- Только recall-safe `normal_activity/high` удаляет исходный кандидат;
  `possible_violation`, `insufficient_evidence` и небезопасный normal сохраняют
  тот же объект события. Evidence JPEG не превращаются в отдельные события.

Изменены manifest/requirements, `vlm_verifier.py`, `semantic_pipeline.py`,
`pipeline.py`, `evidence.py`, `meta_cli.py`, `reporting.py`, `ab_evaluation.py`,
`scripts/inference/qwen_full_evaluation.py` и inference tests. Веса, видео,
evidence, labels и приватные отчёты не добавлены.

## Контракт и failure semantics

Qwen получает 1-5 уникальных хронологических кадров с ролями
`before|start|peak|end|after`, точными timestamp, event interval, peak,
ONNX ranking score, supporting-window count и activity duration. Public evidence
остаётся 1-3 JPEG.

Строгий JSON содержит только: `decision`, `category`, `confidence`, `thesis`,
`observations`, `reasoning`, `risk`, `matched_signal_ids`, `rule_candidate`,
`limitations`. Markdown, лишние поля, неизвестные enum/signal/rule ID, пустые
тексты и несовпадающие timestamps отклоняются. Разрешён один repair retry с
точной причиной. Второй невалидный ответ завершает job технической ошибкой.

Проверенного каталога правил нет, поэтому `rule_candidate` обязан быть `null`,
а результат содержит «Нормативное основание автоматически не установлено».
Ошибка загрузки/checksum/CUDA/OOM/JPEG/schema не становится ни
`no_violation`, ни положительным verdict. Если ONNX events пусты, генерация Qwen
не вызывается.

## Runtime

Переменные окружения:

- `QAITU_VLM_MODE=qwen`;
- `QAITU_VLM_MODEL_PATH=<local snapshot>`;
- `QAITU_VLM_MANIFEST_PATH` (optional);
- `QAITU_VLM_SIGNALS_PATH` (optional);
- `QAITU_VLM_MAX_CONCURRENT_JOBS=1` (default).

Snapshot: `models/pretrained/qwen3-vl-2b-instruct`, один файл
`model.safetensors`, 4,255,140,312 bytes, SHA-256
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
Manifest поддерживает только этот фактически проверенный layout.

Проверенный runtime: Python 3.12.14; PyTorch 2.11.0+cu128; Transformers 5.20.1;
CUDA 12.8; RTX 5050 Laptop GPU, 8151 MiB VRAM. Backend работает local-only,
ограничен одним одновременным job, остаётся в памяти и логирует backend/model/
checksum/device/load/inference/peak VRAM без кадров и чувствительных путей.

## Проверки

`python -m pytest tests/inference -q` -> `59 passed`. Покрыты фильтрация всех
трёх decisions, 0/1/2/3 кандидата, один event с тремя JPEG, отсутствие вызова
без events, singleton backend, malformed JSON after retry, ordinary entry,
inactivity temporal rule, unknown signals/rules, Unicode evidence, обе модели в
diagnostics, missing/bad weights, CUDA/version checks, damaged JPEG/video и
decoder failure. `git diff --check` чист.

### Финальный полный прогон 42 файлов

Локальный набор содержит 42 файла и 40 уникальных источников: `Х-4/Х-47` и
`Х-5/Х-48` являются byte-identical парами. Детальные локальные результаты,
видео и evidence не коммитились.

| Уровень подсчёта | Этап | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|
| 42 файла | ONNX до Qwen | 14 | 10 | 10 | 8 |
| 42 файла | ONNX + Qwen recall guard | 14 | 13 | 7 | 8 |
| 40 уникальных источников | ONNX до Qwen | 14 | 10 | 9 | 7 |
| 40 уникальных источников | ONNX + Qwen recall guard | 14 | 12 | 7 | 7 |

- ONNX candidates: 29; retained: 24; removed: 5.
- Удалённых событий на positive-видео: 0; positive-видео, превращённых в
  negative: 0. Recall относительно ONNX не ухудшился.
- Unrecovered schema failures: 0; technical failures: 0; пять ответов потребовали
  разрешённый один repair retry и после него прошли строгую schema.
- Максимальное время одного видео: 127.05 s; весь batch: 1372.54 s; лимит
  300 s на видео соблюдён.
- Peak allocated VRAM: 5.13 GiB; load time: 7.71 s. Модель загрузилась один раз
  и переиспользовалась для всех 42 файлов.
- Обычные negative-примеры `Х-5/Х-48` и `Х-27` стали `normal_activity/high`
  и были отфильтрованы.
- Positive end-to-end примеры `X-1`, `Х-11`, `Х-15`, `Х-26`, `Х-28`, `Х-30`
  сохранили события. Для них небезопасный `normal_activity` был понижен до
  `insufficient_evidence`, потому что не выполнен полный recall-safe gate.
- Несколько кандидатов проверены на `Х-12`, `Х-30`, `Х-34`, `Х-35`, `Х-39`;
  количество событий не дополнялось и не превращалось в число JPEG.

Recall-safe удаление требует одновременно: `normal_activity`, `high`, минимум
четыре supporting windows, все пять ролей, отдельные кадры строго до и после
интервала, наблюдения по всем ролям и отсутствие material limitations. Любое
сомнение превращается во внутренний `insufficient_evidence`, кандидат остаётся.

Ограничение: на финальном прогоне raw Qwen назвал все 29 кандидатов
`normal_activity/high`. Positive-кандидаты сохранены защитным правилом, а не
надёжным распознаванием possible violation. Поэтому нельзя утверждать, что Qwen
научился распознавать нарушения; измерено только уменьшение FP без потери recall
на этом локальном наборе.

## Quality gate и ограничения

Измеренный file-level результат улучшился с FP 10 до 7 без изменения TP/FN,
но не включать Qwen публично по умолчанию без review тимлида:

- три FP-видео удалены, остальные семь FP остались;
- raw Qwen не распознал ни один финальный кандидат как possible violation;
- recall сохранился только благодаря жёсткому deterministic guard;
- пять первых ответов потребовали repair, хотя unrecovered failures равны нулю;
- Qwen не исправляет ONNX false negatives и не ищет события самостоятельно;
- юридический вывод и вина не устанавливаются.

## Команда воспроизведения

```powershell
$env:PYTHONPATH = "src"
$env:QAITU_VLM_MODE = "qwen"
$env:QAITU_VLM_MODEL_PATH = "models/pretrained/qwen3-vl-2b-instruct"
.\.venv-ml\Scripts\python.exe scripts\inference\qwen_full_evaluation.py `
  --qwen-model "models/pretrained/qwen3-vl-2b-instruct" `
  --output "tmp/qwen-full-evaluation.json" `
  --evidence-root "data/raw/evidence/qwen-full-evaluation" `
  --run-tag "teamlead-full-evaluation"
```

## Что должен сделать тимлид в API

1. При startup и только когда `QAITU_VLM_MODE=qwen` вызвать
   `build_semantic_verifier_from_env()`; startup failure не скрывать.
2. Заменить прямой `analyzer.analyze`/статический текст на
   `analyze_video_with_semantic_verification(video, predictor, config, verifier)`.
3. Преобразовать внутренний `possible_violation|no_violation` в публичный
   `violation|no_violation`; semantic thesis/reasoning/risk/limitations положить
   в описание карточки, не создавать event на каждый JPEG.
4. Любые `VerificationRuntimeError`, `VerificationOutputError`, decode/model/
   evidence exceptions маппить только в `status=failed`, `verdict=null`, пустой
   список violations. Не делать fallback на статический текст/ONNX verdict.
5. Ограничить одновременные jobs тем же semaphore/job worker limit и не создавать
   backend на запрос. Оставить Qwen выключенным до повторного финального holdout
   без schema failures и без удаления TP.
