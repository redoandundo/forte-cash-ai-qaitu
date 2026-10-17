# Inference -> Teamlead handoff

Статус: implementation ready; quality promotion blocked by measured detector/VLM errors.

## Версия и изменения

- Branch: `feature/video-inference`.
- Implementation: `93a1c0c` (`feat(inference): add Qwen semantic event verification`).
- Contract tests: `1a9d02d`; startup/count tests: `0320c78`.
- ONNX: `mobilenet-temporal-pairsafe-v3`; его threshold, smoothing, sampling и
  `max_events` не менялись.
- Qwen backend создаётся один раз через `get_qwen_backend()` в
  `src/fortebank_audit/inference/vlm_verifier.py` и кэшируется на процесс.
- Вызов после ONNX находится в
  `src/fortebank_audit/inference/semantic_pipeline.py`:
  `analyze_video_with_semantic_verification()` -> `apply_semantic_verification()`.
- `normal_activity` удаляет один исходный кандидат; `possible_violation` и
  `insufficient_evidence` сохраняют тот же объект события. Evidence JPEG не
  превращаются в отдельные события.

Изменены manifest/requirements, `vlm_verifier.py`, `semantic_pipeline.py`,
`pipeline.py`, `evidence.py`, `meta_cli.py`, `reporting.py`, `ab_evaluation.py`
и inference tests. Веса, видео, evidence, labels и приватные отчёты не добавлены.

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

`python -m pytest tests/inference -q` -> `54 passed`. Покрыты фильтрация всех
трёх decisions, 0/1/2/3 кандидата, один event с тремя JPEG, отсутствие вызова
без events, singleton backend, malformed JSON after retry, ordinary entry,
inactivity temporal rule, unknown signals/rules, Unicode evidence, обе модели в
diagnostics, missing/bad weights, CUDA/version checks, damaged JPEG/video и
decoder failure. `git diff --check` чист.

Реальные smoke на RTX 5050 (локальные details не коммитились):

- Unicode `Х-15.mov`, обычный вход: ONNX 1 -> Qwen `normal_activity` -> final 0;
  Qwen 29.11 s, total warm-job 31.58 s, peak 4.99 GiB, status done.
- `Х-35.mov`, known FP, long 56.77 s, two candidates: Qwen удалил один как
  normal, второй сохранил как possible; final 1; Qwen 42.40 s, total 106.48 s,
  peak 5.06 GiB, status done. Known FP удалён не полностью.
- `Х-20.mov`, short 11.63 s: ONNX 0, `qwen_called=false`, final 0, warm-job
  0.45 s, status done.
- `Х-22.mov`, known ONNX FN: ONNX 0, `qwen_called=false`, final 0, warm-job
  0.82 s. Второй слой не может искать пропущенное событие по архитектуре.
- Изолированный positive control на размеченном temporal context `Х-26.mov`:
  Qwen `possible_violation/hand_or_object_transfer`, valid strict JSON, 23.16 s,
  5.00 GiB. Но ONNX локализует этот ролик позже размеченного момента, поэтому
  end-to-end positive-control gate пока не доказан.
- Повреждённые video/JPEG дают controlled technical failure. Schema-failure
  smoke также подтвердил `failed`, не `no_violation`.

Все измеренные video jobs меньше 300 s. Модель в одном holdout-процессе была
загружена один раз и переиспользована для последующих событий/jobs.

## Quality gate и ограничения

Не утверждать улучшение системы и не включать Qwen публично по умолчанию:

- known FP удалён лишь частично;
- несколько размеченных TP имеют неверный ONNX timecode и выглядят для Qwen как
  обычный вход, поэтому могут быть удалены;
- full holdout до финальной prompt/repair настройки имел schema failures;
- финальный выбранный smoke не имел schema failures, но это недостаточно для
  promotion;
- Qwen не исправляет ONNX false negatives и не ищет события самостоятельно;
- юридический вывод и вина не устанавливаются.

## Команда воспроизведения

```powershell
$env:PYTHONPATH = "src"
$env:QAITU_VLM_MODE = "qwen"
$env:QAITU_VLM_MODEL_PATH = "models/pretrained/qwen3-vl-2b-instruct"
.\.venv-ml\Scripts\python.exe -m fortebank_audit.inference.meta_cli analyze `
  --video "data/raw/open/videos/Х-15.mov" `
  --enable-qwen `
  --vlm-model "models/pretrained/qwen3-vl-2b-instruct" `
  --run-id "teamlead-qwen-smoke"
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
