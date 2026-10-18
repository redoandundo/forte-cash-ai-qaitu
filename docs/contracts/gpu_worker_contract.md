# GPU worker protocol qaitu-worker-v1

Текущий публичный API не меняется. Bridge использует `POST /api/v1/analyze`,
`GET /api/v1/jobs/{id}`, `GET /internal/api/v1/jobs/{id}/progress` и защищённый
`GET /internal/api/v1/rules` с отдельным `X-API-Key` worker.

`GET /health` обязан содержать `worker_protocol=qaitu-worker-v1`, status=ok,
model_ready=true, inference_mode=onnx и прежние сведения о checksum/mode моделей.
Worker не может сам иметь execution_location=remote: цепочки bridge запрещены.
Ошибки старой сборки, auth, соединения, несовпадение job_id и неполное досье
прерывают анализ; fallback не предусмотрен. Общий бюджет ожидания — 300 секунд.

На сайте сохраняется свой job_id, browser ownership, cookie и ручная проверка.
Полученное досье должно содержать ровно один detail на каждый event, в том же
порядке. Вердикт согласован с наличием событий. Ответы проходят Pydantic-валидацию.
Endpoint worker, его ключ и его job_id не добавляются к клиентскому досье.

semantic_mode=qwen означает semantic Qwen. Отдельные поля sequence_mode,
sequence_adapter_sha256 и sequence_quality_gate_passed описывают бинарный LoRA.
Этот режим не объявляет правила matched и не заменяет ONNX verdict/timecodes.
Выбор режима, запуск и приёмка: `docs/team/GPU_WORKER_RUNBOOK.md`.
