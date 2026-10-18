# Qwen → API: integration contract v1

Владелец границы — тимлид. Совместимость проверяется по интерфейсу
`origin/feature/video-inference@6933204`; улучшенные веса принимает тот же
inference backend. Публичная схема жюри не меняется.

## Точка подключения

При `QAITU_VLM_MODE=off` используется прежний ONNX-путь; Qwen не импортируется.
При `qwen` startup загружает `fortebank_audit.inference.semantic_pipeline` и
один раз вызывает `build_semantic_verifier_from_env()`. Он обязан проверить
локальный snapshot/manifest/checksum и выполнить preload. Backend предоставляет
непустой `model_version` и lowercase SHA-256 `model_checksum`.

Далее каждый job под общим semaphore вызывает:

```python
analyze_video_with_semantic_verification(
    video_path, predictor, config, verifier,
    evidence_root=evidence_root, run_id=job_id,
)
```

Inference-код должен быть передан целиком с поддержкой
`analyze_video(..., include_temporal_context=True)`. Подмена только весов при
отсутствующем semantic pipeline не является готовым handoff. Текущий запуск
остаётся ONNX-only до приёмки улучшенного Qwen.

## Результат

- `events`: оставшиеся события; один элемент — один эпизод, не один JPEG.
- `verdict`: `possible_violation` при непустом списке, иначе `no_violation`.
- Каждый event сохраняет start/end/peak, score, supporting_windows, 1–3 evidence.
- `semantic_verification`: decision (`possible_violation` либо
  `insufficient_evidence`), category, confidence, thesis, observations, reasoning,
  risk, limitations, matched_signal_ids, rule_candidate, model_version/checksum.
- Observation: frame_role (`before/start/peak/end/after`), конечный неотрицательный
  timestamp_sec, непустой visible_fact. Inference сверяет роль и timestamp с
  реально переданным кадром; API не восстанавливает их из текста модели.
- `diagnostics`: candidate_events, rejected_candidates, retained_candidates,
  qwen_called. Counts согласованы; `qwen_called=false` только при 0 кандидатов.
- Если кандидаты были, `runtime.qwen.model_version/checksum` совпадают с загруженным
  backend, включая результат, где все кандидаты отклонены.

`normal_activity` не допускается в оставшихся events. Безопасность удаления
кандидатов и recall guard остаются ответственностью inference. API не отбрасывает
события самостоятельно и не восстанавливает их после ошибки Qwen.

Тезис, наблюдения с таймкодами, reasoning, risk и limitations попадают в description
публичного результата и соответствующие поля внутренних карточек. Категория и
confidence Qwen не превращаются в доказанный юридический вывод.

## Правила

`rule_candidate=null` полностью совместим с текущим handoff. При наличии кандидата
принимаются только существующий rule_id, точные document_name/section из
`GET /internal/api/v1/rules` и `certainty=candidate`. Ссылка берётся из локального
проверенного каталога. Статус остаётся `not_established`, интерфейс показывает
предложение Qwen для проверки. Одного rule_candidate недостаточно для matched.
Автоматический слой тимлида повторно использует локальный backend и добавляет
структурированную оценку всех условий с привязкой к кадрам. Только полный набор
supported/high без ограничений даёт автоматический matched. Несколько правил
сохраняются в `AnalysisDetail.rules`; первое дублируется в прежнем `rule`.
Публичный description содержит все основания и источники. Ручное подтверждение
также доступно. Подробности: [automatic_rules_contract.md](automatic_rules_contract.md).

## Ошибки и приёмка

Отсутствующий код/веса, неверный manifest, CUDA/OOM, неверный JSON, потерянная
семантика события, несовпадение модели или counts — техническая ошибка.
Startup прекращается либо job получает failed/null/[]; fallback на ONNX
внутри включённого режима Qwen запрещён. Переключить весь сервис обратно на
ONNX можно явно через `QAITU_VLM_MODE=off` при следующем запуске.

Команды приёмки: [руководство](../team/QWEN_INTEGRATION_RUNBOOK.md).
