# Подключение улучшенного Qwen

Наша API-граница готова под semantic interface из `feature/video-inference@6933204`.
Автоматический слой правил использует тот же backend через wrapper; участнику
не нужно менять свой основной JSON или обучать отдельную модель для этого слоя.
Для possible_violation/high с полным контекстом добавляется отдельная локальная
оценка условий. Её время и ошибки входят в общий результат; проверка 300 секунд
обязательна уже с включённым wrapper.
Это не подтверждение работы новых весов: они ещё должны пройти приёмку.
В текущую рабочую копию новый inference pipeline не импортирован: его исходная
ветка меняет также evidence/predictor/pipeline. Нужен согласованный merge с
сохранением исправлений ONNX v4 и декодера, а не копирование одного файла.

## Что получить от участника

1. Commit SHA inference-кода и актуальный handoff.
2. Локальный snapshot: веса, processor/tokenizer/config, manifest, checksums.
3. Точный Python/CUDA/PyTorch/Transformers runtime и команду запуска.
4. Результаты проверки JSON, 0/1/нескольких событий, recall guard, времени и VRAM.

Если передаётся LoRA/adapter, участник должен предоставить совместимый loader
или merged snapshot; путь к адаптеру нельзя молча принять за полную модель.
Manifest должен описывать новые веса, а не checksum старого Qwen.

## Приёмка перед публичным включением

Использовать окружение участника с проверенной CUDA и достаточной VRAM.
После интеграции его кода и помещения snapshot вне Git:

```powershell
$env:PYTHONPATH = 'src'
$env:QAITU_INFERENCE_MODE = 'onnx'
$env:QAITU_MODEL_BUNDLE = 'artifacts/ml/bundles/baseline-audited-v4'
$env:QAITU_ALLOW_DEVELOPMENT_MODEL = 'true'
$env:QAITU_EXPECTED_MODEL_SHA256 = '00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d'
$env:QAITU_PREFER_CUDA = 'false'
$env:QAITU_MAX_CONCURRENT_JOBS = '1'
$env:QAITU_VLM_MODE = 'qwen'
$env:QAITU_VLM_MODEL_PATH = 'models/pretrained/qwen-improved'
$env:QAITU_VLM_MANIFEST_PATH = 'configs/inference/qwen_improved_manifest.json'
$env:QAITU_VLM_MAX_CONCURRENT_JOBS = '1'
python scripts/check_qwen_integration.py
python scripts/check_qwen_integration.py --video <LOCAL_VIDEO.mov>
python -m pytest tests/api tests/integration tests/inference -q
```

Пути snapshot/manifest выше — пример: заменить путями фактически полученного
bundle. Первый вызов проверяет startup; второй выполняет настоящий анализ и
проверяет лимит 300 секунд. `startup_passed` не означает проверку распознавания.
Прогнать positive, negative и несколько эпизодов; сравнить число событий и
таймкоды с принятым handoff, проверить отсутствие потери TP. При 0 ONNX-кандидатов
Qwen не вызывается — такой ролик не доказывает работу его генерации.
Для правил отдельно проверить полный/неполный обзор, уход после завершения
обслуживания, клиента вне кадра и деньги текущего клиента. API должен показывать
matched только при полном доказательном наборе; один category/пустой стол не
считаются основанием. Проверить два правила на одном эпизоде и ручной override/reset.

## Запуск демо после приёмки

Сначала штатно остановить предыдущий демо-процесс. Это прерывает текущие jobs,
поэтому выполнять между демонстрациями. Затем:

```powershell
./scripts/start_public_demo.ps1 -PythonExecutable <QWEN_PYTHON> `
  -CloudflaredExecutable <CLOUDFLARED> -VlmMode qwen `
  -QwenModelPath models/pretrained/qwen-improved `
  -QwenManifestPath configs/inference/qwen_improved_manifest.json
```

`/health` должен показать semantic_mode=qwen и ожидаемый semantic_model_sha256.
Новый временный URL читать из `tmp/public-demo/session.json`; ключ остаётся в
`secrets/public-demo-api-key.txt`. Для отката запускать с `-VlmMode off`.
Запуск с qwen не исправляет отсутствующий inference-код и не скачивает веса.

Детали API и границы автоматизации правил:
[semantic_integration_contract.md](../contracts/semantic_integration_contract.md).
