# Текущий сайт и отдельный GPU-компьютер

Используется текущая teamlead-сборка, а не старый `START_SITE.cmd` из inference-ветки.
Сайт хранит собственные browser sessions и jobs; сервер отправляет видео на
team-owned worker по API, читает его progress и полное досье. Ключ worker хранится
только на сервере сайта. Адреса с ключом в URL не используются.

## На GPU-компьютере участника

Нужны проверенное CUDA/Python 3.12 окружение участника, локальный базовый Qwen,
точный ONNX v4 bundle и эта feature-сборка. `requirements-qwen.txt` описывает
дополнительные пакеты; скрипт запуска ничего не скачивает.

```powershell
# Запускать Python из уже работающего GPU-окружения участника.
python scripts/serve_gpu_worker.py --model models/pretrained/qwen3-vl-2b-instruct
```

По умолчанию запускается ONNX v4 + базовый semantic Qwen + текущие проверки правил.
Поддерживается `START_GPU_WORKER.cmd`, если окружение находится в `.venv-run`.
Если пути другие, передать `--model` и `--bundle` явно. ONNX может работать на CPU
этого же компьютера; Qwen требует CUDA. Это позволяет не устанавливать конфликтующие
варианты ONNX Runtime в уже проверенное CUDA-окружение.

Альтернативный эксперимент с **обученным бинарным LoRA**:

```powershell
python scripts/serve_gpu_worker.py --sequence --allow-experimental `
  --adapter models/qwen-sequence-adapter
```

Он использует ONNX для эпизодов, а обученный Qwen только добавляет оценку всего
ролика. Он не подтверждает правило, не создаёт таймкод и не отменяет ONNX-события.
Одновременно с semantic Qwen второй GPU-моделью не загружается. Вариант обучения
содержит 16 кадров и ответ A/B; quality gate не пройден (test TP=3, TN=1, FP=4,
FN=1 на 9 видео; диагностический набор не является новым слепым тестом).

После успешной загрузки появится `secrets/tester-access.txt`: адрес worker и ключ.
Передать их тимлиду приватно. Компьютеры должны иметь прямую связь через общую
сеть/VPN; адрес `192.168…` сам по себе не доступен через интернет. Не пересылать
банковские видео во внешние AI-сервисы. Нельзя считать локальные contract tests
подтверждением качества или скорости реального Qwen.

## На сервере текущего сайта

Сохранить только ключ worker в игнорируемый `secrets/gpu-worker.key`.
Создать игнорируемый `secrets/gpu-worker.json`:

```json
{"url":"http://GPU_HOST:8000","key_file":"secrets/gpu-worker.key"}
```

Сначала проверить `/health`: `worker_protocol=qaitu-worker-v1`, правильный ONNX v4
SHA, semantic/sequence mode и ожидаемые checksums. Старую API-сборку bridge отклоняет.
Затем перезапустить текущий сайт с сохранением туннеля:

```powershell
./scripts/start_public_demo.ps1 -PythonExecutable <SITE_PYTHON> `
  -CloudflaredExecutable <CLOUDFLARED> -RestartApiOnly `
  -WorkerConfigPath secrets/gpu-worker.json -VlmMode qwen
```

Для экспериментального бинарного LoRA передать `-VlmMode off`, потому что это не
semantic-модель. Jury key текущего сайта остаётся прежним; ключ worker независим.
Доступ к его ключу у браузера отсутствует. Разрыв связи, неверный ответ/досье,
failed у worker или превышение 300 секунд дают failed/null/[], без локального fallback.
Не включать bridge до получения реального адреса и успешной проверки GPU inference.

Для локального отката запустить прежнюю команду без `-WorkerConfigPath` и с
`-VlmMode off`. Перезапуск прерывает незавершённые jobs.

## Приёмка

На GPU выполнить реальные positive/negative/multiple-event ролики локально,
подтвердить `qwen_called`, checksums, досье/правила, JPEG и время до 300 секунд.
Для semantic Qwen нужны проверки ложного исключения ONNX-кандидатов.
Обученный бинарный LoRA не считается улучшенной моделью объяснений по эпизодам.
Источник ограничений: inference branch `94fd1d6`,
`docs/handoffs/qwen-training-runtime.md` и `reports/ml/qwen_sequence_training.md`.
