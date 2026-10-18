# ML/Inference integration status

Дата: 2026-10-18.

Статус: точный `baseline-audited-v4.zip` принят, проверен и подключён к полному
API/UI пути. Технический end-to-end на реальных видео подтверждён. Модель остаётся
development-only и не проходит quality gate.

## Входные версии

### Веб-доступ без ключа — 18.10.2026

База: `e818f28`. Блок доступа удалён из HTML/JS. Сайт автоматически создаёт
HttpOnly/SameSite-сессию и работает через `/web/api/v1/*`. Старый API-ключ из
sessionStorage удаляется. Jury API и старые внутренние маршруты сохраняют
X-API-Key; cookie не заменяет ключ на `/api/v1/*`.

Web jobs привязаны к сессии, чужой job/progress/rule-review возвращает 404.
Добавлены проверки origin/custom header, срок сессии и ограничение очереди.
Обработка и формат результата используют тот же локальный runtime.

Проверки:

- `python -m pytest tests/api tests/integration -q` — 95 passed,
  одно существующее предупреждение Starlette/httpx;
- `node --test tests/web/app.test.cjs` — 15 passed;
- JS syntax, PowerShell parse обоих scripts, git diff check — passed;
- публичный smoke: passed. Синтетический ролик 9 сек. через jury API — 1.812 сек.,
  через веб без API-ключа — 1.037 сек.; испорченный ролик — failed за 0.437 сек.;
  API без ключа — 401, изоляция browser sessions — passed. Датасет не загружался.
- Chrome: на публичной странице нет блока/поля ключа. API перезапущен с актуальным
  кодом через `-RestartApiOnly`, адрес туннеля сохранён. Исправлена проверка времени
  процессов для PowerShell, где JSON timestamp автоматически становится DateTime.

Исходный PDF прочитан локально: FAQ 27 не требует пользовательской авторизации;
раздел 10 требует X-API-Key для API жюри. Vercel не подключался; UI hosting не
заменяет локальный runtime. Разделы 7/10 сохраняют ограничения на облачный AI,
передачу тестового видео третьим лицам и автономную обработку после загрузки весов.

### Исправление ввода ключа доступа — 18.10.2026

База: `199f98fb5ed13a807daf49348ce4aba929c32a36`. Пустой ключ ошибочно переводил
выбранное видео в failed с текстом о технических настройках. Исправлено: поле
доступа раскрыто без сохранённого ключа, missing/401 показывают ошибку рядом с
полем и возвращают ready без загрузки видео. Выбранный файл сохраняется;
ключ запоминается по input, включая вставку без blur. Авторизация API сохранена.

Проверки: `node --test tests/web/app.test.cjs` — 15 passed (два новых regression
теста сначала воспроизвели failed вместо ready); `node --check web/app.js`,
`git diff --check` — passed. В открытой Chrome-вкладке публичного сервиса проверены
новая форма, заполнение существующего ключа и его сохранение после reload.
Никакое видео в ходе браузерной проверки не отправлялось. Статические файлы
публичного сервиса обновились без перезапуска API.

### Автоматическое сопоставление правил — 18.10.2026

База этапа: `a364740e039aad6b2f79491ea0f2b7f13ca56ff6`. Добавлен слой тимлида
`automatic_rules.py` и контракт `docs/contracts/automatic_rules_contract.md`.
Три проверенных правила оцениваются через восемь явно определённых условий.
RuleAwareVerifier сохраняет интерфейс участника и повторно использует его local
backend для оценки условий по тем же кадрам; чужие inference исходники не менялись.

Полный supported/high набор, подтверждённые ссылки на временной контекст и
отсутствие ограничений позволяют коду выбрать нормативный источник автоматически.
Одного названия события или rule_candidate недостаточно. При недостаточных
условиях — not_established; неисправимая schema/backend ошибка — failed/null/[].
В одном эпизоде поддерживаются несколько правил; публичный description содержит
все основания и ссылки. Отмена ручного override восстанавливает автоматический
результат. Текущий публичный ONNX процесс не перезапускался; новый слой там выключен.

Фактические проверки:

- `python -m pytest -q` — 156 passed, одно предупреждение Starlette/httpx;
- `python -m pytest tests/api tests/integration -q` — 87 passed;
- `node --test tests/web/app.test.cjs` — 12 passed;
- `node --check web/app.js` и `git diff --check` — passed;
- replay `tmp/check_existing_qwen_handoff.py` с настоящим semantic pipeline
  commit `6933204`: 0/1/2/3 события; отдельно wrapper с двумя эпизодами и двумя
  правилами на каждом — passed, ответы модели синтетические;
- в браузере проверено отображение двух автоматических оснований и официальных
  ссылок на синтетической странице. Изображение не является проверкой модели.

Открыто до реального Qwen: качество визуальных условий (особенно отсутствие
клиента, принадлежность денег, завершение обслуживания), новые ракурсы и общий
лимит 300 секунд с дополнительной генерацией/возможным одним repair. Максимальные
баллы или полный охват нормативной базы не заявляются. Следующий шаг — приёмка
совместимого snapshot по обновлённому QWEN_INTEGRATION_RUNBOOK.md и проверка всего
пути на локальных реальных примерах до публичного включения.

### Передача командной версии — 18.10.2026

Изменения тимлида сохранены commit
`2bf2eaf6e656447e6c7bfacfe9e8eb6ce465ee1c` и отправлены в
`origin/feature/teamlead-ml-inference-integration`. Для коммита использована
подтверждённая текущая GitHub identity; глобальные Git-настройки не менялись.
Ранние записи ниже об отсутствии commit относятся к моменту выполнения этапов.

Перед передачей повторно выполнены:

- `python -m pytest -q` — 119 passed, одно предупреждение Starlette/httpx;
- `python -m pytest tests/api tests/integration -q` — 50 passed;
- `node --test tests/web/app.test.cjs` — 11 passed;
- `git diff --cached --check` — passed.

Передача в develop выполняется через PR. Это сохранение и review кода;
публичный процесс не перезапускался, Qwen по-прежнему выключен. Дальнейшая
интеграция модели: [руководство](../team/QWEN_INTEGRATION_RUNBOOK.md).

### Подготовка подключения улучшенного Qwen — 18.10.2026

Teamlead adapter добавлен в рабочую копию; базовый commit остаётся
`e2e0a4a12b6e1fdd39ac336a53202368ac14e34c`, новый commit не создан. Inference/ML
исходники участников не изменялись. Совместимый интерфейс прочитан из
`feature/video-inference@6933204`. Его новый semantic pipeline пока не импортирован
в эту ветку; improved snapshot не получен. Публичный процесс продолжает ONNX-only.

- Runtime явно выбирает off/qwen; verifier preload выполняется при startup,
  backend переиспользуется; общий semaphore охватывает ONNX и Qwen.
- Semantic output валидируется по модели, SHA-256, counts, verdict, обязательным
  наблюдениям. 0–3 события сохраняют раздельные карточки и 1–3 JPEG на событие.
- Тезис, наблюдения, reasoning, risk и limitations идут в публичный description
  и внутренние карточки. Несовместимость/ошибка даёт failed/null/[], без fallback.
- Предложение правила проверяется по каталогу и остаётся кандидатом для ручного
  подтверждения. Автоматическое юридическое сопоставление не заявляется.
- Добавлены health identity обоих слоёв, режим запуска `-VlmMode qwen`, локальная
  команда приёмки `scripts/check_qwen_integration.py`.

Проверки: `python -m pytest tests/api tests/integration -q` — 50 passed,
одно предупреждение Starlette/httpx. Первоначальные 12 ошибок новой fixture
исправлены передачей обязательных параметров PipelineConfig; повторный прогон
прошёл. `node --test tests/web/app.test.cjs` — 11 passed;
`node --check web/app.js`, оба git diff --check — passed.
PowerShell startup script — parse passed. Дополнительно локальный replay
`python tmp/check_existing_qwen_handoff.py` исполнил именно semantic pipeline
из commit `6933204` через новый runtime на 0/1/2/3 событиях — passed. Verifier
в replay синтетический; это проверка совместимости, не запуск весов Qwen.

Следующий шаг и точные команды:
[QWEN_INTEGRATION_RUNBOOK.md](../team/QWEN_INTEGRATION_RUNBOOK.md).
Принять согласованный inference commit, snapshot и manifest/checksums; выполнить
startup + реальный positive/negative/multi-event прогон, затем включить qwen
на публичном сервисе. Процесс сервиса в этой доработке не перезапускался.

### Доработка критериев тимлида — 18.10.2026

Рабочая ветка: `feature/teamlead-ml-inference-integration`.
Базовый commit: `e2e0a4a12b6e1fdd39ac336a53202368ac14e34c`.
Новые изменения пока не закоммичены и не отправлены: Git отклонил commit из-за
отсутствующих `user.name` / `user.email`. Identity не изменялась.

- Карточки пронумерованы; подписи JPEG показывают положение относительно пика
  модели по реальным timestamps. Кадры отсортированы для показа без изменения
  результата API; отсутствующие кадры не добавляются.
- Клик по маркеру, таймкоду или JPEG синхронизирует выбранную карточку и маркер.
  Выбор карточки работает даже при недоступном браузерном воспроизведении.
- Несогласованный completed JSON отклоняется, а не показывается как
  `no_violation`. Ответ отменённого запроса не перерисовывает другую запись.
- В интерфейсе добавлен справочник и ручная проверка трёх правил: пункты 20 и 22 Правил
  НБРК № 231 от 29.11.2019. Текст сверен 18.10.2026 на официальной странице
  https://adilet.zan.kz/rus/docs/V1900019680 . Для каждого случая указаны
  ограничения наблюдения. Проверяющий выбирает правило и подтверждает все
  необходимые наблюдения. Только после этого сохраняется `matched` с явной
  отметкой ручной проверки. Сброс восстанавливает исходное описание события.
  Автоматическое сопоставление и Qwen пока не подключены.
- Публичный HTTPS-доступ запущен через временный Cloudflare Tunnel с ONNX v4
  на CPU. Текущий адрес и процессы записаны локально в
  `tmp/public-demo/session.json`; ключ — в игнорируемом
  `secrets/public-demo-api-key.txt`. Адрес действует, пока работают компьютер
  и процессы. Трафик проходит через HTTPS-прокси; внешние AI API не используются.
- Публичная схема результата сохранена: ручная проверка дополняет description,
  не изменяя verdict, timecode и JPEG. Внутренние endpoints каталога и проверки
  требуют тот же X-API-Key.

Проверки этого изменения:

- API/integration: `20 passed`, одно предупреждение `fastapi.testclient` об
  httpx compatibility shim; запуск в существующем integration venv.
- `node --test tests/web/app.test.cjs`: `10 passed`.
- `node --check web/app.js`: passed.
- `git diff --check` и `git diff --cached --check`: passed.
- В браузере на локальной синтетической странице проверены две карточки,
  три кадра одного эпизода и один кадр другого, выбор маркера и раскрытие
  справочника. Синтетическая страница не является ML-демо.
- В браузере проверены сохранение и сброс правила через настоящий backend
  на синтетическом событии; неполный набор наблюдений блокирует сохранение.
- `scripts/smoke_public_api.py` на публичном адресе: passed. Синтетический
  ролик 9 секунд прошёл реальный ONNX-путь за 1.037 секунды от загрузки до
  результата; получены один эпизод и декодируемые JPEG. Это техническая
  проверка, не оценка качества распознавания. Повреждённый ролик дал failed
  за 0.424 секунды. Проверены 401 без ключа и с неверным ключом, 404 для
  неизвестной задачи и закрытых файлов, 415 для неподдерживаемого расширения.
  Датасет через публичный туннель не передавался.
- Оба PowerShell-скрипта запуска/остановки прошли синтаксический разбор.

Воспроизведение после активации интеграционного Python-окружения:

```powershell
python -m pytest tests/api tests/integration -q
node --test tests/web/app.test.cjs
node --check web/app.js
git diff --check
python scripts/smoke_public_api.py --base-url <ACTIVE_URL> --key-file secrets/public-demo-api-key.txt --output tmp/public-demo/smoke.json
```

Системный Python без этого окружения не содержит pytest. Полный прогон реальных
видео в этой доработке не повторялся; показатели ниже относятся к предыдущей
интеграции. Публичный доступ подтверждён; автоматическое сопоставление события
с правилом остаётся открытым до интеграции проверенных наблюдений Qwen.
Следующий шаг: передать жюри адрес и ключ отдельно, сохранить процессы до
показа; при смене ноутбука повторить проверку. Commit/push отложены до решения
пользователя о Git author identity; новый аккаунт для этого не требуется.
Последняя локальная проверка `rule_id` на непустую строку протестирована,
но ещё не загружена в работающий процесс: автоматическая проверка разрешений
заблокировала перезапуск API. Туннель и предыдущая проверенная версия продолжают
работать. Перед следующим запуском используется актуальный код из рабочей копии.

- Integration branch: `feature/teamlead-ml-inference-integration`.
- ML handoff: `origin/feature/ml-model@f891b213b8de5f923c136e4f228db08122553fe6`.
- ML code/artifact source: `aba56d31030aa9830fe2770e4aa9d54d2879ef24`.
- Inference source fix: `origin/feature/video-inference@405f06e`.
- Integrated decoder/Unicode fix: `875c09b`.
- Evidence/diagnostics integration: `da658c9`.
- API/runtime/UI integration: `806d10d`.
- Expected bundle: `baseline-audited-v4-development`.
- Expected archive: `baseline-audited-v4.zip`, 7,425,647 bytes.
- Expected archive SHA-256:
  `c387bdea9bd80a124c5587fccfcec3243d2550658ee09ca6834acaa9e54a1241`.
- Expected canonical `model.onnx` SHA-256:
  `00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d`.
- Release status: `development_only_not_promoted`; `quality_passed=false`.

Принятый архив имел точный размер 7,425,647 bytes и ожидаемый archive SHA-256.
После распаковки `model.onnx` имел ожидаемый SHA-256; случайные веса и
репозиторный historical `models/best_model.onnx` не использовались.

## Подключение

- `QAITU_INFERENCE_MODE` теперь обязателен: неявного default mock нет.
- ONNX mode требует `QAITU_MODEL_BUNDLE`; development bundle требует отдельный
  `QAITU_ALLOW_DEVELOPMENT_MODEL=true`.
- `QAITU_EXPECTED_MODEL_SHA256` позволяет зафиксировать именно согласованные
  веса поверх внутренней проверки полного `SHA256SUMS.txt`.
- FastAPI создаёт и проверяет runtime при startup. Ошибка bundle/model завершает
  startup и не переключает сервис на mock.
- `get_runtime()` кэширует predictor один раз на процесс.
- `QAITU_MAX_CONCURRENT_JOBS` ограничивает параллельный доступ к одной session;
  значение для demo — `1`.
- `/health` показывает mode, predictor class, model version, model SHA-256,
  release status и ONNX Runtime providers без локальных путей.
- Внутренний progress endpoint сохраняет полный window timeline и для событий
  `start_sec`, `end_sec`, `peak_time_sec`, `peak_score`, supporting count/indices.
  Публичная схема API не расширена этими данными.
- Технические decode/model/evidence ошибки дают `failed`, `verdict=null` и пустой
  список violations.

## Исправленные дефекты

- Unicode JPEG output: `cv2.imencode` и `Path.write_bytes`; JPEG повторно
  декодируется в регрессионном тесте.
- Unknown codec: отсутствие `stream.codec_context` проверяется до чтения
  metadata/decode и даёт `UnsupportedVideoError`.
- Evidence: до девяти кандидатов выбираются внутри границ события, сначала
  сохраняется ближайший к peak кадр, затем временно удалённые и визуально
  отличимые кадры. При одном уникальном кадре возвращается один.
- UI показывает десятые доли секунды для события и JPEG, поэтому разные
  timestamp больше не превращаются в подписи `09/10/10`.
- UI получает identity из `/health` до запуска и явно показывает `ONNX` либо
  `Demo model`.

## Выполненные проверки

```powershell
python -m compileall -q src scripts
python -m pytest tests/ml -q
python -m pytest tests/inference -q
python -m pytest tests/api tests/integration -q
python -m pytest -q
python -m pip check
node --check web/app.js
git diff --check
```

Фактический результат:

- ML tests: `32 passed`.
- Inference tests: `37 passed`.
- API + integration tests: `15 passed`, одно предупреждение о будущем изменении
  compatibility shim в `fastapi.testclient`.
- Полный доступный набор: `84 passed`, то же одно предупреждение.
- Dependency check: `No broken requirements found`.
- Python compile, JavaScript syntax и whitespace checks: passed.

## End-to-end matrix

| Сценарий | Ожидание | Фактический результат |
|---|---|---|
| Canonical synthetic ONNX, 9 s, Unicode input/output | `done`, ONNX identity, event, valid JPEG | passed через реальный HTTP API и UI; 1 event at 3.0 s, 1 уникальный JPEG |
| Видимый runtime до загрузки | ONNX/model/provider видимы | passed: `ONNX`, `integration-development`, `CPUExecutionProvider` |
| File picker | выбор по кнопке, не только drag/drop | passed в headed Chromium через native file chooser |
| Повторная загрузка | две независимые задачи | passed; обе `done` |
| Два одновременных запроса при limit=1 | без падения/смешения evidence | passed; обе задачи `done`, общий wall time 0.879 s на synthetic fixture |
| Unicode evidence directory | JPEG существует и декодируется | passed |
| Визуально одинаковые кадры | не создавать три дубликата | passed; возвращён один уникальный JPEG |
| Повреждённый MOV | `failed`, не `no_violation` | passed через API |
| Неизвестный codec | typed failure, не native crash | passed через API и отдельный subprocess regression test |
| Неверный expected model SHA | runtime startup/load rejected | passed |
| Повреждённый/отсутствующий model file | bundle/model load rejected | passed в bundle tests |
| CPU fallback | `CPUExecutionProvider` | passed |
| Browser console | no errors | passed: 0 errors, 0 warnings |
| Exact v4 identity | правильные bundle/version/SHA/provider | passed через `/health`: `OnnxPredictor`, `baseline-audited-v4-development`, exact SHA, `CPUExecutionProvider` |
| 42 локальных размеченных видео | все завершаются технически | passed: 42 `done`, 0 `failed`, все созданные JPEG декодируются |
| True positive через API/UI | событие и уникальные evidence | passed на `Х-11`: peak 19.0 s, JPEG 16.0/19.0/22.0 s |
| True negative | `no_violation` | passed на `Х-3` |
| Known false positive | ошибка качества остаётся видимой | confirmed на `Х-12`: peak 37.0 s |
| Known false negative | ошибка качества остаётся видимой | confirmed на `X-1`: размеченный момент 22 s, API вернул `no_violation` |
| Unicode filename | загрузка и обработка без ошибки | passed через API и headed browser на `Х-11.mov` |
| Повреждённый Unicode MOV | `failed`, не `no_violation` | passed через exact-v4 API |
| Integrated temporal shift `0/+4/-4` | `19/23/15 s` | passed через exact-v4 API; JPEG также сдвинулись `16/19/22 -> 20/23/26 -> 12/15/18` |

Synthetic fixture проверяет wiring, а не качество распознавания. Он не является
`baseline-audited-v4` и не используется в demo после проверки.

## Производительность

Машина текущей интеграции: Windows 11, Intel Core 5 120U, 12 logical CPUs,
ONNX Runtime 1.23.2. Доступен `CPUExecutionProvider`; `nvidia-smi` и CUDA provider
в этом окружении недоступны.

- Synthetic ONNX cold import + runtime creation: 1.1461 s.
- Повторный cached runtime lookup: 0.000002 s; тот же объект.
- Первый полный HTTP upload-to-result, 9 s fixture: 0.271 s; pipeline 0.178 s.
- Повторный запрос: 0.140 s; pipeline 0.081 s.
- Две ограниченно-параллельные задачи: обе `done`, общий wall 0.879 s.

Exact-v4 полный HTTP upload-to-result был измерен на 42 локальных видео:

- максимальная проверенная длительность видео: 142.533 s;
- максимальный wall time: 33.371 s;
- technical failures: 0;
- runtime provider: `CPUExecutionProvider`;
- ограничение 300 s выполнено для этого набора и этой машины, но не является
  гарантией для произвольного размера видео или другого оборудования.

## Temporal shift

Temporal shift повторён через текущий HTTP API на `Х-11`: исходный peak 19.0 s,
held-frame prefix +4 s дал 23.0 s, удаление первых 4 s дало 15.0 s. Оба результата
`done/violation`; evidence timestamps сдвинулись согласованно. Это подтверждает,
что интегрированный pipeline реагирует на положение содержимого, а не выдаёт
фиксированный пик около десятой секунды.

## Решение и блокеры

- Code compatibility: verified.
- Synthetic real-ONNX API/UI path: verified.
- Exact `baseline-audited-v4` API/UI path: verified.
- Ready to show as a technical hackathon demo: **yes**, на `127.0.0.1:8123`.
- Ready for production or autonomous legal conclusions: **no**.
- Quality promotion: **no**; baseline metrics остаются precision 0.500,
  recall 0.619, positive F1 0.553, macro F1 0.458, FP events/min 1.033,
  event recall ±5 s 0.364, peak MAE 7.133 s на reused development dataset.

`scripts/start_onnx_demo.ps1` проверил archive/model SHA-256, установил exact
bundle без молчаливой перезаписи и запустил API в явном ONNX-режиме. Перед
презентацией `/health` должен снова подтвердить exact SHA, `OnnxPredictor` и
активный provider; `8124` остаётся старым mock-прототипом и не используется.
