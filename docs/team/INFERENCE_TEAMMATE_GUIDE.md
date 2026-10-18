# Руководство для участника: video inference и таймкоды

Версия: 1.0

Дата: 17 октября 2026 года

Владелец архитектуры и интеграции: тимлид проекта

## 1. Твоя роль

Ты отвечаешь за путь от готового видео до проверяемого результата модели:

```text
.mov
  -> проверка файла
  -> декодирование кадров
  -> скользящие временные окна
  -> preprocessing
  -> Predictor
  -> scores по времени
  -> объединение срабатываний
  -> итоговый таймкод
  -> JPEG-кадры-доказательства
```

Первый участник команды обучает и экспортирует модель. Тимлид отвечает за архитектуру, API, фоновые задачи и финальную интеграцию. Твоя задача — построить между ними надежный и быстрый inference-модуль.

## 2. Что нельзя менять

Без согласования с тимлидом не изменяй:

- публичные API endpoints;
- job-систему;
- формат хранения заданий;
- юридическую классификацию;
- исходные видео;
- исходную таблицу разметки;
- ML-архитектуру и веса первого участника.

Запрещено отправлять видео, кадры или модель во внешние AI API, Colab, Kaggle и публичные хранилища. Всё выполняется локально.

Не добавляй в Git исходные `.mov`, созданные clips, JPEG-кадры, веса, кэши моделей и `.venv`.

## 3. Что прочитать перед началом

Попроси Codex изучить:

1. `docs/team/INFERENCE_TEAMMATE_GUIDE.md`.
2. `analysis/dataset_manifest.json`.
3. `analysis/preliminary_video_review.md`.
4. `analysis/annotation_schema.json`.
5. `legal/README.md` — только для понимания границ описания события.
6. `docs/team/ML_TEAMMATE_GUIDE.md` — разделы об экспорте и интеграционном контракте.
7. `AGENTS.md`, если он существует.

Данные:

- видео: `data/open/videos/`;
- исходная разметка: `data/open/open_labels.xlsx`;
- manifest: `analysis/dataset_manifest.json`;
- готовые диагностические clips: `analysis/clips/`.

Характеристики набора:

- 42 видео;
- H.264 в `.mov`;
- `464x848`;
- около 30 FPS;
- без аудио;
- длительность от 11,6 до 142,5 секунды;
- присутствуют латинская `X` и кириллическая `Х` в именах файлов.

## 4. Результат двухчасового спринта

Через два часа должны существовать:

```text
src/inference/
  __init__.py
  contracts.py
  video_reader.py
  windowing.py
  preprocessing.py
  predictor.py
  aggregation.py
  evidence.py
  pipeline.py
  benchmark.py

tests/inference/
  test_video_reader.py
  test_windowing.py
  test_aggregation.py
  test_pipeline_smoke.py

reports/
  inference_benchmark.md
```

Обязательная команда проверки:

```powershell
python -m pytest tests/inference -q
python -m src.inference.benchmark --video data/open/videos/X-1.mov --predictor mock
```

Минимальный программный интерфейс:

```python
result = analyze_video(
    video_path="data/open/videos/X-1.mov",
    predictor=predictor,
    config=config,
)
```

Ожидаемый результат модуля:

```python
{
    "verdict": "violation",
    "score": 0.91,
    "timecode_sec": 22.1,
    "events": [
        {
            "score": 0.91,
            "start_sec": 19.0,
            "end_sec": 24.0,
            "peak_time_sec": 22.1,
            "evidence": [
                {
                    "timecode_sec": 21.6,
                    "jpeg_path": "runtime/evidence/event_000_01.jpg"
                }
            ]
        }
    ],
    "runtime": {
        "duration_sec": 25.933,
        "processing_sec": 3.42,
        "device": "CPU",
        "model_version": "mock-v1"
    }
}
```

API-кодирование JPEG в base64 выполняет тимлид. Твой модуль передает локальный путь, bytes или согласованную структуру evidence.

## 5. Skills для Codex

Для этого участника нужен минимальный набор:

1. `systematic-debugging` — обязателен при проблемах с FFmpeg, timestamps, shapes, CUDA и ONNX providers.
2. `security-best-practices` — использовать только в конце для проверки безопасной обработки входных путей и временных файлов.

`jupyter-notebook`, `playwright` и frontend-skills для двухчасовой задачи не нужны.

### Команда для Codex

```text
Используй системный skill-installer. Установи systematic-debugging из репозитория obra/superpowers по пути skills/systematic-debugging и security-best-practices из openai/skills по пути skills/.curated/security-best-practices. Прочитай SKILL.md до применения и сообщи установленные пути. Новые skills используй со следующего хода.
```

Ручной PowerShell fallback:

```powershell
$installer = Join-Path $env:USERPROFILE ".codex\skills\.system\skill-installer\scripts\install-skill-from-github.py"

py $installer --repo obra/superpowers --path skills/systematic-debugging
py $installer --repo openai/skills --path skills/.curated/security-best-practices
```

Источники:

- https://github.com/obra/superpowers/tree/main/skills/systematic-debugging
- https://github.com/openai/skills/tree/main/skills/.curated/security-best-practices

## 6. Локальные программы и библиотеки

### 6.1. Git и отдельная ветка

```powershell
git status --short
git switch -c inference/video-pipeline
```

Если рабочая папка содержит чужие untracked-файлы, не трогай их и не используй `git add .`.

### 6.2. FFmpeg

Windows:

```powershell
winget install --id Gyan.FFmpeg -e
ffmpeg -version
ffprobe -version
```

Официальная страница:

https://ffmpeg.org/download.html

### 6.3. Python

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 6.4. Пакеты

CPU-вариант:

```powershell
python -m pip install `
  numpy opencv-python av pillow pyyaml tqdm psutil `
  onnx onnxruntime pytest
```

NVIDIA-вариант:

```powershell
python -m pip install `
  numpy opencv-python av pillow pyyaml tqdm psutil `
  onnx onnxruntime-gpu pytest
```

Не устанавливай одновременно `onnxruntime` и `onnxruntime-gpu` в одно окружение.

Проверка ONNX providers:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Документация:

- https://onnxruntime.ai/docs/install/
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

### 6.5. Нужны ли локальные модели

В первые два часа отдельные модели скачивать не нужно.

Используй:

- `MockPredictor` для проверки pipeline;
- готовый `.onnx` первого участника, когда он появится;
- маленький искусственный ONNX fixture только для теста адаптера, если его можно создать локально.

Не скачивай RT-DETR, ByteTrack, SmolVLM2 или X3D. Это не твоя задача на первый спринт. Если первый участник передаст R3D/X3D, ты подключаешь экспортированный artifact, а не создаешь вторую конкурирующую модель.

## 7. Главный промпт для Codex

Открой репозиторий в Codex и отправь следующий текст целиком:

```text
Ты отвечаешь за video inference pipeline в командном проекте локального анализа кассовых видео.

Тимлид владеет архитектурой, API, фоновой job-системой и интеграцией. Другой участник обучает и экспортирует ML-модель. Не меняй их зоны без явного согласования.

Сначала прочитай:
- docs/team/INFERENCE_TEAMMATE_GUIDE.md;
- analysis/dataset_manifest.json;
- analysis/preliminary_video_review.md;
- analysis/annotation_schema.json;
- docs/team/ML_TEAMMATE_GUIDE.md в части model contract и export;
- AGENTS.md, если он существует.

Твоя цель на ближайшие два часа: реализовать и проверить локальный pipeline
.mov -> metadata -> streaming windows -> Predictor -> temporal aggregation -> timecode -> JPEG evidence -> benchmark.

Ограничения:
- не отправляй видео, кадры или модель во внешние сервисы;
- не изменяй data/open, исходную таблицу и юридические материалы;
- не обучай модель и не меняй model architecture;
- не реализуй публичный API и job queue;
- не коммить видео, веса, runtime evidence, caches и .venv;
- не используй git add .;
- учитывай Unicode-имена с латинской X и кириллической Х;
- используй реальные временные метки кадров/окон, не полагайся слепо на frame_index / 30;
- не загружай целое длинное видео в RAM;
- при ошибках сначала применяй systematic-debugging и устанавливай root cause;
- продолжай до работающих тестов и benchmark, а не останавливайся после плана.

Сначала выполни read-only аудит структуры репозитория и проверь ffmpeg/ffprobe, Python и ONNX Runtime providers. Затем создай минимальный план файлов и сразу реализуй его.

Definition of Done:
1. VideoReader возвращает metadata и поток кадров с timestamps.
2. WindowGenerator корректно обрабатывает short/exact/tail cases.
3. Predictor является интерфейсом; существуют MockPredictor и OnnxPredictor.
4. Aggregator сглаживает scores, создает события и выбирает peak_time_sec.
5. EvidenceExtractor сохраняет 1-3 JPEG рядом с peak.
6. analyze_video соединяет компоненты без API-зависимостей.
7. pytest проходит.
8. benchmark запускается на X-1.mov и одном длинном видео.
9. reports/inference_benchmark.md содержит команды, времена, RAM/VRAM, проблемы и следующий шаг.

Не делай масштабный framework. Предпочитай небольшие типизированные Python-модули, dataclasses, явные контракты, dependency injection predictor и тестируемые pure functions для windowing/aggregation.
```

## 8. Контракты данных

Создай в `contracts.py` компактные dataclasses или эквивалентные typed structures.

### VideoMetadata

```python
@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    duration_sec: float
    width: int
    height: int
    nominal_fps: float
    frame_count: int | None
    codec: str
    has_audio: bool
```

### VideoFrame

```python
@dataclass(frozen=True)
class VideoFrame:
    timestamp_sec: float
    image_rgb: np.ndarray
```

### ClipWindow

```python
@dataclass(frozen=True)
class ClipWindow:
    index: int
    start_sec: float
    end_sec: float
    center_sec: float
    frames_rgb: np.ndarray
```

### WindowPrediction

```python
@dataclass(frozen=True)
class WindowPrediction:
    window_index: int
    start_sec: float
    end_sec: float
    center_sec: float
    score: float
```

### DetectedEvent

```python
@dataclass(frozen=True)
class DetectedEvent:
    start_sec: float
    end_sec: float
    peak_time_sec: float
    peak_score: float
```

Публичные контракты не должны зависеть от PyTorch. Это позволит заменить ONNX/PyTorch predictor без переписывания pipeline.

## 9. План работы на ближайшие два часа

### 00:00–00:10 — контекст и preflight

Выполнить:

- прочитать перечисленные документы;
- проверить `git status`;
- проверить `ffmpeg`, `ffprobe`, Python;
- проверить ONNX Runtime providers;
- подтвердить пути минимум к `X-1.mov` и одному длинному видео;
- записать стартовые условия в `reports/inference_benchmark.md`.

Результат: известное окружение и отсутствие предположений о путях.

### 00:10–00:30 — VideoReader

Реализовать:

- получение metadata через `ffprobe` JSON или PyAV;
- проверку расширения и существования файла;
- понятные исключения;
- итератор кадров с `timestamp_sec`;
- конвертацию BGR/YUV в RGB;
- streaming без чтения всего ролика в список.

Важно: используй реальные timestamps из контейнера/PyAV PTS. `frame_index / FPS` допустим только как документированный fallback.

Тесты:

- открывается `X-1.mov`;
- duration согласуется с manifest с разумным tolerance;
- первый timestamp неотрицателен;
- timestamps монотонны;
- несуществующий файл дает понятную ошибку;
- Unicode-путь работает.

### 00:30–00:50 — WindowGenerator

Реализовать чистую, тестируемую логику:

- `window_sec` и `stride_sec` задаются config;
- одинаковое число модельных кадров в каждом окне;
- uniform temporal sampling;
- последнее неполное окно обрабатывается согласно config;
- короткое видео не падает;
- center time рассчитывается явно.

Тесты без реального видео:

- ролик короче окна;
- длительность ровно равна окну;
- несколько перекрывающихся окон;
- хвост ролика;
- пустой поток;
- немонотонные timestamps отклоняются.

### 00:50–01:10 — Predictor

Определи Protocol или abstract interface:

```python
class Predictor(Protocol):
    @property
    def model_version(self) -> str: ...

    def predict_batch(self, windows: Sequence[ClipWindow]) -> list[float]: ...
```

Реализуй:

- `MockPredictor` с детерминированными scores;
- `OnnxPredictor`, который:
  - выбирает CUDA provider при наличии;
  - имеет CPU fallback;
  - читает input name и проверяет shape;
  - не знает ничего про API;
  - получает preprocessing/config извне.

Если реального `.onnx` пока нет, OnnxPredictor должен быть покрыт unit test с monkeypatch/fake session, а pipeline полностью работать с MockPredictor.

### 01:10–01:30 — TemporalAggregator

Сделай pure functions:

- smoothing: moving average или median;
- thresholding;
- merge соседних положительных окон;
- минимальная длительность события;
- `peak_time_sec` по максимальному score;
- verdict по наличию хотя бы одного валидного события.

Начальный config:

```yaml
aggregation:
  smoothing_window: 3
  threshold: 0.5
  merge_gap_sec: 1.5
  min_event_sec: 0.5
  max_events: 5
```

Это только defaults для pipeline-тестов. Финальный threshold передаст первый участник в `thresholds.json`.

Тесты:

- все scores ниже threshold;
- один пик;
- соседние пики объединяются;
- два удаленных события остаются разными;
- короткий одиночный шум удаляется;
- peak time соответствует максимуму;
- scores вне `[0, 1]` отклоняются.

### 01:30–01:45 — EvidenceExtractor

Реализовать:

- выбор 1–3 временных точек около `peak_time_sec`;
- границы `0 ... duration`;
- извлечение JPEG через FFmpeg или уже декодированный frame cache;
- уникальные кадры;
- контроль качества JPEG;
- безопасные имена файлов;
- отдельную runtime-директорию;
- возврат timecode и пути.

Не добавляй evidence в Git. Не кодируй base64 — это зона API.

Тесты:

- событие у начала ролика;
- событие у конца;
- три timecodes не выходят за duration;
- JPEG действительно читается OpenCV/Pillow;
- повторный запуск не смешивает результаты разных jobs.

### 01:45–02:00 — Pipeline, benchmark и handoff

Соедини:

```text
VideoReader
  -> WindowGenerator
  -> Predictor.predict_batch
  -> TemporalAggregator
  -> EvidenceExtractor
  -> result dict/dataclass
```

Запусти:

1. `X-1.mov` — короткий положительный пример.
2. `Х-14.mov` — длинный отрицательный пример.
3. Один Unicode-путь с кириллической `Х`.

С MockPredictor benchmark измеряет overhead декодирования и windowing. Отдельно укажи, что model latency еще отсутствует.

В `reports/inference_benchmark.md` запиши:

- hardware и OS;
- команды запуска;
- версии FFmpeg, Python и ONNX Runtime;
- provider;
- длительность видео;
- время декодирования;
- количество окон;
- общее время;
- realtime factor `processing_sec / video_duration_sec`;
- peak RAM;
- известные ограничения;
- что требуется от первого участника.

## 10. Preprocessing contract

Первый участник передаст `preprocessing.json`. Твой pipeline не должен угадывать normalization.

Минимально читать:

```json
{
  "input_layout": "NCTHW",
  "dtype": "float32",
  "color": "RGB",
  "num_frames": 16,
  "resize": [128, 171],
  "crop": [112, 112],
  "window_sec": 6.0,
  "stride_sec": 1.0
}
```

Валидируй:

- layout поддерживается;
- dtype известен;
- `num_frames > 0`;
- размеры положительные;
- stride не больше window без явного разрешения;
- normalization присутствует или имеет ссылку на согласованный preset;
- цветовой порядок не перепутан.

Если config отсутствует, MockPredictor может использовать тестовый config. Реальный OnnxPredictor должен отказать с понятной ошибкой, а не молча выбрать случайные значения.

## 11. Ошибки и статусы

Используй собственные исключения, например:

```python
class VideoOpenError(RuntimeError): ...
class UnsupportedVideoError(ValueError): ...
class PreprocessingConfigError(ValueError): ...
class PredictorLoadError(RuntimeError): ...
class PredictionError(RuntimeError): ...
class EvidenceExtractionError(RuntimeError): ...
```

Не проглатывай exception и не возвращай `score=0` при технической ошибке. Ошибка анализа и `no_violation` — разные состояния.

Не включай абсолютные пути пользователя и большие stack traces в публичный результат. Для локального debug-лога они допустимы.

## 12. Производительность

Целевой лимит всей системы — до пяти минут на видео. Твой pipeline должен оставлять запас для модели.

Считать:

```text
realtime_factor = processing_sec / video_duration_sec
```

Интерпретация:

- `< 1.0` — быстрее реального времени;
- `1.0–2.0` — обычно приемлемо для текущего набора;
- `> 2.0` — требуется профилирование;
- любое видео должно укладываться в общий лимит 300 секунд.

Порядок оптимизации после корректности:

1. batch inference;
2. не декодировать один кадр повторно для перекрывающихся окон;
3. уменьшить лишние RGB copies;
4. prefetch с ограниченной очередью;
5. ONNX CUDA provider;
6. фиксированные input shapes;
7. FP16 только после parity test;
8. TensorRT только по согласованию.

Не оптимизируй до прохождения correctness tests.

## 13. Улучшения после двух часов

В backlog, но не в обязательный двухчасовой scope:

- реальный model artifact от первого участника;
- batching нескольких окон;
- async/prefetch decoder;
- повторное использование перекрывающихся кадров;
- FP16 benchmark;
- ONNX graph optimization;
- несколько событий в одном видео;
- perceptual deduplication evidence;
- blur/quality score кадров;
- отдельный adapter для TorchScript;
- profile через `py-spy`, `cProfile` или profiler;
- RT-DETR/ByteTrack только после отдельного архитектурного решения.

## 14. Тестовая матрица

| Случай | Ожидаемый результат |
|---|---|
| Короткое валидное `.mov` | Успешный анализ |
| Длинное `.mov` | Streaming, без загрузки целиком в RAM |
| Кириллическое имя | Успешное открытие |
| Несуществующий файл | `VideoOpenError` |
| Поврежденный контейнер | Техническая ошибка, не `no_violation` |
| Видео короче окна | Одно корректное окно или документированный padding |
| Все scores низкие | `no_violation`, evidence отсутствует |
| Один устойчивый пик | Одно событие и правильный peak |
| Два соседних пика | Одно объединенное событие |
| Два удаленных пика | Два события |
| Событие у 0 секунд | Evidence не уходит в отрицательное время |
| Событие у конца | Evidence не превышает duration |
| Нет CUDA provider | Автоматический CPU fallback |
| Неверный input shape | Понятная ошибка до долгого запуска |

## 15. Оценка работы Codex: 100 баллов

| Раздел | Баллы | Критерий |
|---|---:|---|
| Preflight и отчет окружения | 5 | Зафиксированы версии и providers |
| VideoReader | 15 | Metadata, timestamps, streaming, ошибки |
| WindowGenerator | 15 | Границы, sampling, short/tail cases |
| Predictor contract | 15 | Mock + ONNX adapter + CPU fallback |
| Aggregation | 15 | Smoothing, merge, peak, тесты |
| Evidence | 10 | 1–3 валидных JPEG и timecodes |
| End-to-end pipeline | 10 | Запуск на реальном `.mov` |
| Тесты | 5 | `pytest` проходит |
| Benchmark | 5 | Время, RTF, RAM/VRAM |
| Handoff | 5 | Команды, ограничения, список ожиданий от ML |

Жесткие провалы:

- отправка данных наружу;
- изменение исходных видео;
- загрузка полного длинного видео в RAM без причины;
- таймкоды вычисляются неверно;
- техническая ошибка маскируется как `no_violation`;
- публичный API изменен без тимлида;
- создана собственная конкурирующая модель;
- результат не имеет тестов или команды воспроизведения.

Цель спринта — минимум 75/100 без жестких провалов.

## 16. Что передать тимлиду через два часа

Одно сообщение следующего формата:

```text
Ветка: inference/video-pipeline
Последний commit: <sha или причина отсутствия commit>

Готово:
- VideoReader: ...
- WindowGenerator: ...
- MockPredictor: ...
- OnnxPredictor: ...
- Aggregation: ...
- Evidence: ...
- Tests: X passed

Benchmark:
- video: ...
- duration: ... sec
- processing: ... sec
- realtime factor: ...
- provider: ...
- peak RAM/VRAM: ...

Запуск:
<одна точная команда>

Ожидаю от ML-участника:
- model file;
- preprocessing.json;
- thresholds.json;
- labels.json;
- model_version;
- SHA-256;
- ожидаемый input/output shape.

Ограничения/риски:
1. ...
2. ...

Следующий рекомендуемый шаг:
...
```

## 17. Финальный чек-лист

- [ ] Работа ведется в отдельной ветке.
- [ ] Чужие untracked-файлы не изменены.
- [ ] FFmpeg и ONNX providers проверены.
- [ ] Metadata совпадает с manifest в tolerance.
- [ ] Используются реальные timestamps.
- [ ] Видео обрабатывается streaming.
- [ ] Windowing покрыт unit tests.
- [ ] MockPredictor детерминирован.
- [ ] OnnxPredictor имеет CPU fallback.
- [ ] Неверный config дает понятную ошибку.
- [ ] Aggregation покрыт edge cases.
- [ ] Evidence не выходит за границы видео.
- [ ] Pipeline запускается на реальном `.mov`.
- [ ] Benchmark содержит realtime factor и память.
- [ ] Runtime artifacts не попали в Git.
- [ ] Тимлиду передано одно воспроизводимое сообщение.

## 18. Проверенные ресурсы

- FFmpeg downloads: https://ffmpeg.org/download.html
- PyAV documentation: https://pyav.org/docs/stable/
- OpenCV documentation: https://docs.opencv.org/
- ONNX Runtime installation: https://onnxruntime.ai/docs/install/
- ONNX Runtime CUDA provider: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- Official OpenAI skills: https://github.com/openai/skills
- Systematic debugging skill: https://github.com/obra/superpowers/tree/main/skills/systematic-debugging
- Codex prompting guide: https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide
