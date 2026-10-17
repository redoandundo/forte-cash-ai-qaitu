# Архитектура системы Qaitu

## Поток данных

```text
API
  -> X-API-Key validation
  -> Job service
  -> Video validation
  -> VideoReader
  -> WindowGenerator
  -> Predictor
  -> TemporalAggregator
  -> EvidenceExtractor
  -> Result storage
  -> exact jury API response
  -> Web UI result
```

## Компоненты

### API и jobs

Принимают `.mov` в multipart-поле `file`, проверяют `X-API-Key`, создают job, контролируют статус и возвращают точную JSON-схему жюри. Не содержат внутреннюю ML-логику.

### VideoReader

Проверяет контейнер, получает metadata и декодирует кадры streaming с реальными timestamps.

### WindowGenerator

Создаёт перекрывающиеся временные окна и выполняет uniform temporal sampling согласно preprocessing contract.

### Predictor

Имеет общий интерфейс для `MockPredictor`, `OnnxPredictor` и возможного TorchScript adapter. Публичные pipeline-контракты не зависят от PyTorch.

### TemporalAggregator

Сглаживает scores, применяет threshold, объединяет соседние окна и выбирает peak time.

### EvidenceExtractor

Сохраняет 1–3 JPEG около peak в локальную runtime-директорию.

## Границы

- API зависит от интерфейса pipeline, а не от конкретной модели.
- Внутренние events преобразуются в публичный список `violations` только в API adapter.
- Pipeline зависит от predictor protocol, manifest и preprocessing.
- Predictor зависит от локального runtime artifact.
- ML training не импортируется production API.
- Runtime artifacts не являются исходным кодом.

## Ошибки

Ошибки открытия видео, preprocessing, загрузки модели, inference и evidence являются техническими статусами. Они не могут возвращаться как `no_violation`.

## Публичный интерфейс

- `POST /api/v1/analyze` возвращает `job_id`.
- `GET /api/v1/jobs/{job_id}` возвращает `processing`, `done` или `failed`.
- При `done` verdict равен `violation` или `no_violation`.
- Evidence содержит 1–3 JPEG, закодированных в base64.
- Frontend использует тот же backend contract.

## Производительность

Цель всей системы — не более 300 секунд на видео. Pipeline измеряет `processing_sec / video_duration_sec`, RAM, VRAM и runtime provider.
