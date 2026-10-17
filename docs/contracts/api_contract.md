# Public API contract v1

## Purpose

The API screens a `.mov` recording and returns visual fragments that deserve human review. A result is decision support. It does not establish employee misconduct or a legal violation.

## Create a job

`POST /api/v1/analyze` accepts multipart field `file` and returns HTTP 202:

```json
{"job_id": "4f7e6a84d12c"}
```

Only `.mov` files up to 100 MB are accepted. The request requires `X-API-Key`.

## Read a job

`GET /api/v1/jobs/{job_id}` returns one of three states:

- `processing`: analysis is still running;
- `done`: inference and evidence extraction completed;
- `failed`: decoding, model inference, or evidence extraction failed.

A technical failure must never be returned as `no_violation`.

Successful response:

```json
{
  "job_id": "4f7e6a84d12c",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Иное визуально наблюдаемое событие",
      "timecode_sec": 17.0,
      "description": "Модельный score фрагмента превысил настроенный порог: визуальный паттерн похож на размеченные события. Тип действия и правило не сопоставлены; фрагмент требует проверки сотрудником.",
      "confidence": 0.3871,
      "evidence_frames": [
        {"timestamp_sec": 16.0, "image_base64": "..."},
        {"timestamp_sec": 17.0, "image_base64": "..."},
        {"timestamp_sec": 18.0, "image_base64": "..."}
      ]
    }
  ],
  "error": null
}
```

## Field semantics

| Field | Meaning | User action |
|---|---|---|
| `verdict=violation` | At least one model score crossed the configured threshold | Review every returned fragment |
| `verdict=no_violation` | No evaluated fragment crossed the threshold | Treat as “signs were not found in this recording”, not proof of correct conduct |
| `type` | Cautious observable category | Do not replace it with a specific action until that class is validated |
| `timecode_sec` | Peak of the grouped candidate event | Open the recording at this time |
| `description` | Why the candidate was returned and what remains unknown | Use it to guide review |
| `confidence` | Model ranking score in `[0, 1]` | Compare and prioritize candidates; do not read it as a probability |
| `evidence_frames` | Up to three frames before, at, and after the peak | Check local visual context, then inspect the video |

If no verified rule mapping exists, the service says that the rule is not mapped. It never invents a legal source from a binary model score.
