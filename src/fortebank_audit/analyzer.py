from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2

from .baseline import ScoredWindow, score_video as score_video_legacy
from .onnx_runtime import score_video as score_video_onnx


@dataclass(frozen=True)
class DetectedEvent:
    timestamp_sec: float
    score: float


def _cluster_events(windows: list[ScoredWindow], threshold: float) -> list[DetectedEvent]:
    active = [window for window in windows if window.score >= threshold]
    if not active:
        return []

    clusters: list[list[ScoredWindow]] = [[active[0]]]
    for window in active[1:]:
        if window.timestamp_sec - clusters[-1][-1].timestamp_sec <= 2.1:
            clusters[-1].append(window)
        else:
            clusters.append([window])

    events = [
        DetectedEvent(timestamp_sec=peak.timestamp_sec, score=peak.score)
        for cluster in clusters
        for peak in [max(cluster, key=lambda item: item.score)]
    ]
    strongest = sorted(events, key=lambda event: event.score, reverse=True)[:3]
    return sorted(strongest, key=lambda event: event.timestamp_sec)


def _evidence_frames(video_path: Path, timestamp_sec: float) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps
    latest = max(0.0, duration - 1 / fps)
    moments = sorted({round(max(0.0, min(latest, timestamp_sec + offset)), 3) for offset in (-1.0, 0.0, 1.0)})
    evidence = []
    try:
        for moment in moments:
            capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read evidence frame at {moment:.2f}s")
            encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
            if not encoded:
                raise ValueError("Could not encode evidence frame")
            evidence.append(
                {
                    "timestamp_sec": round(moment, 2),
                    "image_base64": base64.b64encode(payload.tobytes()).decode("ascii"),
                }
            )
    finally:
        capture.release()
    return evidence


def analyze(video_path: str | Path, model_path: str | Path) -> dict[str, object]:
    path = Path(video_path)
    model = Path(model_path)
    scorer = score_video_onnx if model.suffix.lower() == ".onnx" else score_video_legacy
    windows, threshold = scorer(path, model)
    events = _cluster_events(windows, threshold)
    violations = [
        {
            "type": "Иное визуально наблюдаемое событие",
            "timecode_sec": round(event.timestamp_sec, 2),
            "description": (
                "Модельный score фрагмента превысил настроенный порог: визуальный паттерн похож "
                "на размеченные события. Тип действия и правило не сопоставлены; фрагмент требует "
                "проверки сотрудником."
            ),
            "confidence": round(event.score, 4),
            "evidence_frames": _evidence_frames(path, event.timestamp_sec),
        }
        for event in events
    ]
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
    }

