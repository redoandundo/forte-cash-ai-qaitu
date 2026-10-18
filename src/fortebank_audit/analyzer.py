from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2

from .baseline import ScoredWindow, score_video


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
    return sorted(events, key=lambda event: event.score, reverse=True)[:3]


def _frame_base64(video_path: Path, timestamp_sec: float) -> str:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read evidence frame at {timestamp_sec:.2f}s")
    encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded:
        raise ValueError("Could not encode evidence frame")
    return base64.b64encode(payload.tobytes()).decode("ascii")


def analyze(video_path: str | Path, model_path: str | Path) -> dict[str, object]:
    path = Path(video_path)
    windows, threshold = score_video(path, model_path)
    events = _cluster_events(windows, threshold)
    violations = [
        {
            "type": "Визуально нетипичное действие",
            "timecode_sec": round(event.timestamp_sec, 2),
            "description": (
                "В этом фрагменте локальный baseline обнаружил визуальное изменение, "
                "похожее на размеченные события. Результат требует проверки сотрудником."
            ),
            "confidence": round(event.score, 4),
            "evidence_frames": [
                {
                    "timestamp_sec": round(event.timestamp_sec, 2),
                    "image_base64": _frame_base64(path, event.timestamp_sec),
                }
            ],
        }
        for event in events
    ]
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
    }
