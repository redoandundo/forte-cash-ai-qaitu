from __future__ import annotations

from pathlib import Path

import cv2


def probe_video(path: str | Path) -> dict[str, float | int | str]:
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    duration_sec = frame_count / fps if fps > 0 else 0.0
    return {
        "path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def extract_frame(
    video_path: str | Path,
    timestamp_sec: float,
    output_path: str | Path | None = None,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise ValueError(f"Could not read {video_path} at {timestamp_sec:.2f}s")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), frame):
            raise OSError(f"Could not write frame to {target}")
    return frame_rgb

