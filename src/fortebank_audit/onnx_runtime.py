from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .baseline import ScoredWindow


def _centers(duration: float, window: float, stride: float) -> np.ndarray:
    values = np.arange(window / 2, max(window / 2 + 0.01, duration - window / 2 + 0.01), stride)
    return values if len(values) else np.array([duration / 2])


def _sample_windows(path: Path, centers: np.ndarray, seconds: float, count: int, image_size: int) -> np.ndarray:
    """Decode the video once and reuse overlapping frames across every window."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps
    plans = []
    for center in centers:
        start = max(0.0, min(float(center) - seconds / 2, max(0.0, duration - seconds)))
        moments = np.linspace(start, min(duration, start + seconds), count, endpoint=False)
        plans.append(np.clip(np.rint(moments * fps).astype(int), 0, max(0, total - 1)))
    required = {int(index) for plan in plans for index in plan}
    decoded: dict[int, np.ndarray] = {}
    try:
        for index in range(max(required) + 1):
            ok, bgr = capture.read()
            if not ok:
                break
            if index in required:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded[index] = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    finally:
        capture.release()
    missing = required.difference(decoded)
    if missing:
        raise ValueError(f"Could not decode {path}; missing {len(missing)} sampled frames")

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
    windows = []
    for plan in plans:
        frames = np.stack([decoded[int(index)] for index in plan]).astype(np.float32) / 255.0
        video = frames.transpose(3, 0, 1, 2)
        windows.append((video - mean) / std)
    return np.stack(windows).astype(np.float32, copy=False)


@lru_cache(maxsize=4)
def _load(model_path: str) -> tuple[ort.InferenceSession, dict, float, str]:
    model = Path(model_path)
    preprocessing = json.loads((model.parent / "preprocessing.json").read_text(encoding="utf-8"))
    threshold = float(json.loads((model.parent / "thresholds.json").read_text(encoding="utf-8"))["violation"])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    return session, preprocessing, threshold, session.get_providers()[0]


def score_video(path: str | Path, model_path: str | Path, stride: float = 2.0, batch_size: int = 32) -> tuple[list[ScoredWindow], float]:
    video_path = Path(path)
    model = Path(model_path).resolve()
    session, config, threshold, _ = _load(str(model))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    duration = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    capture.release()

    window = float(config["window_sec"])
    centers = _centers(duration, window, stride)
    batch = _sample_windows(video_path, centers, window * float(config.get("temporal_span_fraction", 1.0)), int(config["num_frames"]), int(config["image_size"]))
    scores = []
    for offset in range(0, len(batch), batch_size):
        logits = session.run([config["output_name"]], {config["input_name"]: batch[offset : offset + batch_size]})[0].reshape(-1)
        scores.extend((1 / (1 + np.exp(-logits))).tolist())
    values = np.asarray(scores, dtype=float)
    if len(values) >= 3:
        values = np.convolve(np.pad(values, (1, 1), mode="edge"), np.array([0.25, 0.5, 0.25]), mode="valid")
    return [ScoredWindow(float(center), float(score)) for center, score in zip(centers, values)], threshold


def provider(model_path: str | Path) -> str:
    return _load(str(Path(model_path).resolve()))[3]
