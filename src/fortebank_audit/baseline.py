from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import VideoRecord
from .video import probe_video


HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


@dataclass(frozen=True)
class ScoredWindow:
    timestamp_sec: float
    score: float


def _read_rgb(capture: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame at {timestamp_sec:.2f}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_pair_feature(previous_rgb: np.ndarray, current_rgb: np.ndarray) -> np.ndarray:
    previous = cv2.resize(previous_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    current = cv2.resize(current_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    hog = HOG.compute(gray).reshape(-1)

    histograms = []
    hsv = cv2.cvtColor(current, cv2.COLOR_RGB2HSV)
    for channel, bins, upper in ((0, 16, 180), (1, 16, 256), (2, 16, 256)):
        histogram = cv2.calcHist([hsv], [channel], None, [bins], [0, upper]).reshape(-1)
        histograms.append(histogram / max(histogram.sum(), 1.0))

    motion = cv2.absdiff(current, previous)
    motion_gray = cv2.cvtColor(motion, cv2.COLOR_RGB2GRAY)
    motion_grid = cv2.resize(motion_gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0
    motion_stats = np.array([motion_gray.mean(), motion_gray.std(), np.percentile(motion_gray, 90)]) / 255.0
    return np.concatenate([hog, *histograms, motion_grid, motion_stats]).astype(np.float32)


def sample_video(path: str | Path, step_sec: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    metadata = probe_video(path)
    duration = float(metadata["duration_sec"])
    timestamps = np.arange(step_sec, max(step_sec + 0.01, duration - 0.05), step_sec)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    features: list[np.ndarray] = []
    valid_timestamps: list[float] = []
    try:
        for timestamp in timestamps:
            try:
                previous = _read_rgb(capture, max(0.0, float(timestamp) - step_sec))
                current = _read_rgb(capture, float(timestamp))
            except ValueError:
                continue
            features.append(frame_pair_feature(previous, current))
            valid_timestamps.append(float(timestamp))
    finally:
        capture.release()

    if not features:
        raise ValueError(f"No frames sampled from {path}")
    return np.vstack(features), np.asarray(valid_timestamps, dtype=np.float32)


def build_training_set(
    records: list[VideoRecord],
    step_sec: float = 1.0,
    event_radius_sec: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []

    for record in records:
        video_features, timestamps = sample_video(record.path, step_sec)
        if record.has_violation:
            if record.timecode_sec is None:
                continue
            keep = np.abs(timestamps - record.timecode_sec) <= event_radius_sec
            video_features = video_features[keep]
            video_labels = np.ones(int(keep.sum()), dtype=np.int8)
        else:
            video_labels = np.zeros(len(timestamps), dtype=np.int8)

        if len(video_features) == 0:
            continue
        features.append(video_features)
        labels.append(video_labels)
        groups.append(np.full(len(video_labels), record.video_id, dtype=np.int32))

    return np.vstack(features), np.concatenate(labels), np.concatenate(groups)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1500, C=0.2)),
        ]
    )


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray) -> tuple[float, dict[str, float]]:
    group_ids = np.unique(groups)
    video_labels = np.array([labels[groups == group].max() for group in group_ids])
    video_scores = np.array([probabilities[groups == group].max() for group in group_ids])

    candidates = np.linspace(0.05, 0.95, 181)
    choices: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predicted = video_scores >= threshold
        precision, recall, f2, _ = precision_recall_fscore_support(
            video_labels, predicted, beta=2, average="binary", zero_division=0
        )
        choices.append((float(threshold), float(precision), float(recall), float(f2)))

    eligible = [choice for choice in choices if choice[2] >= 0.9]
    threshold, precision, recall, f2 = max(eligible or choices, key=lambda choice: (choice[1], choice[3]))
    return threshold, {"precision": precision, "recall": recall, "f2": f2}


def train(records: list[VideoRecord], output_path: str | Path) -> dict[str, object]:
    features, labels, groups = build_training_set(records)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    pipeline = make_pipeline()
    probabilities = cross_val_predict(
        pipeline,
        features,
        labels,
        groups=groups,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    threshold, metrics = choose_threshold(labels, probabilities, groups)
    pipeline.fit(features, labels)

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "step_sec": 1.0,
        "metrics": metrics,
        "feature_count": int(features.shape[1]),
        "sample_count": int(features.shape[0]),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target)
    return {key: value for key, value in artifact.items() if key != "pipeline"}


def score_video(path: str | Path, model_path: str | Path) -> tuple[list[ScoredWindow], float]:
    artifact = joblib.load(model_path)
    features, timestamps = sample_video(path, float(artifact["step_sec"]))
    raw_scores = artifact["pipeline"].predict_proba(features)[:, 1]
    padded = np.pad(raw_scores, (1, 1), mode="edge")
    smoothed = np.maximum.reduce([padded[:-2], padded[1:-1], padded[2:]])
    windows = [ScoredWindow(float(timestamp), float(score)) for timestamp, score in zip(timestamps, smoothed)]
    return windows, float(artifact["threshold"])
