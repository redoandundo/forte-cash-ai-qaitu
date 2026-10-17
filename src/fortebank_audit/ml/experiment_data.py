from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from .preprocessing import VideoSpec, normalize_rgb
from .video import cached_video


def fingerprint(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def load_sources(manifest: Path, data_root: Path | None = None, review: Path | None = None) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("conflicts"):
        raise ValueError("Unresolved dataset conflicts")
    sources = [dict(row) for row in payload["sources"]]
    reviews = json.loads(review.read_text(encoding="utf-8")) if review else {}
    if set(reviews) - {row["source_id"] for row in sources}:
        raise ValueError("Review contains unknown sources")
    for row in sources:
        if row["source_id"] in reviews:
            update = reviews[row["source_id"]]
            if update.get("review_status") != "confirmed":
                raise ValueError("Only explicitly confirmed review corrections can change labels")
            allowed = {"label", "events_sec", "events", "annotation_confidence", "events_complete", "review_status", "reviewer"}
            if set(update) - allowed:
                raise ValueError("Unsupported review fields")
            if not update.get("reviewer"):
                raise ValueError("Review needs an identified reviewer")
            row.update(update)
        path = Path(row["path"])
        if data_root:
            path = data_root / path.name
        elif not path.is_absolute():
            path = manifest.parent / path
        row["path"] = str(path.resolve())
    validate_sources(sources)
    return sources


def validate_sources(sources: list[dict]) -> None:
    seen_ids, seen_hashes = set(), set()
    for row in sources:
        if row["source_id"] in seen_ids or row["sha256"] in seen_hashes:
            raise ValueError("Duplicate source or SHA must be collapsed before splitting")
        seen_ids.add(row["source_id"])
        seen_hashes.add(row["sha256"])
        if row["label"] not in (0, 1) or not row.get("scenario_group"):
            raise ValueError("Missing label or scenario group")
        if row.get("annotation_confidence", "high") not in {"high", "medium", "low"}:
            raise ValueError("Unknown annotation confidence")
        duration = float(row["duration_sec"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid source duration")
        events = row.get("events_sec", [])
        if (row["label"] == 0 and (events or row.get("events"))) or (row["label"] == 1 and not events and not row.get("events")):
            raise ValueError("Label and event annotations disagree")
        if any(not math.isfinite(t) or not 0 <= t <= duration for t in events):
            raise ValueError("Invalid event timestamp")
        for event in row.get("events", []):
            if event.get("review_status") != "confirmed" or not 0 <= event["start_sec"] < event["end_sec"] <= duration:
                raise ValueError("Intervals require confirmed, in-bounds annotations")


def source_identity(row: dict) -> dict:
    return {key: row.get(key) for key in ("source_id", "sha256", "scenario_group", "label", "events_sec", "events", "annotation_confidence", "events_complete")}


def make_folds(sources: list[dict], seed: int = 42, count: int = 5) -> dict[str, int]:
    validate_sources(sources)
    ordered = sorted(sources, key=lambda row: row["source_id"])
    labels = np.array([row["label"] for row in ordered])
    groups = np.array([row["scenario_group"] for row in ordered])
    # This uses labels only to balance partitions, never model predictions.
    for attempt in range(100):
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed + attempt)
        folds = list(splitter.split(np.zeros(len(ordered)), labels, groups))
        if all(set(labels[indices]) == {0, 1} for _, indices in folds):
            return {ordered[index]["source_id"]: fold for fold, (_, indices) in enumerate(folds) for index in indices}
    raise ValueError("Could not create folds with both classes")


def assert_disjoint(*partitions: list[dict]) -> None:
    for field in ("source_id", "sha256", "scenario_group"):
        seen = set()
        for rows in partitions:
            current = {row[field] for row in rows}
            if seen & current:
                raise ValueError(f"Leakage across partitions: {field}")
            seen |= current


def split_sources(sources: list[dict], folds: dict[str, int], outer: int, calibration: int):
    if outer == calibration or set(folds) != {row["source_id"] for row in sources}:
        raise ValueError("Invalid fold assignment")
    train = [row for row in sources if folds[row["source_id"]] not in {outer, calibration}]
    cal = [row for row in sources if folds[row["source_id"]] == calibration]
    test = [row for row in sources if folds[row["source_id"]] == outer]
    assert_disjoint(train, cal, test)
    if any({row["label"] for row in part} != {0, 1} for part in (train, cal, test)):
        raise ValueError("Every partition must contain both classes")
    return train, cal, test


def selected_training_windows(source: dict, samples) -> list[tuple[int, int, float]]:
    confidence = {"high": 1.0, "medium": 0.5, "low": 0.25}[source.get("annotation_confidence", "high")]
    if source["label"] == 0:
        indices = np.unique(np.rint(np.linspace(0, len(samples.frames) - 1, min(5, len(samples.frames)))).astype(int))
        return [(int(index), 0, 1.0 / len(indices)) for index in indices]
    intervals = source.get("events", [])
    if not intervals:
        indices = sorted({int(np.argmin(np.abs(samples.centers - time))) for time in source["events_sec"]})
        return [(index, 1, confidence / len(indices)) for index in indices]
    selected = []
    for index, (start, end) in enumerate(zip(samples.starts, samples.ends)):
        overlap = max(max(0, min(end, event["end_sec"]) - max(start, event["start_sec"])) / min(end - start, event["end_sec"] - event["start_sec"]) for event in intervals)
        if overlap >= 0.5:
            selected.append((index, 1, confidence))
        elif overlap > 0:
            selected.append((index, 1, confidence * 0.25))
        elif source.get("events_complete") and all(end < event["start_sec"] - 2 or start > event["end_sec"] + 2 for event in intervals):
            selected.append((index, 0, confidence * 0.5))
    if not selected:
        raise ValueError("No usable windows for confirmed events")
    return [(i, label, weight / len(selected)) for i, label, weight in selected]


def augment_frames(frames: np.ndarray) -> np.ndarray:
    size = frames.shape[1]
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), random.uniform(-7, 7), random.uniform(0.94, 1.06))
    matrix[:, 2] += [random.uniform(-0.04, 0.04) * size for _ in range(2)]
    brightness, gamma = random.uniform(0.8, 1.2), random.uniform(0.85, 1.15)
    quality = random.randint(45, 95) if random.random() < 0.5 else None
    output = []
    for frame in frames:
        transformed = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        transformed = np.clip((transformed.astype(np.float32) / 255) ** gamma * brightness * 255, 0, 255).astype(np.uint8)
        if quality:
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError("JPEG augmentation failed")
            transformed = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        output.append(transformed)
    return np.stack(output)


class SourceWindowDataset(Dataset):
    def __init__(self, sources: list[dict], spec: VideoSpec, cache: Path, augment: bool = True):
        self.spec, self.augment = spec, augment
        self.items = []
        for source in sources:
            path = Path(source["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Video checksum disagrees with dataset")
            samples = cached_video(path, spec, cache)
            for index, label, weight in selected_training_windows(source, samples):
                self.items.append((samples.frames[index], label, weight))
        totals = Counter()
        for _, label, weight in self.items:
            totals[label] += weight
        if set(totals) != {0, 1}:
            raise ValueError("Training requires both classes")
        self.items = [(frames, label, weight / totals[label]) for frames, label, weight in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        frames, label, weight = self.items[index]
        frames = augment_frames(frames) if self.augment else frames
        return torch.from_numpy(normalize_rgb(frames, self.spec.image_size)), torch.tensor(float(label)), torch.tensor(float(weight))
