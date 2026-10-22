from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from .preprocessing import VideoSpec


@dataclass
class VideoSamples:
    frames: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    centers: np.ndarray
    actual_timestamps: np.ndarray
    duration: float


def plan_windows(duration: float, spec: VideoSpec) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be positive and finite")
    last_start = max(0.0, duration - spec.window_sec)
    starts = np.arange(0.0, last_start + 1e-9, spec.stride_sec)
    if spec.include_tail and last_start - starts[-1] > 1e-6:
        starts = np.append(starts, last_start)
    return starts, np.minimum(starts + spec.window_sec, duration)


def sample_video(path: Path, spec: VideoSpec) -> VideoSamples:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else float((container.duration or 0) / av.time_base)
        starts, ends = plan_windows(duration, spec)
        plans = np.stack([np.linspace(start, end, spec.num_frames, endpoint=False) for start, end in zip(starts, ends)])
        targets = np.unique(plans)
        selected, times = [], []
        origin = float((stream.start_time or 0) * stream.time_base)
        fps = float(stream.average_rate or 0)
        previous = None
        last_time = -math.inf
        index = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Missing frame PTS; ML calibration requires real timestamps")
            timestamp = float(frame.pts * frame.time_base) - origin
            if timestamp < -1e-6 or timestamp < last_time - 1e-9:
                raise ValueError("Invalid or non-monotonic PTS")
            timestamp = max(0.0, timestamp)
            rgb = cv2.resize(frame.to_ndarray(format="rgb24"), (spec.image_size, spec.image_size), interpolation=cv2.INTER_LINEAR)
            current = (timestamp, rgb)
            while index < len(targets) and targets[index] <= timestamp:
                chosen = min([previous, current] if previous is not None else [current], key=lambda pair: abs(pair[0] - targets[index]))
                times.append(chosen[0])
                selected.append(chosen[1])
                index += 1
            previous = current
            last_time = timestamp
        if previous is None:
            raise ValueError("Video has no decodable frames")
        tolerance = max(0.25, 2 / fps) if fps else 0.25
        if duration - last_time > tolerance:
            raise ValueError("Decode ended before the advertised video duration")
        while index < len(targets):
            if targets[index] - last_time > tolerance:
                raise ValueError("Missing requested video frame")
            times.append(last_time)
            selected.append(previous[1])
            index += 1
    indices = np.searchsorted(targets, plans)
    return VideoSamples(np.stack(selected)[indices], starts, ends, (starts + ends) / 2, np.asarray(times)[indices], duration)


def cached_video(path: Path, spec: VideoSpec, cache: Path) -> VideoSamples:
    # Content hash protects against stale samples after replacing a recording.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = hashlib.sha256((digest + json.dumps(spec.to_dict(), sort_keys=True) + "pts-v1").encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (key + ".npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            return VideoSamples(saved["frames"], saved["starts"], saved["ends"], saved["centers"], saved["actual_timestamps"], float(saved["duration"]))
    samples = sample_video(path, spec)
    np.savez(target, frames=samples.frames, starts=samples.starts, ends=samples.ends, centers=samples.centers, actual_timestamps=samples.actual_timestamps, duration=samples.duration)
    return samples
