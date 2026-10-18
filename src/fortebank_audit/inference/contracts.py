from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class InferenceError(RuntimeError):
    """Base class for technical inference failures."""


class VideoOpenError(InferenceError):
    pass


class UnsupportedVideoError(InferenceError):
    pass


class PreprocessingConfigError(InferenceError):
    pass


class PredictorLoadError(InferenceError):
    pass


class PredictionError(InferenceError):
    pass


class EvidenceExtractionError(InferenceError):
    pass


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


@dataclass(frozen=True)
class VideoFrame:
    timestamp_sec: float
    image_rgb: np.ndarray


@dataclass(frozen=True)
class WindowPlan:
    index: int
    start_sec: float
    end_sec: float
    center_sec: float
    sample_times_sec: tuple[float, ...]


@dataclass(frozen=True)
class ClipWindow:
    index: int
    start_sec: float
    end_sec: float
    center_sec: float
    frames_rgb: np.ndarray


@dataclass(frozen=True)
class WindowPrediction:
    window_index: int
    start_sec: float
    end_sec: float
    center_sec: float
    score: float


@dataclass(frozen=True)
class DetectedEvent:
    start_sec: float
    end_sec: float
    peak_time_sec: float
    peak_score: float
    supporting_windows: int
    supporting_window_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class EvidenceFrame:
    timecode_sec: float
    jpeg_path: Path


@dataclass(frozen=True)
class PipelineConfig:
    window_sec: float
    stride_sec: float
    num_frames: int
    threshold: float
    aggregation: str = "none"
    sampling: str = "pts_nearest"
    short_sequence_rule: str = "edge_pad"
    event_merge: str = "overlapping_windows"
    batch_size: int = 8
    merge_gap_sec: float = 1.5
    max_events: int = 5
    include_tail_window: bool = False
    evidence_count: int = 3
    max_windows: int = 100_000

    def __post_init__(self) -> None:
        if self.window_sec <= 0 or self.stride_sec <= 0:
            raise ValueError("window_sec and stride_sec must be positive")
        if self.stride_sec > self.window_sec:
            raise ValueError("stride_sec cannot exceed window_sec")
        if self.num_frames <= 0 or self.batch_size <= 0:
            raise ValueError("num_frames and batch_size must be positive")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if self.aggregation not in {"none", "three_window_weighted_smoothing_then_peak"}:
            raise ValueError("Unsupported aggregation strategy")
        if self.sampling not in {"pts_nearest", "opencv_frame_index_round"}:
            raise ValueError("Unsupported video sampling strategy")
        if self.short_sequence_rule not in {"edge_pad", "identity_if_fewer_than_three"}:
            raise ValueError("Unsupported short-sequence smoothing rule")
        if self.event_merge not in {"overlapping_windows", "adjacent_centers_within_stride"}:
            raise ValueError("Unsupported event merge rule")
        if self.merge_gap_sec < 0 or self.max_events <= 0:
            raise ValueError("merge_gap_sec must be non-negative and max_events positive")
        if self.evidence_count not in {1, 2, 3}:
            raise ValueError("evidence_count must be 1, 2, or 3")
        if self.max_windows <= 0:
            raise ValueError("max_windows must be positive")


InferenceResult = dict[str, Any]
