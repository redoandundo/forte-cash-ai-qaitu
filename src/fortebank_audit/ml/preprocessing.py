from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoSpec:
    num_frames: int = 4
    image_size: int = 160
    window_sec: float = 6.0
    stride_sec: float = 2.0
    include_tail: bool = True

    def __post_init__(self):
        if self.num_frames < 2 or self.image_size < 16:
            raise ValueError("Invalid frame count or resolution")
        if not 0 < self.stride_sec <= self.window_sec or not np.isfinite(self.window_sec):
            raise ValueError("Invalid window or stride")

    def to_dict(self) -> dict:
        return asdict(self)


def resize_rgb(frames: np.ndarray, image_size: int) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB frames, shape T,H,W,3")
    return np.stack([cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR) for frame in frames])


def normalize_rgb(frames: np.ndarray, image_size: int) -> np.ndarray:
    tensor = resize_rgb(frames, image_size).astype(np.float32).transpose(3, 0, 1, 2) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None, None]
    return np.ascontiguousarray((tensor - mean) / std)


def preprocessing_payload(spec: VideoSpec) -> dict:
    return {
        "input_layout": "NCTHW", "input_name": "video", "output_name": "logits",
        "dtype": "float32", "color": "RGB", "normalization": "imagenet",
        "num_frames": spec.num_frames, "image_size": spec.image_size,
        "window_sec": spec.window_sec, "recommended_stride_sec": spec.stride_sec,
        "temporal_span_fraction": 1.0, "resize": "opencv_inter_linear_stretch",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "sampling": "uniform_endpoint_false_nearest_pts", "include_tail_window": spec.include_tail,
        "timestamp_origin": "stream_start_time", "pipeline_version": "pts-tail-smooth-v1",
    }
