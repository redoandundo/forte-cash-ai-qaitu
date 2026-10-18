from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .contracts import PreprocessingConfigError


@dataclass(frozen=True)
class PreprocessingConfig:
    input_layout: str
    input_name: str
    output_name: str
    dtype: str
    color: str
    num_frames: int
    image_size: int
    normalization: str
    window_sec: float
    stride_sec: float

    @classmethod
    def from_file(cls, path: str | Path) -> "PreprocessingConfig":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            config = cls(
                input_layout=str(payload["input_layout"]),
                input_name=str(payload["input_name"]),
                output_name=str(payload["output_name"]),
                dtype=str(payload["dtype"]),
                color=str(payload["color"]),
                num_frames=int(payload["num_frames"]),
                image_size=int(payload["image_size"]),
                normalization=str(payload["normalization"]),
                window_sec=float(payload["window_sec"]),
                stride_sec=float(payload["recommended_stride_sec"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PreprocessingConfigError("Invalid preprocessing configuration") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.input_layout != "NCTHW":
            raise PreprocessingConfigError(f"Unsupported layout: {self.input_layout}")
        if self.dtype != "float32" or self.color != "RGB":
            raise PreprocessingConfigError("Only float32 RGB input is supported")
        if self.num_frames <= 0 or self.image_size <= 0:
            raise PreprocessingConfigError("Frame count and image size must be positive")
        if self.window_sec <= 0 or self.stride_sec <= 0:
            raise PreprocessingConfigError("Window and stride must be positive")
        if self.stride_sec > self.window_sec:
            raise PreprocessingConfigError("Stride cannot exceed the inference window")
        if self.normalization != "imagenet":
            raise PreprocessingConfigError(f"Unsupported normalization: {self.normalization}")


def preprocess_frames(frames_rgb: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise PreprocessingConfigError("Expected frames with shape T x H x W x 3")
    if frames_rgb.shape[0] != config.num_frames:
        raise PreprocessingConfigError(
            f"Expected {config.num_frames} frames, got {frames_rgb.shape[0]}"
        )
    resized = np.stack(
        [cv2.resize(frame, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR) for frame in frames_rgb]
    ).astype(np.float32) / 255.0
    tensor = resized.transpose(3, 0, 1, 2)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
    return ((tensor - mean) / std).astype(np.float32, copy=False)
