from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

KINETICS_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
R3D_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
R3D_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def read_clip(path: str | Path, frames: int, temporal_span: float = 1.0, temporal_jitter: bool = False) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if length < 1:
        capture.release()
        raise RuntimeError(f"Could not decode {path}")
    span = max(frames, round(length * temporal_span))
    maximum_start = max(0, length - span)
    start = random.randint(0, maximum_start) if temporal_jitter and maximum_start else maximum_start // 2
    indices = np.linspace(start, min(length - 1, start + span - 1), frames).round().astype(int)
    output = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not read frame {index} from {path}")
        output.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(output)


def _jpeg(frames: np.ndarray, quality: int) -> np.ndarray:
    compressed = []
    for rgb in frames:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG augmentation failed")
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        compressed.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(compressed)


def preprocess(frames: np.ndarray, image_size: int, augment: bool, normalization: str = "kinetics") -> torch.Tensor:
    if augment and random.random() < 0.55:
        frames = _jpeg(frames, random.randint(35, 90))
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    if augment:
        angle = random.uniform(-7, 7)
        translate = [round(random.uniform(-0.06, 0.06) * image_size) for _ in range(2)]
        scale = random.uniform(0.92, 1.08)
        tensor = TF.affine(tensor, angle, translate, scale, 0.0, interpolation=InterpolationMode.BILINEAR)
        if random.random() < 0.4:
            displacement = round(image_size * 0.06)
            startpoints = [[0, 0], [image_size - 1, 0], [image_size - 1, image_size - 1], [0, image_size - 1]]
            endpoints = [
                [random.randint(0, displacement), random.randint(0, displacement)],
                [image_size - 1 - random.randint(0, displacement), random.randint(0, displacement)],
                [image_size - 1 - random.randint(0, displacement), image_size - 1 - random.randint(0, displacement)],
                [random.randint(0, displacement), image_size - 1 - random.randint(0, displacement)],
            ]
            tensor = TF.perspective(tensor, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR)
        brightness = random.uniform(0.75, 1.25)
        contrast = random.uniform(0.75, 1.25)
        gamma = random.uniform(0.8, 1.2)
        tensor = TF.adjust_brightness(tensor, brightness)
        tensor = TF.adjust_contrast(tensor, contrast)
        tensor = TF.adjust_gamma(tensor.clamp(0, 1), gamma)
        if random.random() < 0.2:
            tensor = TF.gaussian_blur(tensor, kernel_size=3, sigma=random.uniform(0.1, 1.0))
        if random.random() < 0.2:
            tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.003, 0.015)).clamp(0, 1)
    video = tensor.permute(1, 0, 2, 3).contiguous()
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalization == "r3d":
        mean, std = R3D_MEAN, R3D_STD
    else:
        mean, std = KINETICS_MEAN, KINETICS_STD
    return (video - mean) / std


class ClipDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, frames: int, image_size: int, augment: bool = False, normalization: str = "kinetics", temporal_span: float = 1.0):
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.items = [item for item in payload["clips"] if item["split"] == split]
        self.frames = frames
        self.image_size = image_size
        self.augment = augment
        self.normalization = normalization
        self.temporal_span = temporal_span

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        video = preprocess(read_clip(item["path"], self.frames, self.temporal_span, self.augment), self.image_size, self.augment, self.normalization)
        weight = float(item.get("sample_weight", 1.0))
        return video, torch.tensor(float(item["label"])), torch.tensor(weight), item["source_id"], item["clip_id"]


class ScenarioPairDataset(Dataset):
    """Positive/negative clip pairs from the same verified scenario group."""

    def __init__(self, manifest: str | Path, split: str, frames: int, image_size: int, augment: bool = False, normalization: str = "kinetics", temporal_span: float = 1.0):
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: {0: [], 1: []})
        for item in payload["clips"]:
            if item["split"] == split and item.get("scenario_group"):
                grouped[item["scenario_group"]][int(item["label"])].append(item)
        self.pairs = [
            (positive, negative, group)
            for group, labels in grouped.items()
            for positive in labels[1]
            for negative in labels[0]
        ]
        self.frames = frames
        self.image_size = image_size
        self.augment = augment
        self.normalization = normalization
        self.temporal_span = temporal_span

    def __len__(self) -> int:
        return len(self.pairs)

    def _video(self, item: dict) -> torch.Tensor:
        return preprocess(
            read_clip(item["path"], self.frames, self.temporal_span, self.augment),
            self.image_size,
            self.augment,
            self.normalization,
        )

    def __getitem__(self, index: int):
        positive, negative, group = self.pairs[index]
        confidence_weight = {"high": 1.0, "medium": 0.75, "low": 0.5}[positive.get("annotation_confidence", "high")]
        return self._video(positive), self._video(negative), torch.tensor(confidence_weight), group
