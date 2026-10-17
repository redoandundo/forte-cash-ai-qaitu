from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.video import R3D_18_Weights, r3d_18


class TemporalMobileNet(nn.Module):
    """A small 2D encoder applied per frame with temporal mean pooling."""

    def __init__(self, pretrained: bool = True, train_backbone: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        feature_count = backbone.classifier[0].in_features
        self.classifier = nn.Sequential(nn.Linear(feature_count, 128), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(128, 1))
        for parameter in self.features.parameters():
            parameter.requires_grad = train_backbone

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        encoded = self.pool(self.features(frames)).flatten(1).reshape(batch, time, -1).mean(dim=1)
        return self.classifier(encoded).squeeze(1)


class TemporalConvHead(nn.Module):
    """Small temporal classifier for frozen per-frame foundation features."""

    def __init__(self, feature_count: int, hidden_count: int = 256, dropout: float = 0.25):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_count, hidden_count),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_count, hidden_count, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_count, hidden_count, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.attention = nn.Linear(hidden_count, 1)
        self.classifier = nn.Linear(hidden_count, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(features)
        encoded = encoded + self.temporal(encoded.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled).squeeze(1)


class DinoV3Temporal(nn.Module):
    """Frozen local DINOv3 frame encoder with a trainable temporal head."""

    def __init__(self, pretrained_path: str | Path, train_backbone: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(str(pretrained_path), local_files_only=True)
        self.train_backbone = train_backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if train_backbone:
            encoder = getattr(self.backbone, "encoder", None)
            if encoder is None:
                raise ValueError("The local DINO backbone has no supported encoder stages to unfreeze")
            if hasattr(encoder, "layer"):
                last_stage = encoder.layer[-1]
            elif hasattr(encoder, "stages"):
                last_stage = encoder.stages[-1]
            else:
                raise ValueError("The local DINO backbone has no supported final stage to unfreeze")
            for parameter in last_stage.parameters():
                parameter.requires_grad = True
        feature_count = getattr(self.backbone.config, "hidden_size", None)
        if feature_count is None:
            feature_count = self.backbone.config.hidden_sizes[-1]
        self.temporal_head = TemporalConvHead(int(feature_count))

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        context = nullcontext() if self.train_backbone else torch.no_grad()
        with context:
            outputs = self.backbone(pixel_values=frames)
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                features = outputs.last_hidden_state
                if features.ndim == 4:
                    features = features.mean(dim=(-2, -1))
                elif features.ndim == 3:
                    features = features.mean(dim=1)
        return self.temporal_head(features.reshape(batch, time, -1))


class VideoMAEBinary(nn.Module):
    """Local Kinetics-pretrained VideoMAE with a binary classification head."""

    def __init__(self, pretrained_path: str | Path, train_backbone: bool = False):
        super().__init__()
        from transformers import VideoMAEForVideoClassification

        self.model = VideoMAEForVideoClassification.from_pretrained(str(pretrained_path), local_files_only=True)
        self.train_backbone = train_backbone
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if train_backbone:
            for parameter in self.model.videomae.encoder.layer[-1].parameters():
                parameter.requires_grad = True
        feature_count = self.model.classifier.in_features
        self.model.classifier = nn.Linear(feature_count, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.model.videomae.eval()
        return self

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        frames = video.permute(0, 2, 1, 3, 4).contiguous()
        context = nullcontext() if self.train_backbone else torch.no_grad()
        with context:
            hidden = self.model.videomae(pixel_values=frames).last_hidden_state
        pooled = self.model.fc_norm(hidden.mean(dim=1))
        return self.model.classifier(pooled).squeeze(1)


def build_x3d(pretrained: bool = True, train_backbone: bool = False) -> nn.Module:
    from pytorchvideo.models.hub import x3d_xs

    model = x3d_xs(pretrained=pretrained)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, 1)
    model.blocks[-1].activation = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = train_backbone
    for parameter in model.blocks[-1].parameters():
        parameter.requires_grad = True
    return X3DBinary(model)


def build_r3d18(pretrained: bool = True, train_backbone: bool = False) -> nn.Module:
    """Kinetics-pretrained 3D baseline with an optional trainable final stage."""
    weights = R3D_18_Weights.DEFAULT if pretrained else None
    model = r3d_18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if train_backbone:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 1)
    return R3DBinary(model)


class R3DBinary(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.model(video).flatten()


class X3DBinary(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.model(video).flatten(1).mean(1)


def build_model(
    name: str,
    pretrained: bool = True,
    train_backbone: bool = False,
    pretrained_path: str | Path | None = None,
) -> nn.Module:
    if name == "mobilenet_v3_small_temporal":
        return TemporalMobileNet(pretrained, train_backbone)
    if name == "x3d_xs":
        return build_x3d(pretrained, train_backbone)
    if name == "r3d_18":
        return build_r3d18(pretrained, train_backbone)
    if name in {"dinov2_small_temporal", "dinov3_convnext_tiny_temporal"}:
        if pretrained_path is None:
            raise ValueError(f"{name} requires a local pretrained_path")
        return DinoV3Temporal(pretrained_path, train_backbone)
    if name == "videomae_base":
        if pretrained_path is None:
            raise ValueError("videomae_base requires a local pretrained_path")
        return VideoMAEBinary(pretrained_path, train_backbone)
    raise ValueError(f"Unknown model: {name}")
