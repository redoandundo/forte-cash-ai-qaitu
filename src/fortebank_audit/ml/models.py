from __future__ import annotations

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


class TemporalStatsMobileNet(nn.Module):
    """Frame encoder that keeps appearance, salient content, and short motion cues."""

    def __init__(self, pretrained: bool = True, train_backbone: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        feature_count = backbone.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_count * 3, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        if train_backbone:
            for block in self.features[-2:].parameters():
                block.requires_grad = True

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        encoded = self.pool(self.features(frames)).flatten(1).reshape(batch, time, -1)
        appearance = encoded.mean(dim=1)
        salient = encoded.amax(dim=1)
        if time > 1:
            motion = (encoded[:, 1:] - encoded[:, :-1]).abs().mean(dim=1)
        else:
            motion = torch.zeros_like(appearance)
        return self.classifier(torch.cat((appearance, salient, motion), dim=1)).squeeze(1)


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


def build_model(name: str, pretrained: bool = True, train_backbone: bool = False) -> nn.Module:
    if name == "mobilenet_v3_small_temporal":
        return TemporalMobileNet(pretrained, train_backbone)
    if name == "mobilenet_v3_small_temporal_stats":
        return TemporalStatsMobileNet(pretrained, train_backbone)
    if name == "x3d_xs":
        return build_x3d(pretrained, train_backbone)
    if name == "r3d_18":
        return build_r3d18(pretrained, train_backbone)
    raise ValueError(f"Unknown model: {name}")
