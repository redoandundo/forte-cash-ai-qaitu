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
        self.set_trainable_blocks(2 if train_backbone else 0)

    def set_trainable_blocks(self, count: int) -> None:
        if not 0 <= count <= len(self.features):
            raise ValueError("Invalid number of trainable backbone blocks")
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        if count:
            for block in self.features[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        # Small batches must not change pretrained BatchNorm running statistics.
        for module in self.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        encoded = self.pool(self.features(frames)).flatten(1).reshape(batch, time, -1).mean(dim=1)
        return self.classifier(encoded).squeeze(1)


class TemporalConvMobileNet(TemporalMobileNet):
    """A small temporal convolution preserves local frame order."""

    def __init__(self, pretrained: bool = True, train_backbone: bool = False):
        super().__init__(pretrained, train_backbone)
        feature_count = self.classifier[0].in_features
        self.temporal = nn.Sequential(
            nn.Conv1d(feature_count, 128, kernel_size=1),
            nn.Hardswish(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.Hardswish(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        encoded = self.pool(self.features(frames)).flatten(1).reshape(batch, time, -1)
        return self.classifier(self.temporal(encoded.transpose(1, 2)).squeeze(-1)).squeeze(1)


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

    def set_trainable_blocks(self, count: int) -> None:
        if type(count) is not int or not 0 <= count <= 4:
            raise ValueError("R3D trainable stages must be an integer from 0 to 4")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for stage in range(5 - count, 5):
            for parameter in getattr(self.model, f"layer{stage}").parameters():
                parameter.requires_grad_(True)
        for parameter in self.model.fc.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        # Updating pretrained statistics with tiny video batches corrupts the
        # supposedly frozen feature extractor, even without backbone gradients.
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.model(video).flatten()


class KineticsR3DBinary(R3DBinary):
    """Keep the public ImageNet input contract; adapt inside the ONNX graph.

    A separate architecture name preserves legacy r3d_18 checkpoint semantics.
    Spatial sampling remains the project's full-frame stretch, not center crop.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        transform = R3D_18_Weights.KINETICS400_V1.transforms()
        video_mean = torch.tensor(transform.mean).view(1, 3, 1, 1, 1)
        video_std = torch.tensor(transform.std).view(1, 3, 1, 1, 1)
        self.register_buffer("input_scale", std / video_std)
        self.register_buffer("input_shift", (mean - video_mean) / video_std)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.model(video * self.input_scale + self.input_shift).flatten()


class X3DBinary(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.model(video).flatten(1).mean(1)


def build_model(name: str, pretrained: bool = True, train_backbone: bool = False) -> nn.Module:
    if name == "mobilenet_v3_small_temporal":
        return TemporalMobileNet(pretrained, train_backbone)
    if name == "mobilenet_v3_small_temporal_conv1d":
        return TemporalConvMobileNet(pretrained, train_backbone)
    if name == "x3d_xs":
        return build_x3d(pretrained, train_backbone)
    if name == "r3d_18":
        return build_r3d18(pretrained, train_backbone)
    if name == "r3d_18_kinetics":
        return KineticsR3DBinary(build_r3d18(pretrained, train_backbone).model)
    raise ValueError(f"Unknown model: {name}")
