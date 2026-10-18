"""Frozen spatial DINO features with a trainable, order-sensitive video head."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SpatialTemporalDino(nn.Module):
    feature_dim = 384 * 5
    pretrained_revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    pretrained_files = {
        "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
        "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    }

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Optional dependency: existing MobileNet/R3D deployments do not import it.
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                             image_size=518, patch_size=14, layerscale_value=1.0)
        config._attn_implementation = "eager"
        if pretrained:
            folder = Path(os.environ.get("QAITU_DINO_WEIGHTS", "artifacts/ml/pretrained/dinov2-small"))
            for name, expected in self.pretrained_files.items():
                artifact = folder / name
                if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Missing or incompatible pinned DINO artifact: {artifact}")
            self.encoder = Dinov2Model.from_pretrained(folder, local_files_only=True,
                use_safetensors=True, attn_implementation="eager")
            if self.encoder.config.hidden_size != 384 or self.encoder.config.patch_size != 14:
                raise ValueError("Expected the pinned DINOv2-small encoder")
        else:
            self.encoder = Dinov2Model(config)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.projection = nn.Sequential(nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, 64), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(256, 64), nn.GELU(), nn.Dropout(.3), nn.Linear(64, 1))
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        tokens = self.encoder(pixel_values=frames, return_dict=False)[0]
        spatial = tokens[:, 1:].transpose(1, 2).reshape(batch * time, 384, height // 14, width // 14)
        grid = F.adaptive_avg_pool2d(spatial, (2, 2)).flatten(1)
        return torch.cat([tokens[:, 0], grid], dim=1).reshape(batch, time, self.feature_dim)

    def forward_encoded(self, features: torch.Tensor) -> torch.Tensor:
        sequence = self.projection(features)
        summary = torch.cat([sequence.mean(1), sequence[:, -1] - sequence[:, 0],
                             (sequence[:, 1:] - sequence[:, :-1]).abs().mean(1), sequence[:, -1]], dim=1)
        return self.classifier(summary).squeeze(1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.forward_encoded(self.encode(video))
