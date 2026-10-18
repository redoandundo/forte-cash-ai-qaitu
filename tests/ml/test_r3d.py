import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights

from fortebank_audit.ml.models import KineticsR3DBinary, build_model
from fortebank_audit.ml.preprocessing import normalize_rgb


def test_r3d_adapts_contract_to_pretrained_normalization():
    capture = nn.Identity()
    model = KineticsR3DBinary(capture)
    rgb = np.random.default_rng(7).integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
    inputs = torch.from_numpy(normalize_rgb(rgb, 16))[None]
    transform = R3D_18_Weights.KINETICS400_V1.transforms()
    pixels = torch.from_numpy(rgb.astype(np.float32) / 255).permute(3, 0, 1, 2)[None]
    mean = torch.tensor(transform.mean).view(1, 3, 1, 1, 1)
    std = torch.tensor(transform.std).view(1, 3, 1, 1, 1)
    assert torch.allclose(model(inputs), ((pixels - mean) / std).flatten(), atol=5e-7)


@pytest.mark.parametrize("architecture", ["r3d_18", "r3d_18_kinetics"])
def test_r3d_frozen_statistics_and_partial_finetune(architecture):
    model = build_model(architecture, pretrained=False)
    model.train()
    norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm3d)]
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in norms]
    model(torch.randn(1, 3, 4, 16, 16)).sum().backward()
    assert all(not m.training for m in norms)
    for norm, (mean, variance) in zip(norms, before):
        assert torch.equal(norm.running_mean, mean)
        assert torch.equal(norm.running_var, variance)
    assert all(p.grad is None for p in model.model.layer4.parameters())
    model.set_trainable_blocks(1)
    assert all(p.requires_grad for p in model.model.layer4.parameters())
    assert all(not p.requires_grad for p in model.model.layer3.parameters())
    assert all(p.requires_grad for p in model.model.fc.parameters())
    assert all(not m.training for m in norms)
    with pytest.raises(ValueError):
        model.set_trainable_blocks(5)
