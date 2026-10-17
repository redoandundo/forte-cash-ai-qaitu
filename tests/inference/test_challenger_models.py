from __future__ import annotations

import pytest
import torch

from fortebank_audit.ml.models import TemporalConvHead, build_model


def test_temporal_conv_head_returns_one_logit_per_video() -> None:
    head = TemporalConvHead(feature_count=32, hidden_count=16, dropout=0.0)
    output = head(torch.randn(3, 7, 32))
    assert output.shape == (3,)


@pytest.mark.parametrize("name", ["dinov2_small_temporal", "dinov3_convnext_tiny_temporal", "videomae_base"])
def test_local_foundation_models_require_explicit_weights(name: str) -> None:
    with pytest.raises(ValueError, match="local pretrained_path"):
        build_model(name)
