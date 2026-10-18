import numpy as np
import pytest
import torch

from fortebank_audit.ml.metrics import smooth_scores
from fortebank_audit.ml.mil_training import smooth_probabilities, validate_features, video_loss
from fortebank_audit.ml.preprocessing import VideoSpec


@pytest.mark.parametrize("count", [1, 2, 8])
def test_mil_aggregation_matches_exported_runtime(count):
    logits = torch.linspace(-2, 2, count)
    expected = smooth_scores(logits.sigmoid().numpy())
    assert np.allclose(smooth_probabilities(logits).numpy(), expected, atol=1e-7)


def test_every_negative_window_is_suppressed_including_unselected_peaks():
    logits = torch.tensor([-2., -1., 3., -1., 0., -2.], requires_grad=True)
    video_loss(logits, 0, [], .5).backward()
    assert torch.all(logits.grad > 0)
    assert logits.grad[2] > logits.grad[0]


def test_positive_context_is_not_invented_as_negative():
    logits = torch.tensor([-2., 0., 2., 1., -3., -2.], requires_grad=True)
    video_loss(logits, 1, [1], .5).backward()
    assert torch.all(logits.grad <= 0)
    assert logits.grad[1] < 0
    with pytest.raises(ValueError, match="anchors"):
        video_loss(logits, 0, [1], .5)
    with pytest.raises(ValueError, match="anchors"):
        video_loss(logits, 1, [], .5)


@pytest.mark.parametrize("fault", ["nan", "wrong_frames", "unordered_times", "missing_times", "wrong_encoder"])
def test_corrupt_or_incompatible_feature_cache_is_rejected(fault):
    spec = VideoSpec(num_frames=8, image_size=224)
    saved = {"key": "expected", "features": torch.zeros(2, 8, 1920), "centers": torch.tensor([3., 5.])}
    validate_features(saved, "expected", spec, 1920)
    if fault == "nan":
        saved["features"][0, 0, 0] = float("nan")
    elif fault == "wrong_frames":
        saved["features"] = saved["features"][:, :4]
    elif fault == "unordered_times":
        saved["centers"] = saved["centers"].flip(0)
    elif fault == "missing_times":
        del saved["centers"]
    else:
        saved["key"] = "different_encoder"
    with pytest.raises(ValueError, match="feature cache"):
        validate_features(saved, "expected", spec, 1920)


def test_spatial_encoder_cache_and_temporal_head_agree():
    pytest.importorskip("transformers")
    from fortebank_audit.ml.models import build_model
    torch.manual_seed(3)
    torch.set_num_threads(2)
    model = build_model("dinov2_small_spatial_temporal", pretrained=False).eval()
    video = torch.randn(1, 3, 2, 28, 28)
    with torch.no_grad():
        encoded = model.encode(video)
        assert encoded.shape == (1, 2, 1920)
        assert torch.allclose(model(video), model.forward_encoded(encoded), atol=1e-7)
        assert not torch.allclose(model.forward_encoded(encoded), model.forward_encoded(encoded.flip(1)), atol=1e-7)
    model.train()
    assert not model.encoder.training
    model.forward_encoded(encoded).sum().backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.classifier.parameters())
