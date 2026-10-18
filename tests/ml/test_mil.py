import numpy as np
import pytest
import torch

from fortebank_audit.ml.metrics import smooth_scores
from fortebank_audit.ml.mil_training import multi_view_loss, smooth_probabilities, validate_features, video_loss
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


def test_single_view_training_keeps_the_previous_objective_and_gradient():
    logits = torch.tensor([-2., 0., 2., 1.], requires_grad=True)
    previous = video_loss(logits, 1, [2], .5)
    current = multi_view_loss(logits[None], 1, [2], .5, 1.)
    assert torch.equal(previous, current)
    assert torch.equal(torch.autograd.grad(previous, logits, retain_graph=True)[0],
                       torch.autograd.grad(current, logits)[0])


def test_consistency_penalizes_changed_predictions_without_changing_timestamps():
    logits = torch.tensor([[-2., 0., 2.], [2., 0., -2.]])
    assert multi_view_loss(logits, 1, [1], .5, 1.) > multi_view_loss(logits, 1, [1], .5, 0.)
    identical = logits[:1].repeat(2, 1)
    assert torch.equal(multi_view_loss(identical, 1, [1], .5, 1.),
                       multi_view_loss(identical, 1, [1], .5, 0.))
    with pytest.raises(ValueError):
        multi_view_loss(logits, 1, [1], .5, float("nan"))


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


@pytest.mark.parametrize("size", [(2, 2), (16, 16), (5, 7)])
def test_exportable_spatial_regions_preserve_adaptive_pooling(size):
    from fortebank_audit.ml.spatial_mil import spatial_grid
    features = torch.randn(3, 384, *size)
    expected = torch.nn.functional.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
    assert torch.allclose(spatial_grid(features, *size), expected, atol=1e-6)


def test_scoring_reuses_frozen_features_but_never_reuses_them_for_stress(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fortebank_audit.ml import evaluation, mil_training

    class TinyEncoder(torch.nn.Module):
        feature_dim = 3

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 3, bias=False)
            self.encoder.weight.data.copy_(torch.eye(3))
            self.calls = 0

        def encode(self, video):
            self.calls += 1
            return self.encoder(video.mean((-2, -1)).transpose(1, 2))

        def forward_encoded(self, features):
            return features.mean((1, 2))

        def forward(self, video):
            return self.forward_encoded(self.encode(video))

    model = TinyEncoder()
    spec = VideoSpec(num_frames=2, image_size=16)
    samples = SimpleNamespace(frames=np.full((3, 2, 16, 16, 3), 200, dtype=np.uint8),
                              centers=np.array([1., 2., 3.]), starts=np.array([0., 1., 2.]),
                              ends=np.array([2., 3., 4.]), duration=4.)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(evaluation, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(evaluation, "cached_video", lambda *args: samples)
    monkeypatch.setattr(mil_training, "cached_video", lambda *args: samples)
    video = tmp_path / "source.mov"
    video.write_bytes(b"synthetic sample supplied by test decoder")
    source = {"path": str(video), "sha256": evaluation.file_hash(video), "source_id": "held-out",
              "scenario_group": "held-out-group", "label": 0}
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": "dinov2_small_spatial_temporal", "state_dict": model.state_dict(),
                "spec": spec.to_dict(), "provenance": {"train_source_ids": [], "train_sha256": [],
                                                      "train_scenario_groups": []}}, checkpoint)
    cold, _, _ = evaluation.score_sources(checkpoint, [source], tmp_path / "cache")
    calls = model.calls
    warm, _, _ = evaluation.score_sources(checkpoint, [source], tmp_path / "cache")
    assert model.calls == calls
    direct, _, _ = evaluation.score_sources(checkpoint, [source], tmp_path / "cache", frame_transform=lambda x: x)
    assert cold[0]["raw_scores"] == warm[0]["raw_scores"] == direct[0]["raw_scores"]
    changed, _, _ = evaluation.score_sources(checkpoint, [source], tmp_path / "cache", frame_transform=lambda x: x * 0)
    assert model.calls > calls
    assert changed[0]["raw_scores"] != warm[0]["raw_scores"]
    digest = mil_training.encoder_hash(model)
    clean_features = mil_training.encode_source(model, source, spec, tmp_path / "cache", "cpu", digest, 2)
    dark_features = mil_training.encode_source(model, source, spec, tmp_path / "cache", "cpu", digest, 2, "dark")
    assert clean_features["key"] != dark_features["key"]
    assert not torch.equal(clean_features["features"], dark_features["features"])
    assert torch.equal(clean_features["centers"], dark_features["centers"])
    calls = model.calls
    mil_training.encode_source(model, source, spec, tmp_path / "cache", "cpu", digest, 2, "dark")
    assert model.calls == calls
