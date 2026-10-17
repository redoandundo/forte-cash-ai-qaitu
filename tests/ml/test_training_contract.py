from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from fortebank_audit.ml.experiment_data import assert_disjoint, make_folds, selected_training_windows, validate_sources
from fortebank_audit.ml.metrics import calibrate_threshold, events_from_scores, match_events, smooth_scores
from fortebank_audit.ml.models import build_model
from fortebank_audit.ml.preprocessing import VideoSpec, normalize_rgb
from fortebank_audit.ml.training import weighted_bce
from fortebank_audit.ml.video import plan_windows

torch.set_num_threads(2)


def sources():
    return [{"source_id": str(i), "sha256": f"{i:064x}", "scenario_group": f"pair-{i // 2}", "label": i % 2, "events_sec": [3.0] if i % 2 else [], "duration_sec": 6.0, "annotation_confidence": "high"} for i in range(20)]


def test_duplicate_and_scenario_leakage_rejected():
    rows = sources()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sources(rows + [rows[0]])
    with pytest.raises(ValueError, match="scenario_group"):
        assert_disjoint([rows[0]], [rows[1]])


def test_negative_source_cannot_have_confirmed_positive_interval():
    row = sources()[0]
    row["events"] = [{"start_sec": 1, "end_sec": 2, "review_status": "confirmed"}]
    with pytest.raises(ValueError, match="disagree"):
        validate_sources([row])


def test_negative_windows_cover_both_ends_and_intervals_enable_safe_negatives():
    samples = SimpleNamespace(centers=np.arange(3, 34, 2), frames=np.zeros((16, 4)),
                              starts=np.arange(0, 32, 2), ends=np.arange(6, 38, 2))
    negative = selected_training_windows(sources()[0], samples)
    assert negative[0][0] == 0 and negative[-1][0] == 15 and len(negative) == 5
    positive = sources()[1]
    positive.update(events=[{"start_sec": 14, "end_sec": 18}], events_complete=True)
    selected = selected_training_windows(positive, samples)
    assert {label for _, label, _ in selected} == {0, 1}
    for index, label, _ in selected:
        if not label:
            assert samples.ends[index] < 12 or samples.starts[index] > 20


def test_folds_deterministic_and_scenario_safe():
    rows = sources()
    folds = make_folds(rows)
    assert folds == make_folds(rows[::-1])
    for i in range(0, 20, 2):
        assert folds[str(i)] == folds[str(i + 1)]
    assert set(folds.values()) == set(range(5))


def test_unreviewed_positive_does_not_create_negative_windows():
    samples = SimpleNamespace(centers=np.array([3., 5., 7.]), frames=np.zeros((3, 4)), starts=np.array([0., 2., 4.]), ends=np.array([6., 8., 10.]))
    row = sources()[1]
    row["annotation_confidence"] = "low"
    selected = selected_training_windows(row, samples)
    assert {label for _, label, _ in selected} == {1}
    assert sum(weight for _, _, weight in selected) == 0.25


def test_partial_unfreeze_and_batchnorm_remain_fixed():
    model = build_model("mobilenet_v3_small_temporal", pretrained=False, train_backbone=True)
    assert all(not p.requires_grad for block in model.features[:-2] for p in block.parameters())
    assert all(p.requires_grad for block in model.features[-2:] for p in block.parameters())
    model.train()
    norms = [m for m in model.features.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in norms]
    model(torch.randn(2, 3, 4, 32, 32)).sum().backward()
    assert all(not m.training for m in norms)
    for m, (mean, variance) in zip(norms, before):
        assert torch.equal(m.running_mean, mean) and torch.equal(m.running_var, variance)


def test_temporal_head_uses_frame_order():
    torch.manual_seed(7)
    model = build_model("mobilenet_v3_small_temporal_conv1d", pretrained=False)
    values = torch.randn(2, model.temporal[0].in_channels, 8)
    assert not torch.allclose(model.temporal(values), model.temporal(values.flip(-1)), atol=1e-6)


def test_accumulation_matches_weighted_batch_including_remainder():
    x = torch.tensor([0.5, -0.4, 2.0, 0.1, -0.8])
    targets = torch.tensor([1., 0., 1., 0., 1.])
    weights = torch.tensor([0.1, 0.5, 0.2, 0.4, 0.6])
    full = torch.tensor(0.7, requires_grad=True)
    weighted_bce(full * x, targets, weights, weights.sum()).backward()
    accumulated = torch.tensor(0.7, requires_grad=True)
    for start in range(0, len(x), 2):
        weighted_bce(accumulated * x[start:start+2], targets[start:start+2], weights[start:start+2], weights.sum()).backward()
    assert torch.allclose(full.grad, accumulated.grad)


@pytest.mark.parametrize("duration,last_start,center", [(1., 0., 0.5), (6., 0., 3.), (11., 5., 8.)])
def test_short_exact_tail(duration, last_start, center):
    starts, ends = plan_windows(duration, VideoSpec())
    assert starts[-1] == last_start and ends[-1] == duration
    assert (starts[-1] + ends[-1]) / 2 == center
    assert len(np.unique(starts)) == len(starts)


def test_validation_preprocessing_matches_opencv_inference():
    frames = np.random.default_rng(42).integers(0, 256, (4, 55, 77, 3), dtype=np.uint8)
    resized = np.stack([cv2.resize(f, (160, 160), interpolation=cv2.INTER_LINEAR) for f in frames]).astype(np.float32) / 255.0
    mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None, None]
    assert np.array_equal(normalize_rgb(frames, 160), (resized.transpose(3, 0, 1, 2) - mean) / std)


def test_threshold_constraint_and_invalid_scores():
    result = calibrate_threshold([1, 1, 1, 1, 0, 0], [.9, .8, .2, .1, .7, .6])
    assert result["recall"] >= .75
    with pytest.raises(ValueError):
        calibrate_threshold([0, 1], [float("nan"), .5])


def test_equal_metric_threshold_uses_margin_instead_of_a_score_boundary():
    result = calibrate_threshold([0, 0, 1, 1], [.1, .2, .7, .8])
    assert result["threshold"] == pytest.approx(.45)
    assert result["min_score_margin"] == pytest.approx(.25)


def test_emitted_events_are_separate_and_matches_are_one_to_one():
    starts = np.arange(0, 16, 2)
    events = events_from_scores(starts, starts + 6, [.1, .8, .9, .1, .1, .1, .8, .1], .5, 2)
    assert len(events) == 2
    assert [e["peak_time_sec"] for e in events] == [7., 15.]
    matches = match_events([7.], [7., 8.])
    assert matches["tp"] == 1 and matches["fn"] == 1
    assert match_events([], [3.])["fn"] == 1
    assert match_events([20.], [3.])["errors_sec"] == [17.]


def test_smoothing_constant_and_short_sequences():
    assert np.allclose(smooth_scores([.2] * 5), .2)
    assert smooth_scores([.1, .7]).tolist() == [.1, .7]
