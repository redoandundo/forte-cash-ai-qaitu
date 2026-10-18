from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from fortebank_audit.ml.diagnostics import audit_inputs, shifted_video, compare_shift
from fortebank_audit.ml.experiment_data import validate_sources, selected_training_windows, SourceWindowDataset
from fortebank_audit.ml.preprocessing import normalize_rgb, VideoSpec
from fortebank_audit.ml.training import loss_denominator, weighted_bce
from fortebank_audit.ml.video import sample_video


def test_changed_frames_must_give_changed_inputs():
    frames = np.zeros((2, 4, 32, 32, 3), dtype=np.uint8)
    frames[1] = 127
    tensors = np.stack([normalize_rgb(item, 32) for item in frames])
    assert audit_inputs(frames, tensors)["unique_tensors"] == 2
    with pytest.raises(ValueError, match="collapsed"):
        audit_inputs(frames, np.stack([tensors[0], tensors[0]]))
    assert audit_inputs(frames[:1].repeat(2, axis=0), tensors[:1].repeat(2, axis=0))["changed_adjacent_windows"] == 0


def test_loss_dataset_mean_preserves_weighted_objective_across_batches():
    targets, weights = torch.tensor([1., 0., 0.]), torch.tensor([1., .1, .1])
    full = torch.tensor(0., requires_grad=True)
    weighted_bce(full.expand(3), targets, weights, weights.sum()).backward()
    accumulated = torch.tensor(0., requires_grad=True)
    biased = torch.tensor(0., requires_grad=True)
    for target, weight in zip(targets, weights):
        group = [(None, target[None], weight[None])]
        denominator = loss_denominator(group, weights.mean().item(), "dataset_mean")
        (weighted_bce(accumulated[None], target[None], weight[None], denominator) / 3).backward()
        (weighted_bce(biased[None], target[None], weight[None], weight) / 3).backward()
    assert torch.allclose(full.grad, accumulated.grad)
    assert torch.sign(full.grad) != torch.sign(biased.grad)


@pytest.mark.parametrize("update", [{"events_complete": "false"}, {"events_complete": True},
                                   {"events_complete": True, "review_status": "confirmed", "reviewer": "reviewer"}])
def test_points_cannot_claim_complete_reviewed_intervals(update):
    row = {"source_id": "one", "sha256": "abc", "label": 1, "scenario_group": "group",
           "duration_sec": 10, "events_sec": [4], **update}
    with pytest.raises(ValueError, match="events|Complete"):
        validate_sources([row])


@pytest.mark.parametrize("offset", [2., -2.])
def test_shift_preserves_pts_content_and_original(tmp_path, offset):
    source = tmp_path / "касса.mkv"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index in range(90):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 2, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index * 100 + (10 if index % 2 else 0), Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    original_bytes = source.read_bytes()
    target = tmp_path / "shifted.mkv"
    assert shifted_video(source, target, offset) == offset
    before, after = sample_video(source, VideoSpec(image_size=32)), sample_video(target, VideoSpec(image_size=32))
    pairs = 0
    for index, start in enumerate(before.starts):
        matches = np.flatnonzero(np.isclose(after.starts, start + offset))
        if len(matches):
            assert np.array_equal(before.frames[index], after.frames[matches[0]])
            pairs += 1
    assert pairs > 0 and source.read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        shifted_video(source, target, offset)


def test_shift_diagnostic_does_not_hide_failed_localization():
    before = {"spec": {"stride_sec": 2}, "peak_time_sec": 10, "windows": [], "events": []}
    after = {**before, "peak_time_sec": 10}
    result = compare_shift(before, after, 4)
    assert not result["peak_shift_passed"] and result["peak_shift_error_sec"] == 4


def test_hard_negative_budget_preserves_source_weight_and_anchors(tmp_path):
    from types import SimpleNamespace
    samples = SimpleNamespace(frames=np.zeros((10, 4)))
    source = {"source_id": "train", "label": 0}
    windows = selected_training_windows(source, samples, [1, 8])
    assert [row[0] for row in windows] == [0, 1, 4, 8, 9]
    assert sum(row[2] for row in windows) == pytest.approx(1.)
    with pytest.raises(ValueError, match="training negatives"):
        SourceWindowDataset([source], VideoSpec(), tmp_path, hard_negatives={"test": [1]})
    with pytest.raises(ValueError, match="Invalid"):
        selected_training_windows(source, samples, [10])
