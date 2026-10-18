import numpy as np
import pytest
import torch

from fortebank_audit.ml.evaluation import file_hash, pipeline_hash, score_sources
from fortebank_audit.ml.experiment_data import fingerprint, source_identity
from fortebank_audit.ml.preprocessing import VideoSpec
from fortebank_audit.ml.stress_audit import CONDITIONS, grouped_intervals, transform_frames, validate_fold


@pytest.mark.parametrize("condition", CONDITIONS)
def test_stress_is_deterministic_clip_consistent_and_non_mutating(condition):
    frame = np.random.default_rng(9).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    frames = np.repeat(frame[None], 4, axis=0)
    before = frames.copy()
    result = transform_frames(frames, condition)
    assert result.shape == frames.shape and result.dtype == np.uint8
    assert np.array_equal(frames, before)
    assert np.array_equal(result, transform_frames(frames, condition))
    assert np.array_equal(result[0], result[-1])
    if condition != "clean":
        assert not np.array_equal(result, frames)


def test_stress_brightness_clips_without_uint8_overflow():
    frames = np.full((2, 16, 16, 3), 255, dtype=np.uint8)
    assert transform_frames(frames, "bright").min() == 255
    with pytest.raises(ValueError):
        transform_frames(frames, "unknown")


def test_group_bootstrap_keeps_pairs_together():
    # Each scene contains one correct and one wrong positive. Resampling
    # complete scenes must keep recall at exactly 0.5 in every replicate.
    rows = [{"scenario_group": str(group), "label": label, "prediction": pred}
            for group in range(5) for label, pred in [(1, 1), (1, 0), (0, 0)]]
    result = grouped_intervals(rows, repetitions=100)
    assert result["intervals_95"]["recall"] == [0.5, 0.5]
    assert result == grouped_intervals(rows[::-1], repetitions=100)


def test_stress_rejects_recalibrated_or_wrong_fold_before_scoring(tmp_path):
    checkpoint = tmp_path / "test.pt"
    train = [{"source_id": "train", "sha256": "a", "scenario_group": "g0"}]
    cal = [{"source_id": "cal", "sha256": "b", "scenario_group": "g1"}]
    test = [{"source_id": "test", "sha256": "c", "scenario_group": "g2"}]
    torch.save({"spec": VideoSpec().to_dict(), "provenance": {
        "train_source_ids": ["train"], "train_sha256": ["a"], "train_scenario_groups": ["g0"],
        "dataset_sha256": fingerprint([source_identity(train[0])])}}, checkpoint)
    calibration = {"calibration_source_ids": ["cal"], "calibration_sha256": ["b"],
                   "calibration_scenario_groups": ["g1"], "checkpoint_sha256": file_hash(checkpoint),
                   "pipeline_sha256": pipeline_hash(VideoSpec()), "threshold": .5}
    validate_fold(checkpoint, calibration, train, cal, test)
    with pytest.raises(ValueError, match="calibration partition"):
        validate_fold(checkpoint, {**calibration, "calibration_source_ids": ["test"]}, train, cal, test)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_fold(checkpoint, {**calibration, "checkpoint_sha256": "other"}, train, cal, test)
    with pytest.raises(ValueError, match="pipeline mismatch"):
        validate_fold(checkpoint, {**calibration, "pipeline_sha256": "other"}, train, cal, test)
    with pytest.raises(ValueError, match="annotations differ"):
        validate_fold(checkpoint, calibration, [{**train[0], "label": 1}], cal, test)


def test_scoring_rejects_invalid_batch_before_loading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="batch size"):
        score_sources(tmp_path / "missing.pt", [], tmp_path, batch_size=0)
