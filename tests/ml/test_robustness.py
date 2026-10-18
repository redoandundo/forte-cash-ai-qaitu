import numpy as np
import pytest

from fortebank_audit.ml.robustness import CONDITIONS, paired_changes, perturb
from fortebank_audit.ml.evaluation import calibration_diagnostics


@pytest.mark.parametrize("condition", CONDITIONS)
def test_stress_is_deterministic_nonmutating_temporally_consistent(condition):
    frame = np.random.default_rng(42).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    frames = np.stack([frame, frame])
    before = frames.copy()
    result = perturb(frames, condition)
    assert np.array_equal(frames, before)
    assert result.shape == frames.shape and result.dtype == np.uint8
    assert np.array_equal(result[0], result[1])
    assert np.array_equal(result, perturb(frames, condition))
    assert np.array_equal(result, before) == (condition == "clean")


def test_stress_rgb_preserved():
    frames = np.zeros((1, 24, 24, 3), dtype=np.uint8)
    frames[..., 0] = 200
    output = perturb(frames, "jpeg_35")
    assert output[..., 0].mean() > 180
    assert output[..., 2].mean() < 20


def test_paired_verdict_changes_and_fixed_threshold():
    clean = [{"source_id": "one", "threshold": .5, "score": .6, "label": 1, "prediction": 1}]
    changed = [{**clean[0], "score": .4, "prediction": 0}]
    result = paired_changes(clean, changed)
    assert result["verdict_flips"] == result["correct_to_wrong"] == 1
    assert result["wrong_to_correct"] == 0
    with pytest.raises(ValueError, match="Threshold"):
        paired_changes(clean, [{**changed[0], "threshold": .3}])
    with pytest.raises(ValueError, match="unique"):
        paired_changes(clean, changed * 2)


def test_invalid_stress_input():
    with pytest.raises(ValueError, match="uint8"):
        perturb(np.zeros((1, 2, 2, 3)), "clean")
    with pytest.raises(ValueError, match="Unknown"):
        perturb(np.zeros((1, 2, 2, 3), dtype=np.uint8), "unknown")


def test_calibration_warns_about_all_positive_and_discrete_recall():
    rows = [{"label": label, "score": .6} for label in [1, 1, 1, 0, 0, 0, 0]]
    result = calibration_diagnostics(rows, .5, .75)
    assert result["required_positive_hits"] == 3
    assert result["recall_resolution"] == pytest.approx(1 / 3)
    assert result["predicted_positive_sources"] == 7
    assert len(result["warnings"]) == 2
    with pytest.raises(ValueError, match="both classes"):
        calibration_diagnostics(rows[:3], .5, .75)
