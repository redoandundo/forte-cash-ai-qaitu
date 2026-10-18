import pytest

from fortebank_audit.ml.operating_point import calibrate_alert_budget


def row(label, scores):
    return {"label": label, "window_scores": scores, "starts": list(range(0, 2 * len(scores), 2)),
            "ends": list(range(2, 2 * len(scores) + 2, 2)), "duration_sec": 60.}


def test_separable_calibration_meets_both_constraints():
    result = calibrate_alert_budget([row(0, [.1, .2]), row(1, [.8, .9])], max_false_events_per_minute=0)
    assert result["feasible"]
    assert .2 < result["selected"]["threshold"] <= .9
    assert result["selected"]["recall"] == 1
    assert result["selected"]["false_positive_events_per_minute"] == 0


def test_inseparable_calibration_does_not_silently_emit_all_normal_threshold():
    result = calibrate_alert_budget([row(0, [.8]), row(1, [.7])], max_false_events_per_minute=0)
    assert not result["feasible"] and result["selected"] is None
    assert result["best_recall_under_false_alert_budget"] == 0


def test_budget_counts_separate_events_not_only_positive_videos():
    result = calibrate_alert_budget([row(0, [.9, .1, .9]), row(1, [.8])], max_false_events_per_minute=1)
    # At thresholds <= .1 the normal video has one merged event; above .1 it has two.
    assert result["feasible"] and result["selected"]["threshold"] <= .1
    assert result["selected"]["false_positive_events_per_minute"] == 1


@pytest.mark.parametrize("fault", ["nan", "empty", "time", "one_class"])
def test_bad_calibration_is_rejected(fault):
    rows = [row(0, [.2]), row(1, [.8])]
    if fault == "nan":
        rows[0]["window_scores"] = [float("nan")]
    elif fault == "empty":
        rows[0]["window_scores"] = []
    elif fault == "time":
        rows[0]["ends"] = [100.]
    else:
        rows[1]["label"] = 0
    with pytest.raises(ValueError):
        calibrate_alert_budget(rows)
