from fortebank_audit.ml import evaluation
from fortebank_audit.ml.preprocessing import VideoSpec


def test_calibration_exposes_degenerate_all_positive_threshold(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    rows = [{"source_id": str(i), "sha256": str(i), "scenario_group": str(i),
             "label": label, "score": score}
            for i, (label, score) in enumerate([(1, .2), (1, .4), (1, .9), (0, .5), (0, .6)])]
    monkeypatch.setattr(evaluation, "score_sources", lambda *args: (
        rows, {"model": "test", "provenance": {}}, VideoSpec()))
    result = evaluation.calibration_report(checkpoint, [], tmp_path)
    assert result["diagnostics"]["required_positive_hits"] == 3
    assert result["diagnostics"]["predicted_positive_sources"] == 5
    assert "all_calibration_videos_positive_no_negative_discrimination" in result["diagnostics"]["warnings"]
    assert "small_calibration_support_threshold_is_uncertain" in result["diagnostics"]["warnings"]
    assert result["scoring_backend"]["device"] in {"cpu", "cuda"}
