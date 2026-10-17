import pytest

from fortebank_audit.ml.quality import detailed_metrics, evaluated_rows


def test_ambiguous_duplicate_points_are_not_silently_confirmed_events():
    sources = [{"source_id": "a", "label": 1, "video_ids": ["alias-a", "alias-b"], "events_sec": [3, 8]},
               {"source_id": "b", "label": 1, "events_sec": [3]},
               {"source_id": "c", "label": 0, "events_sec": []}]
    raw = [{"source_id": source["source_id"], "scenario_group": source["source_id"],
            "label": source["label"], "duration_sec": 10, "starts": [0.], "ends": [6.],
            "window_scores": [.8 if source["label"] else .2], "score": .8 if source["label"] else .2,
            "truth_points": source["events_sec"]} for source in sources]
    report = detailed_metrics(evaluated_rows(raw, .5, 2), sources)
    assert report["metrics"]["positive_f1"] == 1.
    assert report["metrics"]["source_count"] == 3
    assert report["metrics"]["excluded_ambiguous_temporal_sources"] == 1
    assert report["metrics"]["event_recall_within_5_sec"] == pytest.approx(2 / 3)
    assert report["metrics"]["unambiguous_point_metrics"]["event_recall_within_5_sec"] == 1.
    assert len(report["private_scenario_groups"]) == 3


def test_error_analysis_reports_observation_without_inventing_visual_cause():
    sources = [{"source_id": "a", "label": 1, "annotation_confidence": "low", "events_sec": [3]}]
    raw = [{"source_id": "a", "scenario_group": "a", "label": 1, "duration_sec": 10,
            "starts": [0., 2., 4.], "ends": [6., 8., 10.], "window_scores": [.49, .48, .49],
            "raw_scores": [.51, .43, .51], "score": .49, "truth_points": [3]}]
    report = detailed_metrics(evaluated_rows(raw, .5, 2), sources)
    assert report["error_counts"]["false_negative_crosses_threshold_before_smoothing"] == 1
    assert report["error_counts"]["classification_error_on_uncertain_annotation"] == 1
    assert "occlusion" not in report["error_counts"]
