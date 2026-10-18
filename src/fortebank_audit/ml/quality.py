"""Evaluation details kept separate from the calibrated scoring pipeline."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .metrics import events_from_scores, match_events, summarize_rows


def evaluated_rows(rows, threshold, stride):
    result = []
    for source in rows:
        row = dict(source)
        row["events"] = events_from_scores(row["starts"], row["ends"], row["window_scores"], threshold, stride)
        row["prediction"] = int(bool(row["events"]))
        row["event_match"] = match_events([event["peak_time_sec"] for event in row["events"]], row["truth_points"])
        row["threshold"] = threshold
        result.append(row)
    return result


def detailed_metrics(rows, sources):
    """All original-point metrics plus explicit ambiguity/coverage diagnostics.

    Two conflicting timestamps on duplicate files are not confirmed two events.
    Keep legacy metrics comparable, and report an additional unambiguous subset.
    """
    metrics = summarize_rows(rows)
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    metrics["positive_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    by_id = {source["source_id"]: source for source in sources}
    ambiguous = {source["source_id"] for source in sources
                 if len(source.get("video_ids", [])) > 1 and len(source.get("events_sec", [])) > 1
                 and source.get("review_status") != "confirmed"}
    unambiguous = [row for row in rows if row["source_id"] not in ambiguous]
    metrics["excluded_ambiguous_temporal_sources"] = len(rows) - len(unambiguous)
    metrics["unambiguous_point_metrics"] = summarize_rows(unambiguous) if unambiguous else None
    metrics["annotation_scope"] = "point_reference_metrics_not_verified_action_intervals"
    metrics["complete_positive_annotation_sources"] = sum(bool(source.get("events_complete")) for source in sources if source["label"])
    groups = {}
    for row in rows:
        groups.setdefault(row["scenario_group"], []).append(row)
    private_groups = {group: summarize_rows(values) for group, values in groups.items()}
    # Aggregate group types may be published; source/group identities stay local.
    cohorts = {}
    for label in ("paired", "single_source"):
        subset = [row for row in rows if ("paired" if len(groups[row["scenario_group"]]) > 1 else "single_source") == label]
        if subset:
            cohorts[label] = summarize_rows(subset)
    errors = Counter()
    for row in rows:
        wrong = row["label"] != row["prediction"]
        if wrong:
            errors["false_negative" if row["label"] else "false_positive_video"] += 1
        if not row["label"]:
            for event in row["events"]:
                if event["peak_time_sec"] <= 3.001 or event["peak_time_sec"] >= row["duration_sec"] - 3.001:
                    errors["false_positive_event_at_video_boundary"] += 1
        if row["label"] and row["event_match"]["fn"]:
            errors["positive_source_with_missed_or_mislocalized_point"] += 1
        if "raw_scores" in row:
            raw = np.asarray(row["raw_scores"])
            if row["label"] and not row["prediction"] and float(raw.max()) >= row["threshold"]:
                errors["false_negative_crosses_threshold_before_smoothing"] += 1
            if wrong and np.ptp(raw) < .02:
                errors["classification_error_with_flat_window_scores_range_lt_0_02"] += 1
            if wrong and abs(row["score"] - row["threshold"]) < .01:
                errors["classification_error_with_threshold_margin_lt_0_01"] += 1
        if wrong and by_id[row["source_id"]].get("annotation_confidence") != "high":
            errors["classification_error_on_uncertain_annotation"] += 1
    return {"metrics": metrics, "scenario_cohorts": cohorts, "error_counts": dict(errors),
            "private_scenario_groups": private_groups}
