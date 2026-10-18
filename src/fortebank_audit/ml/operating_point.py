"""Audit whether calibration supports both recall and false-alert constraints."""
from __future__ import annotations

import math

import numpy as np

from .metrics import events_from_scores


def calibrate_alert_budget(rows, stride=2., min_recall=.75, max_false_events_per_minute=.6):
    """Return no threshold when constraints conflict; never silently relax them.

    Input must be from a separate calibration split of one checkpoint. This
    function audits an operating point, not an estimate of external accuracy.
    """
    if (not rows or not math.isfinite(stride) or stride <= 0 or not 0 <= min_recall <= 1
            or not math.isfinite(max_false_events_per_minute) or max_false_events_per_minute < 0):
        raise ValueError("Invalid operating point constraints")
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Calibration requires both classes")
    all_scores = []
    for row in rows:
        scores = np.asarray(row["window_scores"], dtype=float)
        starts, ends = np.asarray(row["starts"]), np.asarray(row["ends"])
        duration = row["duration_sec"]
        if (scores.ndim != 1 or not len(scores) or starts.shape != scores.shape or ends.shape != scores.shape
                or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()
                or not math.isfinite(duration) or duration <= 0 or not np.isfinite(starts).all()
                or not np.isfinite(ends).all() or (starts < 0).any() or (ends <= starts).any()
                or (ends > duration + 1e-6).any() or (np.diff(starts) <= 0).any()):
            raise ValueError("Invalid calibration timeline")
        all_scores.extend(scores)
    unique = np.unique(all_scores)
    # Every decision/event grouping change occurs at a window score, not only a video maximum.
    candidates = np.unique(np.r_[0., unique, (unique[:-1] + unique[1:]) / 2, 1.])
    negatives = sum(row["label"] == 0 for row in rows)
    positives = len(rows) - negatives
    negative_minutes = sum(row["duration_sec"] for row in rows if not row["label"]) / 60
    feasible = []
    best_recall_under_budget = None
    for threshold in candidates:
        tp = fp = false_events = 0
        for row in rows:
            events = events_from_scores(row["starts"], row["ends"], row["window_scores"], threshold, stride)
            if row["label"]:
                tp += bool(events)
            else:
                fp += bool(events)
                false_events += len(events)
        recall = tp / positives
        rate = false_events / negative_minutes
        if rate > max_false_events_per_minute + 1e-12:
            continue
        best_recall_under_budget = max(recall, best_recall_under_budget or 0.)
        if recall < min_recall:
            continue
        fn, tn = positives - tp, negatives - fp
        f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
        feasible.append({"threshold": float(threshold), "recall": recall,
                         "precision": tp / (tp + fp) if tp + fp else 0.,
                         "macro_f1": (f1_pos + f1_neg) / 2,
                         "false_positive_events_per_minute": rate,
                         "min_score_margin": float(np.min(np.abs(unique - threshold)))})
    selected = max(feasible, key=lambda row: (row["macro_f1"], row["precision"], row["recall"],
                                              row["min_score_margin"], row["threshold"])) if feasible else None
    return {"feasible": selected is not None, "selected": selected,
            "reason": "constraints_met_on_calibration_only" if selected else "no_threshold_satisfies_both_constraints",
            "best_recall_under_false_alert_budget": best_recall_under_budget,
            "min_recall": min_recall, "max_false_events_per_minute": max_false_events_per_minute,
            "positive_sources": positives, "negative_sources": negatives,
            "negative_minutes": negative_minutes, "candidates_evaluated": len(candidates)}
