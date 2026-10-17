from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(labels, scores, threshold: float) -> dict:
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if len(labels) == 0 or labels.shape != scores.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid predictions")
    predicted = scores >= threshold
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "macro_f1": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)) if set(labels) == {0, 1} else None,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def calibrate_threshold(labels, scores, min_recall: float = 0.75) -> dict:
    if not 0 <= min_recall <= 1:
        raise ValueError("Invalid recall constraint")
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if set(labels.tolist()) != {0, 1} or labels.shape != scores.shape:
        raise ValueError("Calibration requires both classes and matching scores")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid calibration scores")
    unique = np.unique(scores)
    candidates = np.unique(np.r_[0.0, unique, (unique[:-1] + unique[1:]) / 2, 1.0])
    choices = []
    for threshold in candidates:
        result = binary_metrics(labels, scores, float(threshold))
        if result["recall"] >= min_recall:
            choices.append({**result, "threshold": float(threshold), "min_score_margin": float(np.min(np.abs(scores - threshold)))})
    if not choices:
        raise ValueError("No threshold satisfies the recall constraint")
    return max(choices, key=lambda row: (row["macro_f1"], row["precision"], row["recall"], row["min_score_margin"], row["threshold"]))


def smooth_scores(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Invalid window scores")
    if len(values) < 3:
        return values.copy()
    return np.convolve(np.pad(values, (1, 1), mode="edge"), [0.25, 0.5, 0.25], mode="valid")


def events_from_scores(starts, ends, scores, threshold: float, stride: float, max_events: int = 3) -> list[dict]:
    centers = (np.asarray(starts) + np.asarray(ends)) / 2
    active = np.flatnonzero(np.asarray(scores) >= threshold)
    groups = []
    for index in active:
        if not groups or centers[index] - centers[groups[-1][-1]] > stride + 1e-6:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    events = []
    for group in groups:
        peak = max(group, key=lambda idx: (scores[idx], -centers[idx]))
        events.append({"start_sec": float(starts[group[0]]), "end_sec": float(ends[group[-1]]), "peak_time_sec": float(centers[peak]), "peak_score": float(scores[peak])})
    return sorted(sorted(events, key=lambda event: -event["peak_score"])[:max_events], key=lambda event: event["peak_time_sec"])


def match_events(predicted, truth, tolerance: float = 5.0) -> dict:
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if not len(predicted) or not len(truth):
        return {"tp": 0, "fp": len(predicted), "fn": len(truth), "errors_sec": []}
    distances = np.abs(predicted[:, None] - truth[None, :])
    # Maximise matches within tolerance before minimising their distance.
    penalty = (max(len(predicted), len(truth)) + 1) * (float(distances.max()) + 1)
    rows, cols = linear_sum_assignment(distances + (distances > tolerance) * penalty)
    errors = distances[rows, cols]
    hits = int(np.sum(errors <= tolerance))
    # Keep distant assignments in MAE so MAE<=5 is not tautological.
    return {"tp": hits, "fp": len(predicted) - hits, "fn": len(truth) - hits, "errors_sec": errors.tolist()}


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation sources")
    labels = [row["label"] for row in rows]
    decisions = [row["prediction"] for row in rows]
    result = binary_metrics(labels, decisions, 0.5)
    result.pop("auroc")  # Cross-model raw scores are not comparable across outer folds.
    minutes = sum(row["duration_sec"] for row in rows if not row["label"]) / 60
    false_videos = sum(not row["label"] and row["prediction"] for row in rows)
    false_events = sum(len(row["events"]) for row in rows if not row["label"])
    counts = {name: sum(row["event_match"][name] for row in rows) for name in ("tp", "fp", "fn")}
    errors = [error for row in rows for error in row["event_match"]["errors_sec"]]
    return {
        **result, "source_count": len(rows),
        "false_positive_videos_per_minute": false_videos / minutes if minutes else None,
        "false_positive_events_per_minute": false_events / minutes if minutes else None,
        "event_tp": counts["tp"], "event_fp": counts["fp"], "event_fn": counts["fn"],
        "event_recall_within_5_sec": counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None,
        "timestamp_mae_sec": float(np.mean(errors)) if errors else None,
        "timestamp_metric": "one_to_one_emitted_peaks_to_annotated_points",
    }
