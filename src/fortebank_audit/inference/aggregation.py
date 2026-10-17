from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import DetectedEvent, WindowPrediction


ACCEPTED_AGGREGATION = "three_window_weighted_smoothing_then_peak"


def smooth_predictions(
    predictions: Sequence[WindowPrediction],
    *,
    weights: tuple[float, float, float] = (0.25, 0.5, 0.25),
) -> list[WindowPrediction]:
    """Apply the accepted three-window smoother with edge padding."""
    if len(weights) != 3 or any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("Smoothing weights must be three finite non-negative values")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
        raise ValueError("Smoothing weights must sum to one")
    if not predictions:
        return []
    ordered = sorted(predictions, key=lambda item: item.window_index)
    for prediction in ordered:
        if not math.isfinite(prediction.score) or not 0 <= prediction.score <= 1:
            raise ValueError("Prediction score must be finite and in [0, 1]")
    scores = [item.score for item in ordered]
    padded = [scores[0], *scores, scores[-1]]
    return [
        WindowPrediction(
            item.window_index,
            item.start_sec,
            item.end_sec,
            item.center_sec,
            weights[0] * padded[index]
            + weights[1] * padded[index + 1]
            + weights[2] * padded[index + 2],
        )
        for index, item in enumerate(ordered)
    ]


def postprocess_predictions(
    predictions: Sequence[WindowPrediction],
    *,
    aggregation: str,
    short_sequence_rule: str = "edge_pad",
) -> list[WindowPrediction]:
    if aggregation == ACCEPTED_AGGREGATION:
        if short_sequence_rule == "identity_if_fewer_than_three" and len(predictions) < 3:
            return list(predictions)
        if short_sequence_rule not in {"edge_pad", "identity_if_fewer_than_three"}:
            raise ValueError(f"Unsupported short-sequence rule: {short_sequence_rule}")
        return smooth_predictions(predictions)
    if aggregation == "none":
        return list(predictions)
    raise ValueError(f"Unsupported aggregation strategy: {aggregation}")


def aggregate_predictions(
    predictions: Sequence[WindowPrediction],
    *,
    threshold: float,
    merge_gap_sec: float,
    max_events: int,
    event_merge: str = "overlapping_windows",
    stride_sec: float | None = None,
) -> list[DetectedEvent]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if merge_gap_sec < 0 or max_events <= 0:
        raise ValueError("Invalid aggregation configuration")
    for prediction in predictions:
        if not math.isfinite(prediction.score) or not 0 <= prediction.score <= 1:
            raise ValueError("Prediction score must be finite and in [0, 1]")
    positives = sorted(
        (prediction for prediction in predictions if prediction.score >= threshold),
        key=lambda item: (item.start_sec, item.window_index),
    )
    if event_merge == "adjacent_centers_within_stride" and (stride_sec is None or stride_sec <= 0):
        raise ValueError("Positive stride is required for adjacent-center merging")
    if event_merge not in {"overlapping_windows", "adjacent_centers_within_stride"}:
        raise ValueError(f"Unsupported event merge rule: {event_merge}")
    groups: list[list[WindowPrediction]] = []
    for prediction in positives:
        if not groups:
            groups.append([prediction])
            continue
        if event_merge == "adjacent_centers_within_stride":
            separated = prediction.center_sec - groups[-1][-1].center_sec > float(stride_sec) + 1e-6
        else:
            separated = prediction.start_sec > max(item.end_sec for item in groups[-1]) + merge_gap_sec
        if separated:
            groups.append([prediction])
        else:
            groups[-1].append(prediction)

    events = []
    for group in groups:
        peak = min(group, key=lambda item: (-item.score, item.center_sec, item.window_index))
        events.append(
            DetectedEvent(
                start_sec=min(item.start_sec for item in group),
                end_sec=max(item.end_sec for item in group),
                peak_time_sec=peak.center_sec,
                peak_score=peak.score,
                supporting_windows=len(group),
            )
        )
    selected = sorted(events, key=lambda item: (-item.peak_score, item.peak_time_sec))[:max_events]
    if event_merge == "adjacent_centers_within_stride":
        selected.sort(key=lambda item: item.peak_time_sec)
    return selected
