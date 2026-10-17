from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import DetectedEvent, WindowPrediction


def aggregate_predictions(
    predictions: Sequence[WindowPrediction],
    *,
    threshold: float,
    merge_gap_sec: float,
    max_events: int,
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
    groups: list[list[WindowPrediction]] = []
    for prediction in positives:
        if not groups or prediction.start_sec > max(item.end_sec for item in groups[-1]) + merge_gap_sec:
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
    events.sort(key=lambda item: (-item.peak_score, item.peak_time_sec))
    return events[:max_events]
