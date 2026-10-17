from __future__ import annotations

import pytest

from fortebank_audit.inference.aggregation import aggregate_predictions
from fortebank_audit.inference.contracts import WindowPrediction


def prediction(index: int, start: float, score: float) -> WindowPrediction:
    return WindowPrediction(index, start, start + 8, start + 4, score)


def test_below_threshold_has_no_event() -> None:
    assert aggregate_predictions([prediction(0, 0, 0.4)], threshold=0.5, merge_gap_sec=1, max_events=5) == []


def test_overlapping_windows_merge_and_earliest_peak_wins_tie() -> None:
    events = aggregate_predictions(
        [prediction(0, 0, 0.8), prediction(1, 2, 0.8), prediction(2, 20, 0.9)],
        threshold=0.5,
        merge_gap_sec=1,
        max_events=5,
    )
    assert len(events) == 2
    merged = next(item for item in events if item.start_sec == 0)
    assert merged.end_sec == 10
    assert merged.peak_time_sec == 4
    assert merged.supporting_windows == 2


def test_invalid_score_is_rejected() -> None:
    with pytest.raises(ValueError):
        aggregate_predictions([prediction(0, 0, 1.1)], threshold=0.5, merge_gap_sec=1, max_events=5)
