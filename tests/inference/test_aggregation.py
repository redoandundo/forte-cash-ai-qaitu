from __future__ import annotations

import pytest

from fortebank_audit.inference.aggregation import (
    aggregate_predictions,
    postprocess_predictions,
    smooth_predictions,
)
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
    assert merged.supporting_window_indices == (0, 1)


def test_invalid_score_is_rejected() -> None:
    with pytest.raises(ValueError):
        aggregate_predictions([prediction(0, 0, 1.1)], threshold=0.5, merge_gap_sec=1, max_events=5)


def test_accepted_three_window_smoothing_uses_edge_padding() -> None:
    smoothed = smooth_predictions(
        [prediction(0, 0, 0.2), prediction(1, 2, 0.8), prediction(2, 4, 0.4)]
    )
    assert [item.score for item in smoothed] == pytest.approx([0.35, 0.55, 0.5])
    assert [item.center_sec for item in smoothed] == [4, 6, 8]


def test_new_ml_bundle_keeps_short_sequences_unsmoothed() -> None:
    values = [prediction(0, 0, 0.2), prediction(1, 2, 0.8)]
    scored = postprocess_predictions(
        values,
        aggregation="three_window_weighted_smoothing_then_peak",
        short_sequence_rule="identity_if_fewer_than_three",
    )
    assert [item.score for item in scored] == [0.2, 0.8]


def test_new_ml_bundle_merges_only_adjacent_centers() -> None:
    events = aggregate_predictions(
        [prediction(0, 0, 0.8), prediction(1, 4, 0.9)],
        threshold=0.5,
        merge_gap_sec=99,
        max_events=3,
        event_merge="adjacent_centers_within_stride",
        stride_sec=2,
    )
    assert len(events) == 2
