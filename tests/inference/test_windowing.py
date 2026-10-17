from __future__ import annotations

import pytest

from fortebank_audit.inference.windowing import plan_windows
from fortebank_audit.inference.contracts import PipelineConfig


def test_empty_video_has_no_windows() -> None:
    assert plan_windows(0, 8, 2, 4) == []


def test_short_video_has_one_bounded_window() -> None:
    windows = plan_windows(3, 8, 2, 4)
    assert len(windows) == 1
    assert windows[0].start_sec == 0
    assert windows[0].end_sec == 3
    assert windows[0].sample_times_sec == pytest.approx((0, 0.75, 1.5, 2.25))


def test_exact_window_has_expected_samples() -> None:
    windows = plan_windows(8, 8, 2, 4)
    assert windows[0].sample_times_sec == pytest.approx((0, 2, 4, 6))


def test_tail_is_opt_in_to_preserve_threshold_calibration() -> None:
    calibrated = plan_windows(11, 8, 2, 4)
    with_tail = plan_windows(11, 8, 2, 4, include_tail=True)
    assert [item.start_sec for item in calibrated] == [0, 2]
    assert [item.start_sec for item in with_tail] == [0, 2, 3]


def test_window_start_never_overshoots_duration_boundary() -> None:
    assert [item.start_sec for item in plan_windows(9.999, 8, 2, 4)] == [0]


def test_pipeline_config_rejects_blind_gaps() -> None:
    with pytest.raises(ValueError, match="stride_sec cannot exceed"):
        PipelineConfig(window_sec=2, stride_sec=3, num_frames=4, threshold=0.5)
