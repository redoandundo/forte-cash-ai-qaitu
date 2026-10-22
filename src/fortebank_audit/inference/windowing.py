from __future__ import annotations

import math

import numpy as np

from .contracts import WindowPlan


def plan_windows(
    duration_sec: float,
    window_sec: float,
    stride_sec: float,
    num_frames: int,
    *,
    include_tail: bool = False,
) -> list[WindowPlan]:
    """Plan deterministic windows without decoding frames.

    ``include_tail=False`` preserves the calibrated ML baseline. Enabling it adds
    a final window ending exactly at the video boundary and requires threshold
    re-calibration before production use.
    """
    values = (duration_sec, window_sec, stride_sec)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Window parameters must be finite")
    if duration_sec <= 0:
        return []
    if window_sec <= 0 or stride_sec <= 0 or num_frames <= 0:
        raise ValueError("Window parameters must be positive")

    if duration_sec <= window_sec:
        starts = [0.0]
    else:
        maximum_start = duration_sec - window_sec
        step_count = int(math.floor(maximum_start / stride_sec + 1e-9))
        starts = [index * stride_sec for index in range(step_count + 1)]
        starts = [start for start in starts if start <= maximum_start + 1e-9]
        if include_tail and not math.isclose(starts[-1], maximum_start, abs_tol=1e-6):
            starts.append(maximum_start)

    plans: list[WindowPlan] = []
    for index, start in enumerate(starts):
        end = min(duration_sec, start + window_sec)
        sample_times = np.linspace(start, end, num_frames, endpoint=False, dtype=float)
        plans.append(
            WindowPlan(
                index=index,
                start_sec=float(start),
                end_sec=float(end),
                center_sec=float((start + end) / 2),
                sample_times_sec=tuple(float(value) for value in sample_times),
            )
        )
    return plans
