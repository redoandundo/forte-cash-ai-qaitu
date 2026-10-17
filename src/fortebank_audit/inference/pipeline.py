from __future__ import annotations

import math
import time
import uuid
from pathlib import Path

import numpy as np
import psutil

from .aggregation import aggregate_predictions
from .contracts import (
    ClipWindow,
    InferenceResult,
    PipelineConfig,
    PredictionError,
    VideoMetadata,
    WindowPrediction,
)
from .evidence import EvidenceExtractor
from .predictor import Predictor
from .video_reader import VideoReader
from .windowing import plan_windows


def score_video(
    video_path: str | Path,
    predictor: Predictor,
    config: PipelineConfig,
) -> tuple[VideoMetadata, list[WindowPrediction], float, float]:
    """Decode once, score all windows, and return timing/memory observations."""
    started = time.perf_counter()
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    reader = VideoReader(video_path)
    metadata = reader.metadata()
    preprocessing = getattr(predictor, "preprocessing", None)
    if preprocessing is not None and (
        config.num_frames != preprocessing.num_frames
        or not math.isclose(config.window_sec, preprocessing.window_sec, abs_tol=1e-9)
        or not math.isclose(config.stride_sec, preprocessing.stride_sec, abs_tol=1e-9)
    ):
        raise PredictionError("Pipeline window configuration does not match the model contract")
    estimated_windows = (
        1
        if metadata.duration_sec <= config.window_sec
        else int(math.floor((metadata.duration_sec - config.window_sec) / config.stride_sec + 1e-9)) + 1
    )
    if estimated_windows > config.max_windows:
        raise PredictionError("Video would exceed the configured inference window limit")
    plans = plan_windows(
        metadata.duration_sec,
        config.window_sec,
        config.stride_sec,
        config.num_frames,
        include_tail=config.include_tail_window,
    )
    if not plans:
        raise RuntimeError("No inference windows could be created")
    if len(plans) > config.max_windows:
        raise PredictionError("Video exceeds the configured inference window limit")

    unique_times = sorted({timestamp for plan in plans for timestamp in plan.sample_times_sec})
    sampled_frames = iter(reader.iter_sample_nearest(unique_times))
    target_index = 0
    frame_cache: dict[float, np.ndarray] = {}
    last_use: dict[float, int] = {}
    for plan in plans:
        for timestamp in plan.sample_times_sec:
            last_use[timestamp] = plan.index

    predictions: list[WindowPrediction] = []
    for offset in range(0, len(plans), config.batch_size):
        batch_plans = plans[offset : offset + config.batch_size]
        required = {timestamp for plan in batch_plans for timestamp in plan.sample_times_sec}
        while required - frame_cache.keys():
            if target_index >= len(unique_times):
                raise RuntimeError("Video sampling ended before all inference windows were filled")
            timestamp = unique_times[target_index]
            frame_cache[timestamp] = next(sampled_frames).image_rgb
            target_index += 1
        windows = [
            ClipWindow(
                plan.index,
                plan.start_sec,
                plan.end_sec,
                plan.center_sec,
                np.stack([frame_cache[timestamp] for timestamp in plan.sample_times_sec]),
            )
            for plan in batch_plans
        ]
        batch_predictions = predictor.predict_batch(windows)
        if len(batch_predictions) != len(windows):
            raise PredictionError("Predictor returned an incomplete batch")
        for window, prediction in zip(windows, batch_predictions, strict=True):
            if (
                prediction.window_index != window.index
                or not math.isclose(prediction.start_sec, window.start_sec, abs_tol=1e-9)
                or not math.isclose(prediction.end_sec, window.end_sec, abs_tol=1e-9)
                or not math.isclose(prediction.center_sec, window.center_sec, abs_tol=1e-9)
                or not math.isfinite(prediction.score)
                or not 0 <= prediction.score <= 1
            ):
                raise PredictionError("Predictor returned an invalid window result")
        predictions.extend(batch_predictions)
        last_batch_index = batch_plans[-1].index
        for timestamp in tuple(frame_cache):
            if last_use[timestamp] <= last_batch_index:
                del frame_cache[timestamp]
    processing_sec = time.perf_counter() - started
    rss_delta_mb = max(0.0, (process.memory_info().rss - initial_rss) / (1024 * 1024))
    return metadata, predictions, processing_sec, rss_delta_mb


def analyze_video(
    video_path: str | Path,
    predictor: Predictor,
    config: PipelineConfig,
    *,
    evidence_root: str | Path = "data/raw/evidence",
    run_id: str | None = None,
) -> InferenceResult:
    started = time.perf_counter()
    reader = VideoReader(video_path)
    metadata, predictions, scoring_sec, rss_delta_mb = score_video(video_path, predictor, config)

    events = aggregate_predictions(
        predictions,
        threshold=config.threshold,
        merge_gap_sec=config.merge_gap_sec,
        max_events=config.max_events,
    )
    actual_run_id = run_id or uuid.uuid4().hex
    extractor = EvidenceExtractor(evidence_root)
    event_payload = []
    for index, event in enumerate(events):
        evidence = extractor.extract(
            reader,
            event,
            event_index=index,
            run_id=actual_run_id,
            count=config.evidence_count,
        )
        event_payload.append(
            {
                "score": event.peak_score,
                "start_sec": event.start_sec,
                "end_sec": event.end_sec,
                "peak_time_sec": event.peak_time_sec,
                "supporting_windows": event.supporting_windows,
                "evidence": [
                    {
                        "timecode_sec": item.timecode_sec,
                        "jpeg_path": item.jpeg_path.relative_to(extractor.output_root.resolve()).as_posix(),
                    }
                    for item in evidence
                ],
            }
        )

    processing_sec = time.perf_counter() - started
    peak_score = max((prediction.score for prediction in predictions), default=0.0)
    peak_time = min(
        (prediction.center_sec for prediction in predictions if prediction.score == peak_score),
        default=None,
    )
    return {
        "verdict": "possible_violation" if events else "no_violation",
        "score": peak_score,
        "timecode_sec": peak_time if events else None,
        "events": event_payload,
        "runtime": {
            "duration_sec": metadata.duration_sec,
            "processing_sec": processing_sec,
            "realtime_factor": processing_sec / metadata.duration_sec,
            "device": predictor.device,
            "model_version": predictor.model_version,
            "windows": len(predictions),
            "scoring_sec": scoring_sec,
            "rss_delta_mb": rss_delta_mb,
        },
    }
