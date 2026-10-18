from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np

from .contracts import DetectedEvent, EvidenceExtractionError, EvidenceFrame, VideoFrame
from .video_reader import VideoReader


class EvidenceExtractor:
    def __init__(
        self,
        output_root: str | Path,
        *,
        jpeg_quality: int = 90,
        min_visual_difference: float = 2.0,
    ) -> None:
        self.output_root = Path(output_root)
        self.jpeg_quality = int(jpeg_quality)
        self.min_visual_difference = float(min_visual_difference)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if not 0 <= self.min_visual_difference <= 255:
            raise ValueError("min_visual_difference must be in [0, 255]")
        self._run_dirs: dict[str, Path] = {}

    @staticmethod
    def _candidate_times(
        event: DetectedEvent, duration_sec: float, count: int
    ) -> list[float]:
        peak = min(duration_sec, max(0.0, event.peak_time_sec))
        if count == 1:
            return [peak]
        start = min(duration_sec, max(0.0, min(event.start_sec, event.end_sec)))
        end = min(duration_sec, max(start, max(event.start_sec, event.end_sec)))
        if end <= start:
            return [peak]
        candidate_count = max(7, count * 3)
        step = (end - start) / (candidate_count - 1)
        requested = {start + step * index for index in range(candidate_count)}
        requested.add(peak)
        return sorted(requested)

    @staticmethod
    def _visual_fingerprint(frame: VideoFrame) -> np.ndarray:
        gray = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )

    def _visually_distinct(
        self,
        frame: VideoFrame,
        selected: list[tuple[VideoFrame, np.ndarray]],
    ) -> bool:
        candidate = self._visual_fingerprint(frame)
        return all(
            float(np.mean(np.abs(candidate - fingerprint)))
            >= self.min_visual_difference
            for _, fingerprint in selected
        )

    def _select_unique_frames(
        self,
        frames: list[VideoFrame],
        *,
        peak_time_sec: float,
        count: int,
    ) -> list[VideoFrame]:
        by_timestamp: dict[float, VideoFrame] = {}
        for frame in frames:
            by_timestamp.setdefault(round(frame.timestamp_sec, 6), frame)
        candidates = list(by_timestamp.values())
        if not candidates:
            return []

        first = min(candidates, key=lambda item: abs(item.timestamp_sec - peak_time_sec))
        selected: list[tuple[VideoFrame, np.ndarray]] = [
            (first, self._visual_fingerprint(first))
        ]
        remaining = [frame for frame in candidates if frame is not first]
        while remaining and len(selected) < count:
            distinct = [
                frame for frame in remaining if self._visually_distinct(frame, selected)
            ]
            if not distinct:
                break
            next_frame = max(
                distinct,
                key=lambda item: (
                    min(
                        abs(item.timestamp_sec - chosen.timestamp_sec)
                        for chosen, _ in selected
                    ),
                    abs(item.timestamp_sec - peak_time_sec),
                    -item.timestamp_sec,
                ),
            )
            selected.append((next_frame, self._visual_fingerprint(next_frame)))
            remaining.remove(next_frame)
        return sorted((frame for frame, _ in selected), key=lambda item: item.timestamp_sec)

    def _prepare_run_dir(self, run_id: str) -> Path:
        if run_id in self._run_dirs:
            return self._run_dirs[run_id]
        display = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)[:60] or "run"
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
        run_dir = (self.output_root / f"{display}_{digest}").resolve()
        root = self.output_root.resolve()
        if root != run_dir and root not in run_dir.parents:
            raise EvidenceExtractionError("Unsafe evidence output path")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise EvidenceExtractionError("Evidence run directory already exists") from exc
        self._run_dirs[run_id] = run_dir
        return run_dir

    def extract(
        self,
        reader: VideoReader,
        event: DetectedEvent,
        *,
        event_index: int,
        run_id: str,
        count: int,
    ) -> list[EvidenceFrame]:
        run_dir = self._prepare_run_dir(run_id)

        duration = reader.metadata().duration_sec
        requested = self._candidate_times(event, duration, count)
        sampled_frames = reader.sample_nearest(requested)
        frames = self._select_unique_frames(
            sampled_frames,
            peak_time_sec=event.peak_time_sec,
            count=count,
        )
        if not frames:
            raise EvidenceExtractionError("Could not extract JPEG evidence")

        evidence: list[EvidenceFrame] = []
        for evidence_index, frame in enumerate(frames, start=1):
            path = run_dir / f"event_{event_index:03d}_{evidence_index:02d}.jpg"
            bgr = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2BGR)
            try:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    raise EvidenceExtractionError("Could not encode JPEG evidence")
                path.write_bytes(encoded.tobytes())
            except (cv2.error, OSError) as exc:
                raise EvidenceExtractionError("Could not save JPEG evidence") from exc
            evidence.append(EvidenceFrame(frame.timestamp_sec, path))
        return evidence

    def extract_temporal_context(
        self,
        reader: VideoReader,
        event: DetectedEvent,
        *,
        event_index: int,
        run_id: str,
    ) -> list[dict[str, object]]:
        """Create 1-5 chronological, visually unique frames for semantic verification."""
        run_dir = self._prepare_run_dir(run_id)
        duration = reader.metadata().duration_sec
        candidates = [
            ("before", max(0.0, event.start_sec - 2.0)),
            ("start", event.start_sec),
            ("peak", event.peak_time_sec),
            ("end", event.end_sec),
            ("after", min(duration, event.end_sec + 2.0)),
        ]
        # At the beginning/end of a video, prefer the outer temporal role: the frame is
        # evidence that no earlier/later context exists, not a duplicate event frame.
        priority = {"before": 0, "after": 1, "peak": 2, "start": 3, "end": 4}
        by_time: dict[float, tuple[str, float]] = {}
        for role, timestamp in candidates:
            timestamp = min(duration, max(0.0, timestamp))
            key = round(timestamp, 6)
            current = by_time.get(key)
            if current is None or priority[role] < priority[current[0]]:
                by_time[key] = (role, timestamp)
        ordered = sorted(by_time.values(), key=lambda item: item[1])
        frames = reader.sample_nearest([timestamp for _, timestamp in ordered])

        output: list[dict[str, object]] = []
        fingerprints: list[object] = []
        for role_and_time, frame in zip(ordered, frames, strict=True):
            role, _ = role_and_time
            gray = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2GRAY)
            fingerprint = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
            # Only collapse near-pixel-identical frames. Small hands/documents are material here;
            # a broad perceptual threshold can erase the very transition Qwen must inspect.
            if any(float(cv2.norm(fingerprint, prior, cv2.NORM_L1)) / fingerprint.size < 0.05
                   for prior in fingerprints):
                continue
            fingerprints.append(fingerprint)
            path = run_dir / f"event_{event_index:03d}_context_{len(output) + 1:02d}.jpg"
            bgr = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2BGR)
            try:
                ok, encoded = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    raise EvidenceExtractionError("Could not encode temporal JPEG evidence")
                path.write_bytes(encoded.tobytes())
            except (cv2.error, OSError) as exc:
                raise EvidenceExtractionError("Could not save temporal JPEG evidence") from exc
            output.append({"role": role, "timestamp_sec": frame.timestamp_sec, "jpeg_path": path})
        if not output:
            raise EvidenceExtractionError("No temporal context frame could be extracted")
        return output
