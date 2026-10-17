from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2

from .contracts import DetectedEvent, EvidenceExtractionError, EvidenceFrame
from .video_reader import VideoReader


class EvidenceExtractor:
    def __init__(self, output_root: str | Path, *, jpeg_quality: int = 90) -> None:
        self.output_root = Path(output_root)
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self._run_dirs: dict[str, Path] = {}

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

        if count == 1:
            requested = [event.peak_time_sec]
        else:
            before = event.peak_time_sec - min(
                2.0, max(0.0, event.peak_time_sec - event.start_sec)
            )
            after = event.peak_time_sec + min(
                2.0, max(0.0, event.end_sec - event.peak_time_sec)
            )
            requested = [before, after]
            if count == 3:
                requested.append(event.peak_time_sec)
        duration = reader.metadata().duration_sec
        requested = sorted({min(duration, max(0.0, value)) for value in requested})
        frames = reader.sample_nearest(requested)

        evidence: list[EvidenceFrame] = []
        seen_actual: set[float] = set()
        for evidence_index, frame in enumerate(frames, start=1):
            rounded = round(frame.timestamp_sec, 6)
            if rounded in seen_actual:
                continue
            seen_actual.add(rounded)
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
