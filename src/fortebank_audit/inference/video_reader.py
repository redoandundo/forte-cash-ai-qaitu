from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Iterator

import av
import numpy as np

from .contracts import UnsupportedVideoError, VideoFrame, VideoMetadata, VideoOpenError


class VideoReader:
    """Sequential PyAV reader that reports container-derived timestamps."""

    SUPPORTED_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise VideoOpenError(f"Video file does not exist: {self.path.name}")
        if self.path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise UnsupportedVideoError(f"Unsupported video extension: {self.path.suffix}")

    def _open(self) -> av.container.InputContainer:
        try:
            return av.open(str(self.path), mode="r")
        except (av.error.FFmpegError, OSError) as exc:
            raise VideoOpenError(f"Could not open video: {self.path.name}") from exc

    def metadata(self) -> VideoMetadata:
        with self._open() as container:
            if not container.streams.video:
                raise VideoOpenError(f"No video stream found: {self.path.name}")
            stream = container.streams.video[0]
            duration = 0.0
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            frame_count = int(stream.frames) if stream.frames else None
            if duration <= 0 and frame_count and fps > 0:
                duration = frame_count / fps
            if duration <= 0:
                raise VideoOpenError(f"Video duration is unavailable: {self.path.name}")
            return VideoMetadata(
                path=self.path,
                duration_sec=duration,
                width=int(stream.codec_context.width or 0),
                height=int(stream.codec_context.height or 0),
                nominal_fps=fps,
                frame_count=frame_count,
                codec=str(stream.codec_context.name or "unknown"),
                has_audio=bool(container.streams.audio),
            )

    def iter_frames(self) -> Iterator[VideoFrame]:
        with self._open() as container:
            if not container.streams.video:
                raise VideoOpenError(f"No video stream found: {self.path.name}")
            stream = container.streams.video[0]
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            previous = -math.inf
            decoded = 0
            try:
                for frame in container.decode(stream):
                    if frame.pts is not None and frame.time_base is not None:
                        timestamp = float(frame.pts * frame.time_base) - origin
                    elif frame.time is not None:
                        timestamp = float(frame.time) - origin
                    elif fps > 0:
                        timestamp = decoded / fps
                    else:
                        raise VideoOpenError(f"Frame timestamp is unavailable: {self.path.name}")
                    timestamp = max(0.0, timestamp)
                    if timestamp + 1e-9 < previous:
                        raise VideoOpenError(f"Non-monotonic timestamps: {self.path.name}")
                    previous = timestamp
                    decoded += 1
                    yield VideoFrame(timestamp, frame.to_ndarray(format="rgb24"))
            except (av.error.FFmpegError, OSError) as exc:
                raise VideoOpenError(f"Could not decode video: {self.path.name}") from exc
            if decoded == 0:
                raise VideoOpenError(f"Video contains no decodable frames: {self.path.name}")

    def iter_sample_nearest(self, timestamps_sec: Iterable[float]) -> Iterator[VideoFrame]:
        targets = [float(value) for value in timestamps_sec]
        if any(not math.isfinite(value) or value < 0 for value in targets):
            raise ValueError("Sample timestamps must be finite and non-negative")
        if targets != sorted(targets):
            raise ValueError("Sample timestamps must be sorted")
        if not targets:
            return

        target_index = 0
        previous: VideoFrame | None = None
        last: VideoFrame | None = None
        for current in self.iter_frames():
            last = current
            while target_index < len(targets) and targets[target_index] <= current.timestamp_sec:
                target = targets[target_index]
                if previous is None or abs(current.timestamp_sec - target) < abs(previous.timestamp_sec - target):
                    chosen = current
                else:
                    chosen = previous
                yield chosen
                target_index += 1
            if target_index >= len(targets):
                break
            previous = current

        if last is None:
            raise VideoOpenError(f"Video contains no decodable frames: {self.path.name}")
        while target_index < len(targets):
            yield last
            target_index += 1

    def sample_nearest(self, timestamps_sec: Iterable[float]) -> list[VideoFrame]:
        return list(self.iter_sample_nearest(timestamps_sec))

    def iter_sample_frame_indices(self, timestamps_sec: Iterable[float]) -> Iterator[VideoFrame]:
        """Reproduce the accepted ML artifact's OpenCV frame-index sampling contract."""
        metadata = self.metadata()
        if metadata.nominal_fps <= 0 or metadata.frame_count is None:
            raise VideoOpenError(f"Frame-index sampling metadata is unavailable: {self.path.name}")
        targets = [float(value) for value in timestamps_sec]
        if targets != sorted(targets):
            raise ValueError("Sample timestamps must be sorted")
        indices = [
            min(
                metadata.frame_count - 1,
                max(0, int(np.rint(timestamp * metadata.nominal_fps))),
            )
            for timestamp in targets
        ]
        target_index = 0
        last: VideoFrame | None = None
        for frame_index, frame in enumerate(self.iter_frames()):
            last = frame
            while target_index < len(indices) and indices[target_index] == frame_index:
                yield VideoFrame(targets[target_index], frame.image_rgb)
                target_index += 1
            if target_index >= len(indices):
                return
        if last is None:
            raise VideoOpenError(f"Video contains no decodable frames: {self.path.name}")
        while target_index < len(indices):
            yield VideoFrame(targets[target_index], last.image_rgb)
            target_index += 1
