from __future__ import annotations

from pathlib import Path

import numpy as np

from fortebank_audit.inference.contracts import (
    DetectedEvent,
    VideoFrame,
    VideoMetadata,
)
from fortebank_audit.inference.evidence import EvidenceExtractor


def test_three_evidence_frames_are_spread_around_peak(tmp_path: Path) -> None:
    class Reader:
        requested: list[float]

        def metadata(self) -> VideoMetadata:
            return VideoMetadata(
                path=Path("video.mov"),
                duration_sec=30.0,
                width=16,
                height=16,
                nominal_fps=10.0,
                frame_count=300,
                codec="mpeg4",
                has_audio=False,
            )

        def sample_nearest(self, timestamps_sec) -> list[VideoFrame]:
            self.requested = list(timestamps_sec)
            return [
                VideoFrame(timestamp, np.full((16, 16, 3), index * 20, dtype=np.uint8))
                for index, timestamp in enumerate(self.requested)
            ]

    reader = Reader()
    event = DetectedEvent(10.0, 20.0, 16.0, 0.9, 3)
    evidence = EvidenceExtractor(tmp_path).extract(
        reader,
        event,
        event_index=0,
        run_id="spread",
        count=3,
    )
    assert min(reader.requested) == 10.0
    assert max(reader.requested) == 20.0
    assert 16.0 in reader.requested
    assert [item.timecode_sec for item in evidence] == [10.0, 16.0, 20.0]
    assert len({item.jpeg_path.read_bytes() for item in evidence}) == 3


def test_visually_duplicate_frames_are_returned_once(tmp_path: Path) -> None:
    class Reader:
        def metadata(self) -> VideoMetadata:
            return VideoMetadata(
                path=Path("video.mov"),
                duration_sec=20.0,
                width=16,
                height=16,
                nominal_fps=10.0,
                frame_count=200,
                codec="mpeg4",
                has_audio=False,
            )

        def sample_nearest(self, timestamps_sec) -> list[VideoFrame]:
            return [
                VideoFrame(float(timestamp), np.full((16, 16, 3), 80, dtype=np.uint8))
                for timestamp in timestamps_sec
            ]

    event = DetectedEvent(4.0, 12.0, 8.0, 0.9, 3)
    evidence = EvidenceExtractor(tmp_path).extract(
        Reader(),
        event,
        event_index=0,
        run_id="duplicates",
        count=3,
    )

    assert len(evidence) == 1
    assert evidence[0].timecode_sec == 8.0


def test_temporal_context_is_chronological_and_deduplicated(tmp_path: Path) -> None:
    class Reader:
        def metadata(self) -> VideoMetadata:
            return VideoMetadata(Path("v.mov"), 12.0, 16, 16, 10.0, 120, "mpeg4", False)

        def sample_nearest(self, timestamps_sec) -> list[VideoFrame]:
            return [
                VideoFrame(value, np.full((16, 16, 3), index * 30, dtype=np.uint8))
                for index, value in enumerate(timestamps_sec)
            ]

    context = EvidenceExtractor(tmp_path).extract_temporal_context(
        Reader(), DetectedEvent(2.0, 8.0, 5.0, 0.8, 3),
        event_index=0, run_id="unicode-контекст",
    )
    assert [item["role"] for item in context] == ["before", "start", "peak", "end", "after"]
    assert [item["timestamp_sec"] for item in context] == sorted(
        item["timestamp_sec"] for item in context
    )
    assert all(Path(item["jpeg_path"]).is_file() for item in context)
