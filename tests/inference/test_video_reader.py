from __future__ import annotations

from pathlib import Path

import pytest

from fortebank_audit.inference.contracts import VideoOpenError
from fortebank_audit.inference.video_reader import VideoReader


ROOT = Path(__file__).resolve().parents[2]


def test_video_metadata_unicode_path_and_timestamps(synthetic_video: Path) -> None:
    reader = VideoReader(synthetic_video)
    metadata = reader.metadata()
    assert metadata.duration_sec == pytest.approx(1.2, abs=0.2)
    frames = reader.sample_nearest([0.0, 0.3, 0.8])
    assert [frame.timestamp_sec for frame in frames] == sorted(frame.timestamp_sec for frame in frames)
    assert frames[0].image_rgb.shape[-1] == 3


def test_nearest_sampling_can_be_consumed_as_a_stream(synthetic_video: Path) -> None:
    frames = VideoReader(synthetic_video).iter_sample_nearest([0.0, 0.3, 0.8])
    first = next(frames)
    assert first.timestamp_sec == pytest.approx(0.0, abs=0.1)
    assert len(list(frames)) == 2


def test_strict_pts_sampling_validates_complete_decode(synthetic_video: Path) -> None:
    reader = VideoReader(synthetic_video)
    metadata = reader.metadata()
    frames = list(
        reader.iter_sample_nearest(
            [0.0, 0.3, 0.8],
            strict_pts=True,
            advertised_duration_sec=metadata.duration_sec,
            nominal_fps=metadata.nominal_fps,
        )
    )
    assert len(frames) == 3


def test_missing_video_is_technical_error() -> None:
    with pytest.raises(VideoOpenError):
        VideoReader(ROOT / "data" / "raw" / "missing.mov")
