from pathlib import Path
from fractions import Fraction

import av
import numpy as np
import pytest

from fortebank_audit.ml.preprocessing import VideoSpec
from fortebank_audit.ml.video import sample_video


@pytest.mark.parametrize("fps", [10, 24, 30])
def test_pts_unicode_short_video(tmp_path: Path, fps):
    path = tmp_path / "касса.mov"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    samples = sample_video(path, VideoSpec())
    assert samples.frames.shape == (1, 4, 160, 160, 3)
    assert samples.centers[0] == pytest.approx(6 / fps)
    assert np.all(samples.actual_timestamps >= 0)
    assert np.all(samples.actual_timestamps < samples.duration)
    assert len(np.unique(samples.actual_timestamps)) == 4


def test_variable_pts_are_used_instead_of_frame_index(tmp_path):
    path = tmp_path / "vfr.mkv"
    timestamps_ms = [0, 40, 80, 160, 240, 280, 400, 440]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, timestamp in enumerate(timestamps_ms):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 20, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    samples = sample_video(path, VideoSpec())
    actual = samples.actual_timestamps[0]
    expected_pts = np.asarray(timestamps_ms) / 1000
    requested = np.linspace(0, samples.duration, 4, endpoint=False)
    for selected, target in zip(actual, requested):
        assert np.min(np.abs(expected_pts - selected)) < 1e-9
        assert abs(selected - target) == pytest.approx(np.min(np.abs(expected_pts - target)))


def test_broken_video_is_error(tmp_path):
    path = tmp_path / "broken.mov"
    path.write_bytes(b"invalid container")
    with pytest.raises(Exception):
        sample_video(path, VideoSpec())
