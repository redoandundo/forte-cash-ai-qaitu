"""Keep decoder regressions isolated: PyAV may crash natively without the guard."""
import os
import subprocess
import sys
from pathlib import Path

import av
import numpy as np


def test_unknown_codec_fails_before_native_decode(tmp_path):
    path = tmp_path / "unknown.mov"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    encoded = path.read_bytes()
    assert b"mp4v" in encoded and b"esds" in encoded
    # Remove both the codec tag and the MPEG4 description; changing only the tag
    # is still decodable because FFmpeg identifies MPEG4 from the description.
    path.write_bytes(encoded.replace(b"mp4v", b"zzzz", 1).replace(b"esds", b"zzds", 1))
    program = """
import sys
from pathlib import Path
from fortebank_audit.ml.video import sample_video
from fortebank_audit.ml.preprocessing import VideoSpec
try:
    sample_video(Path(sys.argv[1]), VideoSpec())
except ValueError as error:
    assert 'Unsupported video codec' in str(error)
    print('controlled_decoder_error')
else:
    raise AssertionError('Unsupported codec produced samples')
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    result = subprocess.run([sys.executable, "-u", "-X", "faulthandler", "-c", program, str(path)],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "controlled_decoder_error" in result.stdout
