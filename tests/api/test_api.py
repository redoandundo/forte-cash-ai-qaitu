from __future__ import annotations

import base64
import sys
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fortebank_audit import api
from fortebank_audit.job_runtime import get_runtime


def _video(path: Path) -> Path:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(160):
            image = np.full((48, 64, 3), 60 + index % 80, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("FORTEBANK_API_KEY", "demo-secret")
    monkeypatch.setenv("QAITU_INFERENCE_MODE", "mock")
    monkeypatch.setenv("QAITU_EVIDENCE_ROOT", str(tmp_path / "evidence"))
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path / "uploads")
    get_runtime.cache_clear()
    with api._jobs_lock:
        api._jobs.clear()
    with TestClient(api.app) as test_client:
        yield test_client
    get_runtime.cache_clear()


def _headers() -> dict[str, str]:
    return {"X-API-Key": "demo-secret"}


def test_api_rejects_invalid_key(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/missing", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_api_rejects_non_mov_upload(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        headers=_headers(),
        files={"file": ("clip.mp4", b"not-a-video", "video/mp4")},
    )
    assert response.status_code == 415


def test_mock_job_runs_full_video_pipeline(client: TestClient, tmp_path: Path) -> None:
    video = _video(tmp_path / "касса.mov")
    response = client.post(
        "/api/v1/analyze",
        headers=_headers(),
        files={"file": (video.name, video.read_bytes(), "video/quicktime")},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    job_response = client.get(f"/api/v1/jobs/{job_id}", headers=_headers())
    assert job_response.status_code == 200
    job = job_response.json()
    assert job["status"] == "done"
    assert job["verdict"] == "violation"
    assert job["violations"]
    assert "confidence" not in job["violations"][0]
    frames = job["violations"][0]["evidence_frames"]
    assert 1 <= len(frames) <= 3
    for frame in frames:
        payload = base64.b64decode(frame["image_base64"], validate=True)
        assert payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")

    progress_response = client.get(
        f"/internal/api/v1/jobs/{job_id}/progress", headers=_headers()
    )
    progress = progress_response.json()
    assert progress["stage"] == "done"
    assert progress["progress_percent"] == 100
    assert progress["runtime_mode"] == "Demo predictor"
    assert "автоматический анализ" in progress["summary"].lower()
    assert "ручн" not in progress["summary"].lower()
    assert len(progress["analysis_details"]) == len(job["violations"])
    detail = progress["analysis_details"][0]
    assert detail["event_index"] == 1
    assert len(detail["observations"]) >= 2
    assert detail["rule"]["status"] == "not_established"
    assert not list(api.UPLOAD_ROOT.glob("*.mov"))


def test_technical_failure_is_not_negative_verdict(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenRuntime:
        mode_label = "Broken test runtime"

        def analyze(self, *args, **kwargs):
            raise RuntimeError(f"secret path: {tmp_path}")

    monkeypatch.setattr(api, "get_runtime", lambda: BrokenRuntime())
    video = _video(tmp_path / "broken.mov")
    response = client.post(
        "/api/v1/analyze",
        headers=_headers(),
        files={"file": (video.name, video.read_bytes(), "video/quicktime")},
    )
    job_id = response.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}", headers=_headers()).json()
    assert job["status"] == "failed"
    assert job["verdict"] is None
    assert job["violations"] == []
    assert str(tmp_path) not in job["error"]
