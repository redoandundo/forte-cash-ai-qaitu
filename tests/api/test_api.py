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


def _unsupported_codec_video(path: Path) -> Path:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    encoded = path.read_bytes()
    path.write_bytes(encoded.replace(b"mp4v", b"zzzz", 1).replace(b"esds", b"zzds", 1))
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


def test_health_exposes_active_predictor_without_sensitive_paths(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "model_ready": True,
        "inference_mode": "mock",
        "runtime_mode": "Demo predictor",
        "predictor_class": "MockPredictor",
        "model_version": "mock-v1",
        "model_sha256": None,
        "runtime_providers": ["MockExecutionProvider"],
        "release_status": "test_only",
        "semantic_mode": "off",
        "semantic_model_version": None,
        "semantic_model_sha256": None,
    }
    assert str(api.UPLOAD_ROOT) not in response.text


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


@pytest.mark.parametrize(
    ("filename", "payload", "expected_error"),
    [
        ("broken.mov", b"not-a-video", "повреждено"),
    ],
)
def test_corrupt_upload_is_failed_not_no_violation(
    client: TestClient, filename: str, payload: bytes, expected_error: str
) -> None:
    accepted = client.post(
        "/api/v1/analyze",
        headers=_headers(),
        files={"file": (filename, payload, "video/quicktime")},
    )
    job_id = accepted.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}", headers=_headers()).json()
    assert job["status"] == "failed"
    assert job["verdict"] is None
    assert job["violations"] == []
    assert expected_error in job["error"]


def test_unsupported_codec_is_failed_not_no_violation(
    client: TestClient, tmp_path: Path
) -> None:
    video = _unsupported_codec_video(tmp_path / "unsupported.mov")
    accepted = client.post(
        "/api/v1/analyze",
        headers=_headers(),
        files={"file": (video.name, video.read_bytes(), "video/quicktime")},
    )
    job_id = accepted.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}", headers=_headers()).json()
    assert job["status"] == "failed"
    assert job["verdict"] is None
    assert job["violations"] == []
    assert "кодек" in job["error"].lower()


def _review_job(client: TestClient, tmp_path: Path) -> tuple[str, dict]:
    video = _video(tmp_path / "review.mov")
    accepted = client.post("/api/v1/analyze", headers=_headers(),
                           files={"file": (video.name, video.read_bytes(), "video/quicktime")})
    job_id = accepted.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}", headers=_headers()).json()
    return job_id, job


def test_rule_catalog_requires_key_and_has_verified_sources(client: TestClient) -> None:
    assert client.get("/internal/api/v1/rules").status_code == 401
    catalog = client.get("/internal/api/v1/rules", headers=_headers()).json()
    assert catalog["source_url"] == "https://adilet.zan.kz/rus/docs/V1900019680"
    assert catalog["verified_at"] == "2026-10-18"
    assert {rule["id"] for rule in catalog["rules"]} == {"CASH-02", "CASH-03", "CASH-04"}


@pytest.mark.parametrize("rule_id", ["CASH-02", "CASH-03", "CASH-04"])
def test_rule_review_requires_facts_and_can_be_reset(client: TestClient, tmp_path: Path, rule_id: str) -> None:
    job_id, before = _review_job(client, tmp_path)
    catalog = client.get("/internal/api/v1/rules", headers=_headers()).json()
    rule = next(rule for rule in catalog["rules"] if rule["id"] == rule_id)
    facts = [item["id"] for item in rule["facts"]]
    url = f"/internal/api/v1/jobs/{job_id}/events/1/rule-review"
    assert client.put(url, json={"rule_id": rule_id, "confirmed_facts": facts}).status_code == 401
    for invalid in ([], facts[:-1], facts + ["guessed"], facts + [facts[0]]):
        assert client.put(url, headers=_headers(), json={"rule_id": rule_id, "confirmed_facts": invalid}).status_code == 422
    response = client.put(url, headers=_headers(), json={"rule_id": rule_id, "confirmed_facts": facts})
    assert response.status_code == 200
    reference = response.json()["rule"]
    assert reference["status"] == "matched"
    assert "вручную" in reference["applicability"]
    after = client.get(f"/api/v1/jobs/{job_id}", headers=_headers()).json()
    assert after["verdict"] == before["verdict"]
    assert after["violations"][0]["evidence_frames"] == before["violations"][0]["evidence_frames"]
    assert "Ручная проверка" in after["violations"][0]["description"]
    assert set(after["violations"][0]) == set(before["violations"][0])
    assert client.put(url, headers=_headers(), json={"rule_id": "made-up", "confirmed_facts": facts}).status_code == 422
    assert client.put(url, headers=_headers(), json={"rule_id": None, "confirmed_facts": facts}).status_code == 422
    reset = client.put(url, headers=_headers(), json={"rule_id": None, "confirmed_facts": []})
    assert reset.json()["rule"]["status"] == "not_established"
    assert client.get(f"/api/v1/jobs/{job_id}", headers=_headers()).json() == before


def test_rule_review_rejects_missing_events_and_unfinished_jobs(client: TestClient) -> None:
    payload = {"rule_id": None, "confirmed_facts": []}
    assert client.put("/internal/api/v1/jobs/missing/events/1/rule-review", headers=_headers(), json=payload).status_code == 404
    api._set_job("pending", status="processing")
    assert client.put("/internal/api/v1/jobs/pending/events/1/rule-review", headers=_headers(), json=payload).status_code == 409
    api._set_job("negative", status="done", violations=[], analysis_details=[])
    assert client.put("/internal/api/v1/jobs/negative/events/1/rule-review", headers=_headers(), json=payload).status_code == 404
