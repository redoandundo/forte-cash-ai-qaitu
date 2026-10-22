import json
from pathlib import Path

import httpx
import pytest

from fortebank_audit.remote_runtime import RemoteRuntime, RemoteWorkerError


def worker(tmp_path, *, old=False, failed=False, incomplete=False, wrong_job=False, outage=False):
    key = "worker-private-key-" * 3
    key_file = tmp_path / "worker.key"
    key_file.write_text(key)
    seen = []
    def handle(request):
        seen.append(request)
        assert request.headers["X-API-Key"] == key
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "model_ready": True,
                "worker_protocol": "old" if old else "qaitu-worker-v1", "inference_mode": "onnx",
                "runtime_mode": "ONNX + Qwen", "semantic_mode": "qwen"})
        if path.endswith("/rules"):
            return httpx.Response(200, json={"rules": [{}, {}, {}]})
        if path == "/api/v1/analyze":
            if outage:
                raise httpx.ConnectError("private host with secret", request=request)
            assert b"video-content" in request.read()
            return httpx.Response(202, json={"job_id": "abc123"})
        if path.endswith("/progress"):
            return httpx.Response(200, json={"job_id": "wrong" if wrong_job else "abc123",
                "status": "failed" if failed else "done", "stage": "done", "progress_percent": 100,
                "runtime_mode": "ONNX + Qwen", "summary": "result", "runtime": {},
                "analysis_details": [] if incomplete else [{"event_index": 1, "thesis": "observed",
                    "observations": ["frame observation"], "reasoning": "why", "risk": "possible risk",
                    "rule": {"status": "not_established", "label": "review", "applicability": "unconfirmed"}}]})
        return httpx.Response(200, json={"job_id": "abc123", "status": "done", "verdict": "violation",
            "violations": [{"type": "review", "timecode_sec": 1., "description": "evidence",
                           "evidence_frames": [{"timestamp_sec": 1., "image_base64": "jpeg"}]}]})
    client = httpx.Client(base_url="http://gpu.test", transport=httpx.MockTransport(handle))
    return lambda: RemoteRuntime("http://gpu.test", str(key_file), client=client), seen, key


def test_remote_upload_preserves_dossier_without_exposing_key(tmp_path):
    build, seen, key = worker(tmp_path)
    runtime = build()
    video = tmp_path / "input.mov"
    video.write_bytes(b"video-content")
    result = runtime.analyze(video, job_id="localjob")
    assert result.verdict == "violation"
    assert result.analysis_details[0]["thesis"] == "observed"
    assert result.runtime["execution_location"] == "remote"
    assert key not in json.dumps(result.__dict__)
    assert len(seen) == 5


def test_old_worker_is_rejected_before_upload(tmp_path):
    build, seen, _ = worker(tmp_path, old=True)
    with pytest.raises(RemoteWorkerError, match="Incompatible"):
        build()
    assert len(seen) == 1


@pytest.mark.parametrize("option", ["failed", "incomplete", "wrong_job", "outage"])
def test_remote_failure_is_never_negative_verdict(tmp_path, option):
    build, _, _ = worker(tmp_path, **{option: True})
    runtime = build()
    video = tmp_path / "input.mov"
    video.write_bytes(b"video-content")
    with pytest.raises(RemoteWorkerError) as caught:
        runtime.analyze(video, job_id="localjob")
    assert "private host" not in str(caught.value)


@pytest.mark.parametrize("url", ["http://user:secret@gpu.test", "http://gpu.test/#key=secret", "file:///secret"])
def test_worker_url_cannot_carry_credentials(tmp_path, url):
    with pytest.raises(RemoteWorkerError):
        RemoteRuntime(url, str(tmp_path / "missing.key"))
