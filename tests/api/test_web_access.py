from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from test_api import api, client, _video, _headers

WEB = {"X-Qaitu-Client": "web"}


@pytest.fixture(autouse=True)
def clean_web_state(client):
    with api._web_lock:
        api._web_sessions.clear()
        api._web_job_owners.clear()
        api._web_pending.clear()
    yield
    with api._web_lock:
        api._web_sessions.clear()
        api._web_job_owners.clear()
        api._web_pending.clear()


def test_anonymous_browser_can_start_session_without_jury_key(client):
    response = client.post("/web/api/v1/session", headers=WEB)
    assert response.status_code == 200 and response.json() == {"ready": True}
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=lax" in response.headers["set-cookie"].lower()
    assert "Path=/web" in response.headers["set-cookie"]
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/web/api/v1/rules", headers=WEB).status_code == 200
    assert client.get("/api/v1/jobs/missing", headers=WEB).status_code == 401
    assert client.get("/internal/api/v1/rules", headers=WEB).status_code == 401
    assert client.post("/api/v1/analyze", files={"file": ("synthetic.mov", b"x")}, headers=WEB).status_code == 401


def test_full_web_flow_and_manual_review_without_api_key(client, tmp_path):
    client.post("/web/api/v1/session", headers=WEB)
    video = _video(tmp_path / "synthetic.mov")
    created = client.post("/web/api/v1/analyze", headers=WEB, files={"file": (video.name, video.read_bytes())})
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    result = client.get(f"/web/api/v1/jobs/{job_id}", headers=WEB)
    assert result.status_code == 200 and result.json()["status"] == "done"
    assert result.headers["cache-control"] == "no-store"
    assert result.json()["violations"][0]["evidence_frames"]
    progress = client.get(f"/web/api/v1/jobs/{job_id}/progress", headers=WEB)
    assert progress.status_code == 200 and progress.json()["analysis_details"]
    rule = client.get("/web/api/v1/rules", headers=WEB).json()["rules"][0]
    reviewed = client.put(f"/web/api/v1/jobs/{job_id}/events/1/rule-review", headers=WEB,
                          json=dict(rule_id=rule["id"], confirmed_facts=[f["id"] for f in rule["facts"]]))
    assert reviewed.status_code == 200 and reviewed.json()["rule"]["status"] == "matched"
    # Jury retains its existing authenticated access to the same underlying runtime.
    assert client.get(f"/api/v1/jobs/{job_id}", headers=_headers()).status_code == 200


def test_another_browser_cannot_read_or_change_others_jobs(client, tmp_path):
    client.post("/web/api/v1/session", headers=WEB)
    video = _video(tmp_path / "synthetic.mov")
    created = client.post("/web/api/v1/analyze", headers=WEB, files={"file": (video.name, video.read_bytes())})
    job_id = created.json()["job_id"]
    with TestClient(api.app) as other:
        other.post("/web/api/v1/session", headers=WEB)
        assert other.get(f"/web/api/v1/jobs/{job_id}", headers=WEB).status_code == 404
        assert other.get(f"/web/api/v1/jobs/{job_id}/progress", headers=WEB).status_code == 404
        assert other.put(f"/web/api/v1/jobs/{job_id}/events/1/rule-review", headers=WEB,
                         json=dict(rule_id=None)).status_code == 404


@pytest.mark.parametrize("headers", [{}, {**WEB, "Origin": "https://malicious.example"},
                                      {**WEB, "Sec-Fetch-Site": "cross-site"}])
def test_cross_site_requests_are_rejected(client, headers):
    assert client.post("/web/api/v1/session", headers=headers).status_code == 403


def test_missing_or_expired_cookie_requires_new_automatic_session(client):
    assert client.get("/web/api/v1/rules", headers=WEB).status_code == 401
    client.post("/web/api/v1/session", headers=WEB)
    for session in api._web_sessions:
        api._web_sessions[session] = 0
    assert client.get("/web/api/v1/rules", headers=WEB).status_code == 401
    assert client.post("/web/api/v1/session", headers=WEB).status_code == 200
    assert client.get("/web/api/v1/rules", headers=WEB).status_code == 200


def test_web_queue_limit_does_not_open_jury_api(client):
    client.post("/web/api/v1/session", headers=WEB)
    session = next(iter(api._web_sessions))
    for i in range(2):
        job_id = f"busy-{i}"
        api._web_job_owners[job_id] = session
        api._jobs[job_id] = {"status": "processing"}
    response = client.post("/web/api/v1/analyze", headers=WEB, files={"file": ("synthetic.mov", b"x")})
    assert response.status_code == 429
    assert not api._web_pending
    assert client.get("/api/v1/jobs/busy-0").status_code == 401
