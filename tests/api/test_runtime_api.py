from pathlib import Path
import threading
import pytest
from fastapi.testclient import TestClient
from fortebank_audit import api


@pytest.fixture
def client(monkeypatch,tmp_path):
    monkeypatch.setenv('FORTEBANK_API_KEY','local-test-key')
    monkeypatch.setattr(api,'UPLOAD_ROOT',tmp_path/'uploads')
    monkeypatch.setattr(api,'_analysis_slot',threading.BoundedSemaphore(1))
    api._jobs.clear()
    with TestClient(api.app) as client:
        yield client


def test_api_no_longer_requires_legacy_joblib(client,monkeypatch):
    monkeypatch.setattr(api,'MODEL_PATH',Path('missing-old-baseline.joblib'))
    monkeypatch.setattr(api,'analyze',lambda *args: {'verdict':'no_violation','violations':[]})
    r=client.post('/api/v1/analyze',headers={'X-API-Key':'local-test-key'},files={'file':('clip.mov',b'fixture','video/quicktime')})
    assert r.status_code==202
    job=client.get('/api/v1/jobs/'+r.json()['job_id'],headers={'X-API-Key':'local-test-key'}).json()
    assert job['status']=='done'
    assert job['verdict']=='no_violation'


def test_api_error_never_discloses_path_or_negative_verdict(client,monkeypatch):
    monkeypatch.setattr(api,'MODEL_PATH',Path(__file__))
    def fail(*args):
        raise RuntimeError('private-path secret-data')
    monkeypatch.setattr(api,'analyze',fail)
    r=client.post('/api/v1/analyze',headers={'X-API-Key':'local-test-key'},files={'file':('clip.mov',b'fixture','video/quicktime')})
    job=client.get('/api/v1/jobs/'+r.json()['job_id'],headers={'X-API-Key':'local-test-key'}).json()
    assert job['status']=='failed'
    assert job['verdict'] is None
    assert 'private-path' not in str(job)
    assert 'secret-data' not in str(job)


def test_auth_and_upload_validation(client):
    assert client.get('/api/v1/jobs/missing').status_code==401
    headers={'X-API-Key':'local-test-key'}
    assert client.get('/api/v1/jobs/missing',headers=headers).status_code==404
    assert client.post('/api/v1/analyze',headers=headers,files={'file':('x.txt',b'x')}).status_code==415


def test_busy_rejects_without_queuing(client):
    assert api._analysis_slot.acquire(blocking=False)
    try:
        result=client.post('/api/v1/analyze',headers={'X-API-Key':'local-test-key'},files={'file':('x.mov',b'x')})
        assert result.status_code==429
    finally:
        api._analysis_slot.release()


def test_failed_upload_releases_slot(client,monkeypatch):
    monkeypatch.setattr(api,'MAX_UPLOAD_BYTES',1)
    result=client.post('/api/v1/analyze',headers={'X-API-Key':'local-test-key'},files={'file':('x.mov',b'xx')})
    assert result.status_code==413
    assert api._analysis_slot.acquire(blocking=False)
    api._analysis_slot.release()
