from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fortebank_audit import api, semantic_adapter as adapter_module
from fortebank_audit.inference.contracts import PipelineConfig
from fortebank_audit.inference.predictor import MockPredictor
from fortebank_audit.job_runtime import InferenceRuntime, RuntimeSettings, RuntimeConfigurationError
from fortebank_audit.rules import rule_catalog
from fortebank_audit.semantic_adapter import SemanticAdapter, SemanticIntegrationError, map_semantic_event

SHA = "a" * 64


def semantic(index=0, decision="possible_violation"):
    return dict(
        decision=decision, category="unclear_activity", confidence="medium",
        thesis=f"Синтетический эпизод {index + 1}",
        observations=[dict(frame_role="peak", timestamp_sec=float(index * 10 + 2), visible_fact="Тестовое наблюдение")],
        reasoning="Тестовое обоснование", risk="Тестовый риск",
        matched_signal_ids=[], rule_candidate=None, limitations=["Тестовое ограничение"],
        model_version="qwen-fixture-v1", model_checksum=SHA,
    )


def fixture_result(root, retained=2, rejected=0):
    root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "synthetic.jpg"), np.zeros((16, 16, 3), dtype=np.uint8))
    return dict(
        verdict="possible_violation" if retained else "no_violation",
        events=[dict(start_sec=float(i * 10), end_sec=float(i * 10 + 4), peak_time_sec=float(i * 10 + 2),
                     score=0.9, supporting_windows=4, semantic_verification=semantic(i),
                     evidence=[dict(timecode_sec=float(i * 10 + j), jpeg_path="synthetic.jpg") for j in (1, 2, 3)])
                for i in range(retained)],
        runtime=dict(duration_sec=40., qwen=dict(model_version="qwen-fixture-v1", checksum=SHA)),
        diagnostics=dict(candidate_events=retained + rejected, retained_candidates=retained,
                         rejected_candidates=rejected, qwen_called=retained + rejected > 0),
    )


def bridge(payload=None, failure=None):
    def analyze(*args, **kwargs):
        if failure:
            raise failure
        return copy.deepcopy(payload)
    verifier = SimpleNamespace(backend=SimpleNamespace(model_version="qwen-fixture-v1", model_checksum=SHA))
    return SemanticAdapter(SimpleNamespace(analyze_video_with_semantic_verification=analyze), verifier)


def runtime(root, adapter):
    return InferenceRuntime(
        RuntimeSettings("onnx", None, False, root, vlm_mode="qwen"),
        predictor=MockPredictor([.9]), config=PipelineConfig(6., 2., 4, .5), semantic_adapter=adapter,
    )


@pytest.mark.parametrize("retained,rejected", [(0, 0), (0, 3), (1, 0), (2, 1), (3, 0)])
def test_qwen_results_map_events_not_frames(tmp_path, retained, rejected):
    result = runtime(tmp_path, bridge(fixture_result(tmp_path, retained, rejected))).analyze("fixture.mov", job_id="fixture")
    assert len(result.violations) == len(result.analysis_details) == retained
    assert result.verdict == ("violation" if retained else "no_violation")
    assert result.runtime["semantic_model_sha256"] == SHA
    assert result.runtime["diagnostics"]["semantic"]["rejected_candidates"] == rejected
    for i, (event, detail) in enumerate(zip(result.violations, result.analysis_details)):
        assert event["timecode_sec"] == i * 10 + 2
        assert len(event["evidence_frames"]) == 3
        assert detail["event_index"] == i + 1
        assert detail["thesis"] == f"Синтетический эпизод {i + 1}"
        assert "Тестовое наблюдение" in event["description"]
        assert "Тестовое ограничение" in event["description"]
        assert detail["rule"]["status"] == "not_established"


@pytest.mark.parametrize("decision", ["possible_violation", "insufficient_evidence"])
def test_semantic_details_preserve_uncertainty(decision):
    _, description, detail = map_semantic_event(semantic(decision=decision), 1)
    assert "Тестовое ограничение" in description
    assert ("Недостаточно данных" in detail["reasoning"]) == (decision == "insufficient_evidence")


@pytest.mark.parametrize("field,value", [
    ("decision", "normal_activity"), ("decision", "violation"), ("thesis", " "),
    ("observations", []), ("observations", [dict(frame_role="peak", timestamp_sec=float("nan"), visible_fact="x")]),
    ("model_checksum", "wrong"), ("category", "theft"), ("limitations", "none"),
])
def test_malformed_semantics_cannot_become_valid_output(field, value):
    payload = semantic()
    payload[field] = value
    with pytest.raises(SemanticIntegrationError):
        map_semantic_event(payload, 1)


def test_rule_suggestion_requires_human_confirmation():
    catalog = rule_catalog()
    rule = catalog["rules"][0]
    payload = semantic()
    payload["rule_candidate"] = dict(rule_id=rule["id"], document_name=catalog["document_name"],
                                     section=rule["section"], applicability_reason="Тестовая гипотеза", certainty="candidate")
    _, _, detail = map_semantic_event(payload, 1)
    assert detail["rule"]["status"] == "not_established"
    assert detail["rule"]["source_url"] == catalog["source_url"]
    payload["rule_candidate"]["section"] = "выдуманный пункт"
    with pytest.raises(SemanticIntegrationError):
        map_semantic_event(payload, 1)


@pytest.mark.parametrize("mutation", ["missing_semantics", "identity", "verdict", "counts", "missing_diagnostics", "negative_identity"])
def test_incompatible_pipeline_output_fails(tmp_path, mutation):
    payload = fixture_result(tmp_path)
    if mutation == "missing_semantics":
        payload["events"][0].pop("semantic_verification")
    elif mutation == "identity":
        payload["events"][0]["semantic_verification"]["model_checksum"] = "b" * 64
    elif mutation == "verdict":
        payload["verdict"] = "no_violation"
    elif mutation == "counts":
        payload["diagnostics"]["retained_candidates"] = 1
    elif mutation == "missing_diagnostics":
        payload.pop("diagnostics")
    else:
        payload = fixture_result(tmp_path, 0, 2)
        payload["runtime"]["qwen"]["checksum"] = "b" * 64
    with pytest.raises(SemanticIntegrationError):
        runtime(tmp_path, bridge(payload)).analyze("fixture.mov", job_id="fixture")


def test_qwen_api_failure_is_failed_not_onnx_fallback(tmp_path, monkeypatch):
    rt = runtime(tmp_path / "evidence", bridge(failure=RuntimeError("private model path")))
    monkeypatch.setattr(api, "get_runtime", lambda: rt)
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setenv("FORTEBANK_API_KEY", "test-key")
    with TestClient(api.app) as client:
        accepted = client.post("/api/v1/analyze", headers={"X-API-Key": "test-key"},
                               files={"file": ("synthetic.mov", b"fixture", "video/quicktime")})
        job = client.get(f"/api/v1/jobs/{accepted.json()['job_id']}", headers={"X-API-Key": "test-key"}).json()
    assert job["status"] == "failed"
    assert job["verdict"] is None and job["violations"] == []
    assert "Qwen" in job["error"] and "private" not in job["error"]
    assert not list((tmp_path / "uploads").glob("*.mov"))


def test_qwen_api_success_keeps_public_schema_and_manual_review(tmp_path, monkeypatch):
    rt = runtime(tmp_path / "evidence", bridge(fixture_result(tmp_path / "evidence")))
    monkeypatch.setattr(api, "get_runtime", lambda: rt)
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setenv("FORTEBANK_API_KEY", "test-key")
    headers = {"X-API-Key": "test-key"}
    with TestClient(api.app) as client:
        health = client.get("/health").json()
        assert health["semantic_mode"] == "qwen" and health["semantic_model_sha256"] == SHA
        accepted = client.post("/api/v1/analyze", headers=headers,
                               files={"file": ("synthetic.mov", b"fixture", "video/quicktime")})
        job_id = accepted.json()["job_id"]
        job = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        assert job["status"] == "done" and len(job["violations"]) == 2
        assert set(job["violations"][0]) == {"type", "description", "timecode_sec", "evidence_frames"}
        details = client.get(f"/internal/api/v1/jobs/{job_id}/progress", headers=headers).json()["analysis_details"]
        assert details[1]["event_index"] == 2 and details[1]["thesis"] == "Синтетический эпизод 2"
        rule = rule_catalog()["rules"][0]
        review_url = f"/internal/api/v1/jobs/{job_id}/events/2/rule-review"
        reviewed = client.put(review_url, headers=headers, json=dict(rule_id=rule["id"],
                              confirmed_facts=[f["id"] for f in rule["facts"]]))
        assert reviewed.status_code == 200 and reviewed.json()["rule"]["status"] == "matched"
        client.put(review_url, headers=headers, json=dict(rule_id=None, confirmed_facts=[]))
        assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).json() == job


def test_verifier_preloaded_once_and_reused(tmp_path, monkeypatch):
    payload = fixture_result(tmp_path)
    adapter = bridge(payload)
    calls = []
    def build():
        calls.append("load")
        return adapter.verifier
    module = SimpleNamespace(build_semantic_verifier_from_env=build,
                             analyze_video_with_semantic_verification=adapter.module.analyze_video_with_semantic_verification)
    monkeypatch.setattr(adapter_module.importlib, "import_module", lambda name: module)
    from fortebank_audit.inference import pipeline
    def compatible_pipeline(*args, include_temporal_context=False, **kwargs):
        pass
    monkeypatch.setattr(pipeline, "analyze_video", compatible_pipeline)
    loaded = SemanticAdapter.from_environment()
    rt = runtime(tmp_path, loaded)
    rt.analyze("fixture.mov", job_id="one")
    rt.analyze("fixture.mov", job_id="two")
    assert calls == ["load"]


def test_missing_inference_handoff_fails_startup(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(adapter_module.importlib, "import_module", missing)
    with pytest.raises(SemanticIntegrationError, match="startup failed"):
        SemanticAdapter.from_environment()


def test_incomplete_merge_fails_before_loading_weights(monkeypatch):
    module = SimpleNamespace(build_semantic_verifier_from_env=lambda: pytest.fail("must not load weights"),
                             analyze_video_with_semantic_verification=lambda *a, **kw: None)
    monkeypatch.setattr(adapter_module.importlib, "import_module", lambda name: module)
    from fortebank_audit.inference import pipeline
    monkeypatch.setattr(pipeline, "analyze_video", lambda *a, **kw: None)
    with pytest.raises(SemanticIntegrationError, match="temporal context"):
        SemanticAdapter.from_environment()


def test_qwen_off_does_not_import_or_load_model(tmp_path, monkeypatch):
    monkeypatch.setattr(SemanticAdapter, "from_environment", lambda: pytest.fail("unexpected model load"))
    rt = InferenceRuntime(RuntimeSettings("mock", None, False, tmp_path))
    assert rt.semantic is None
    assert rt.model_diagnostics["semantic_mode"] == "off"


@pytest.mark.parametrize("mode", ["typo", "qwen"])
def test_invalid_or_mock_qwen_configuration_rejected(monkeypatch, mode):
    monkeypatch.setenv("QAITU_INFERENCE_MODE", "mock")
    monkeypatch.setenv("QAITU_VLM_MODE", mode)
    with pytest.raises(RuntimeConfigurationError):
        RuntimeSettings.from_environment()
