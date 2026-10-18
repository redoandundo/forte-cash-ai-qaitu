from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fortebank_audit import api
from fortebank_audit.automatic_rules import (
    FACTS, ROLES, FactResponse, FrameReference, RuleAssessment, RuleAwareVerifier, automatic_matches,
)
from fortebank_audit.rules import RULES, SOURCE_URL
from fortebank_audit.semantic_adapter import SemanticAdapter, SemanticIntegrationError, map_semantic_event
from test_semantic_adapter import SHA, semantic, fixture_result, runtime, bridge


FRAMES = [dict(frame_role=role, timestamp_sec=float(t)) for role, t in zip(ROLES, (8, 10, 15, 20, 22))]


def assessment_payload(supported=("CASH-04",)):
    required = {f["id"] for r in RULES if r["id"] in supported for f in r["facts"]}
    return dict(version=1, supplied_frames=copy.deepcopy(FRAMES), event_start_sec=10., event_end_sec=20.,
                model_version="qwen-fixture-v1", model_checksum=SHA, limitations=[],
                facts=[dict(fact_id=name, status="supported" if name in required else "unknown",
                            confidence="high" if name in required else "low",
                            evidence=copy.deepcopy(FRAMES),
                            explanation=f"Синтетическое наблюдение: {label}") for name, label in FACTS.items()])


def matches(payload, **kwargs):
    return automatic_matches(RuleAssessment.model_validate(payload),
                             **dict(decision="possible_violation", confidence="high", limitations=[], **kwargs))


@pytest.mark.parametrize("rule_id", [r["id"] for r in RULES])
def test_each_rule_matches_only_complete_supported_conditions(rule_id):
    result = matches(assessment_payload((rule_id,)))
    rule = next(r for r in RULES if r["id"] == rule_id)
    assert len(result) == 1 and result[0]["section"] == rule["section"]
    assert result[0]["status"] == "matched" and result[0]["source_url"] == SOURCE_URL
    assert "Автоматическое" in result[0]["applicability"] and "15.00 сек." in result[0]["applicability"]
    assert "возможного нарушения" in result[0]["applicability"]


def test_multiple_rules_are_retained_for_one_event():
    assert len(matches(assessment_payload(tuple(r["id"] for r in RULES)))) == 3


def test_single_frame_cannot_establish_a_cashier_departure():
    payload = assessment_payload()
    fact = next(f for f in payload["facts"] if f["fact_id"] == "cashier_leaves_workplace")
    fact["evidence"] = [FRAMES[2]]
    assert matches(payload) == []


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r["id"])
@pytest.mark.parametrize("state", ["unknown", "contradicted"])
def test_each_missing_or_contradicted_condition_blocks_rule(rule, state):
    for required in rule["facts"]:
        payload = assessment_payload((rule["id"],))
        next(f for f in payload["facts"] if f["fact_id"] == required["id"])["status"] = state
        assert matches(payload) == [], required["id"]


@pytest.mark.parametrize("change", ["medium", "low", "limited", "incomplete", "duplicate_time", "no_before", "no_after", "context_uncited"])
def test_uncertain_or_incomplete_visual_context_cannot_match(change):
    payload = assessment_payload()
    if change in {"medium", "low"}:
        payload["facts"][0]["confidence"] = change
        # Use a required condition rather than an unrelated rule's fact.
        next(f for f in payload["facts"] if f["fact_id"] == "cashier_leaves_workplace")["confidence"] = change
    elif change == "limited":
        payload["limitations"] = ["Клиент за пределами кадра"]
    elif change == "context_uncited":
        next(f for f in payload["facts"] if f["fact_id"] == "full_context_visible")["evidence"] = [FRAMES[2]]
    elif change == "incomplete":
        payload["supplied_frames"] = payload["supplied_frames"][1:]
        for fact in payload["facts"]:
            fact["evidence"] = [ref for ref in fact["evidence"] if ref["frame_role"] != "before"]
    else:
        role, timestamp = {"duplicate_time": ("end", 15.), "no_before": ("before", 10.), "no_after": ("after", 20.)}[change]
        for ref in payload["supplied_frames"] + [r for f in payload["facts"] for r in f["evidence"]]:
            if ref["frame_role"] == role:
                ref["timestamp_sec"] = timestamp
    assert matches(payload) == []


@pytest.mark.parametrize("change", ["invented_frame", "missing_fact", "duplicate_fact", "unknown_fact", "no_citation", "nan"])
def test_malformed_assessment_is_error_not_no_violation(change):
    payload = assessment_payload()
    if change == "invented_frame":
        payload["facts"][0]["evidence"][0]["timestamp_sec"] = 999.
    elif change == "missing_fact":
        payload["facts"].pop()
    elif change == "duplicate_fact":
        payload["facts"][-1] = payload["facts"][0]
    elif change == "unknown_fact":
        payload["facts"][0]["fact_id"] = "looks_suspicious"
    elif change == "nan":
        payload["event_start_sec"] = float("nan")
    else:
        fact = next(f for f in payload["facts"] if f["status"] == "supported")
        fact["evidence"] = []
    with pytest.raises(ValidationError):
        RuleAssessment.model_validate(payload)


def prepared_semantic():
    payload = semantic()
    payload.update(confidence="high", limitations=[], rule_assessment=assessment_payload(("CASH-02", "CASH-04")))
    return payload


def test_public_description_contains_all_matched_rules_and_sources():
    _, description, detail = map_semantic_event(prepared_semantic(), 1)
    assert len(detail["rules"]) == 2 and detail["rule"] == detail["rules"][0]
    assert "пункт 20" in description and "пункт 22" in description and SOURCE_URL in description


@pytest.mark.parametrize("field,value", [("decision", "insufficient_evidence"), ("confidence", "medium"), ("limitations", ["Occluded"])])
def test_semantic_uncertainty_overrides_supported_fact_claims(field, value):
    payload = prepared_semantic()
    payload[field] = value
    _, _, detail = map_semantic_event(payload, 1)
    assert not detail["rules"] and detail["rule"]["status"] == "not_established"


def test_wrong_assessment_model_is_rejected():
    payload = prepared_semantic()
    payload["rule_assessment"]["model_checksum"] = "b" * 64
    with pytest.raises(SemanticIntegrationError):
        map_semantic_event(payload, 1)


class FakeVerified(SimpleNamespace):
    def as_dict(self):
        return dict(self.payload)


def wrapper(outputs, decision="possible_violation", confidence="high"):
    calls = []
    def generate(images, prompt, *, max_new_tokens):
        calls.append((images, prompt, max_new_tokens))
        value = outputs[min(len(calls) - 1, len(outputs) - 1)]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(text=value, inference_sec=.5)
    backend = SimpleNamespace(model_version="qwen-fixture-v1", model_checksum=SHA, generate=generate)
    payload = semantic(decision=decision)
    payload.update(confidence=confidence, limitations=[])
    original = FakeVerified(**payload, payload=payload, inference_sec=.1)
    verifier = SimpleNamespace(backend=backend, verify=lambda context: original)
    context = SimpleNamespace(start_sec=10., end_sec=20.,
                              frames=[SimpleNamespace(role=r["frame_role"], timestamp_sec=r["timestamp_sec"],
                                                      jpeg_path=Path(f"synthetic-{i}.jpg")) for i, r in enumerate(FRAMES)])
    return RuleAwareVerifier(verifier), context, calls


def raw_response():
    payload = assessment_payload()
    return json.dumps({k: payload[k] for k in ("facts", "limitations")}, ensure_ascii=False)


def test_local_backend_reused_and_no_second_model_needed():
    verifier, context, calls = wrapper([raw_response()])
    verified = verifier.verify(context)
    assert verified.as_dict()["rule_assessment"]["version"] == 1
    assert verified.inference_sec == .6 and len(calls) == 1
    assert calls[0][0] == [f.jpeg_path for f in context.frames]


def test_schema_error_retries_once_then_fails():
    verifier, context, calls = wrapper(["invalid", raw_response()])
    assert verifier.verify(context).assessment
    assert len(calls) == 2
    verifier, context, calls = wrapper(["invalid"])
    with pytest.raises(ValueError, match="after one repair"):
        verifier.verify(context)
    assert len(calls) == 2


def test_backend_failure_is_not_retried_or_hidden():
    verifier, context, calls = wrapper([RuntimeError("CUDA failure")])
    with pytest.raises(RuntimeError, match="CUDA"):
        verifier.verify(context)
    assert len(calls) == 1


@pytest.mark.parametrize("decision,confidence", [("normal_activity", "high"), ("insufficient_evidence", "high"), ("possible_violation", "medium")])
def test_no_extra_model_call_for_ineligible_events(decision, confidence):
    verifier, context, calls = wrapper([], decision, confidence)
    verifier.verify(context)
    assert calls == []


def test_manual_override_reset_restores_automatic_matches(tmp_path, monkeypatch):
    payload = fixture_result(tmp_path / "evidence", 1)
    payload["events"][0].update(start_sec=10., end_sec=20., peak_time_sec=15., semantic_verification=prepared_semantic())
    rt = runtime(tmp_path / "evidence", bridge(payload))
    monkeypatch.setattr(api, "get_runtime", lambda: rt)
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setenv("FORTEBANK_API_KEY", "test-key")
    headers = {"X-API-Key": "test-key"}
    with TestClient(api.app) as client:
        accepted = client.post("/api/v1/analyze", headers=headers, files={"file": ("synthetic.mov", b"fixture")})
        job_id = accepted.json()["job_id"]
        progress = f"/internal/api/v1/jobs/{job_id}/progress"
        before = client.get(progress, headers=headers).json()["analysis_details"][0]
        assert len(before["rules"]) == 2
        public_before = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        assert "пункт 22" in public_before["violations"][0]["description"]
        rule = RULES[2]
        endpoint = f"/internal/api/v1/jobs/{job_id}/events/1/rule-review"
        changed = client.put(endpoint, headers=headers, json=dict(rule_id=rule["id"], confirmed_facts=[f["id"] for f in rule["facts"]]))
        assert "вручную" in changed.json()["rule"]["applicability"]
        reset = client.put(endpoint, headers=headers, json=dict(rule_id=None, confirmed_facts=[]))
        assert reset.json() == before
        assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).json() == public_before
