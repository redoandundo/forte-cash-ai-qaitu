from __future__ import annotations

import json
from pathlib import Path

import pytest

from fortebank_audit.inference.vlm_verifier import (
    BackendOutput, EventContext, EventVerifier, TemporalFrame,
    VerificationOutputError, VerificationRuntimeError, clear_backend_cache,
    get_qwen_backend, load_signal_policy,
)


def payload(**updates: object) -> str:
    value: dict[str, object] = {
        "decision": "normal_activity", "category": "ordinary_entry_or_presence",
        "confidence": "high", "thesis": "Человек обычно входит в рабочую зону.",
        "observations": [{"frame_role": "peak", "timestamp_sec": 2.0,
                          "visible_fact": "Виден входящий человек."}],
        "reasoning": "Дополнительных связанных подозрительных действий не видно.",
        "risk": "По наблюдаемой последовательности риск не установлен.",
        "matched_signal_ids": [], "rule_candidate": None, "limitations": [],
    }
    value.update(updates)
    return json.dumps(value, ensure_ascii=False)


class FakeBackend:
    model_version = "fake-vlm"
    model_checksum = "abc"

    def __init__(self, answers: list[str]) -> None:
        self.answers, self.calls = answers, 0

    def generate(self, images, prompt, *, max_new_tokens):
        assert images
        assert "ordinary entry" in prompt.lower()
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return BackendOutput(answer, 1.0, 2.0, 3.0)


@pytest.fixture
def policy(tmp_path: Path) -> dict:
    path = tmp_path / "signals.json"
    path.write_text(json.dumps({"signals_version": "v1", "scope": "test", "signals": [
        {"id": "transfer", "description": "Visible transfer"}
    ]}), encoding="utf-8")
    return load_signal_policy(path)


@pytest.fixture
def context(tmp_path: Path) -> EventContext:
    frame = tmp_path / "кадр.jpg"
    frame.write_bytes(b"jpeg")
    return EventContext(0.0, 4.0, 2.0, 0.7, 2, 4.0,
                        (TemporalFrame("peak", 2.0, frame),))


def test_strict_contract_accepts_normal_activity(policy: dict, context: EventContext) -> None:
    result = EventVerifier(FakeBackend([payload()]), policy, max_new_tokens=64).verify(context)
    assert result.decision == "normal_activity"
    assert result.observations[0].timestamp_sec == 2.0


@pytest.mark.parametrize("raw", ["```json\n" + payload() + "\n```", payload(extra="bad"), ""])
def test_markdown_unknown_fields_and_empty_output_are_rejected_after_retry(
    raw: str, policy: dict, context: EventContext
) -> None:
    backend = FakeBackend([raw, raw])
    with pytest.raises(VerificationOutputError):
        EventVerifier(backend, policy).verify(context)
    assert backend.calls == 2


def test_unknown_signal_and_rule_are_rejected(policy: dict, context: EventContext) -> None:
    invalid = payload(matched_signal_ids=["invented"], rule_candidate={"rule_id": "invented"})
    with pytest.raises(VerificationOutputError):
        EventVerifier(FakeBackend([invalid, invalid]), policy).verify(context)


def test_timestamp_must_belong_to_supplied_frame(policy: dict, context: EventContext) -> None:
    invalid = payload(observations=[{"frame_role": "peak", "timestamp_sec": 3.0,
                                    "visible_fact": "fact"}])
    with pytest.raises(VerificationOutputError, match="timestamp"):
        EventVerifier(FakeBackend([invalid, invalid]), policy).verify(context)


def test_inactivity_requires_temporal_sequence(policy: dict, context: EventContext) -> None:
    invalid = payload(decision="possible_violation",
        category="prolonged_inactivity_or_distraction",
        observations=[{"frame_role": "peak", "timestamp_sec": 2.0,
                       "visible_fact": "Человек сидит."}])
    with pytest.raises(VerificationOutputError, match="inactivity"):
        EventVerifier(FakeBackend([invalid, invalid]), policy).verify(context)


def test_qwen_backend_is_created_once_per_process(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "model_id": "qwen-test", "weights_files": ["model.safetensors"],
        "weights_sha256": "expected", "max_image_side": 448, "max_new_tokens": 10,
        "transformers_min_version": "0", "transformers_max_exclusive": "999",
    }), encoding="utf-8")
    monkeypatch.setattr("fortebank_audit.inference.vlm_verifier._sha256", lambda _: "expected")
    monkeypatch.setattr(
        "fortebank_audit.inference.vlm_verifier.QwenLocalBackend._validate_runtime", lambda _: None
    )
    clear_backend_cache()
    first = get_qwen_backend(model, manifest)
    second = get_qwen_backend(model, manifest)
    assert first is second


def test_corrupted_jpeg_is_a_technical_failure(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "model_id": "qwen-test", "weights_files": ["model.safetensors"],
        "weights_sha256": "expected", "max_image_side": 448, "max_new_tokens": 10,
        "transformers_min_version": "0", "transformers_max_exclusive": "999",
    }), encoding="utf-8")
    monkeypatch.setattr("fortebank_audit.inference.vlm_verifier._sha256", lambda _: "expected")
    monkeypatch.setattr(
        "fortebank_audit.inference.vlm_verifier.QwenLocalBackend._validate_runtime", lambda _: None
    )
    clear_backend_cache()
    backend = get_qwen_backend(model, manifest)
    backend._model, backend._processor = object(), object()
    damaged = tmp_path / "broken.jpg"
    damaged.write_bytes(b"not jpeg")
    with pytest.raises(VerificationRuntimeError, match="Invalid JPEG"):
        backend.generate([damaged], "prompt", max_new_tokens=10)


def _manifest(tmp_path: Path, checksum: str = "expected") -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "model_id": "qwen-test", "weights_files": ["model.safetensors"],
        "weights_sha256": checksum, "max_image_side": 448, "max_new_tokens": 10,
        "transformers_min_version": "5.20.0", "transformers_max_exclusive": "5.21.0",
    }), encoding="utf-8")
    return path


def test_missing_weights_and_bad_checksum_fail_startup(tmp_path: Path, monkeypatch) -> None:
    from fortebank_audit.inference.vlm_verifier import QwenLocalBackend

    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(VerificationRuntimeError, match="snapshot"):
        QwenLocalBackend(model, _manifest(tmp_path))
    (model / "model.safetensors").write_bytes(b"wrong")
    monkeypatch.setattr("fortebank_audit.inference.vlm_verifier._sha256", lambda _: "wrong")
    with pytest.raises(VerificationRuntimeError, match="checksum"):
        QwenLocalBackend(model, _manifest(tmp_path))


def test_cuda_and_transformers_are_startup_requirements(monkeypatch) -> None:
    import torch
    from fortebank_audit.inference.vlm_verifier import QwenLocalBackend

    backend = object.__new__(QwenLocalBackend)
    backend.manifest = {
        "transformers_min_version": "5.20.0", "transformers_max_exclusive": "5.21.0"
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(VerificationRuntimeError, match="requires CUDA"):
        backend._validate_runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.20.1"))
    backend.manifest = {
        "transformers_min_version": "999.0", "transformers_max_exclusive": "1000.0"
    }
    with pytest.raises(VerificationRuntimeError, match="Transformers"):
        backend._validate_runtime()
