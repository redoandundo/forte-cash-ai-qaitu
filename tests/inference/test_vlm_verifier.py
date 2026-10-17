from __future__ import annotations

import json
from pathlib import Path

import pytest

from fortebank_audit.inference.reporting import build_report, render_markdown
from fortebank_audit.inference.meta_cli import validate_qwen_options
from fortebank_audit.inference.vlm_verifier import (
    BackendOutput,
    EventVerifier,
    VerificationOutputError,
    load_signal_policy,
    parse_verification_output,
)


class FakeBackend:
    model_version = "fake-vlm"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(
        self,
        images: tuple[Path, ...],
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> BackendOutput:
        assert images
        assert "Do not infer identity" in prompt
        assert max_new_tokens == 64
        return BackendOutput(self.text, 1.0, 2.0, 3.0)


@pytest.fixture
def policy(tmp_path: Path) -> dict:
    path = tmp_path / "signals.json"
    path.write_text(
        json.dumps(
            {
                "signals_version": "test-v1",
                "scope": "test",
                "signals": [{"id": "transfer", "description": "Visible transfer"}],
            }
        ),
        encoding="utf-8",
    )
    return load_signal_policy(path)


def test_event_verifier_accepts_strict_json(tmp_path: Path, policy: dict) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fixture")
    backend = FakeBackend(
        json.dumps(
            {
                "status": "description_available",
                "evidence_quality": "sufficient",
                "summary": "Видима передача предмета.",
                "visible_actions": ["передача предмета"],
                "matched_signal_ids": ["transfer"],
                "limitations": [],
            },
            ensure_ascii=False,
        )
    )
    result = EventVerifier(backend, policy, max_new_tokens=64).verify(
        [image], score=0.7, peak_time_sec=12.0
    )
    assert result.status == "description_available"
    assert result.evidence_quality == "sufficient"
    assert result.model_version == "fake-vlm"
    assert result.matched_signal_ids == ("transfer",)


def test_unknown_signal_is_rejected() -> None:
    output = BackendOutput("", 0.0, 0.0, None)
    with pytest.raises(VerificationOutputError, match="unknown signal"):
        parse_verification_output(
            json.dumps(
                {
                    "status": "description_available",
                    "evidence_quality": "sufficient",
                    "summary": "test",
                    "visible_actions": [],
                    "matched_signal_ids": ["invented"],
                    "limitations": [],
                }
            ),
            allowed_signal_ids={"known"},
            model_version="fake",
            backend_output=output,
        )


def test_report_is_deterministic_and_non_legal(tmp_path: Path, policy: dict) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fixture")
    result = EventVerifier(
        FakeBackend(
            json.dumps(
                {
                    "status": "insufficient_evidence",
                    "evidence_quality": "unusable",
                    "summary": "Кадр перекрыт.",
                    "visible_actions": [],
                    "matched_signal_ids": [],
                    "limitations": ["occlusion"],
                }
            )
        ),
        policy,
        max_new_tokens=64,
    ).verify([image], score=0.6, peak_time_sec=4.0)
    report = build_report(
        run_id="test-run",
        first_layer={
            "events": [{"score": 0.6, "peak_time_sec": 4.0}],
        },
        second_layer={"human_review_required": True, "next_action": "request_human_review"},
        verifications=[result],
        signals_version="test-v1",
    )
    markdown = render_markdown(report)
    assert report["recommendation"] == "request_human_review"
    assert report["qwen_experiment"]["public_verdict_influence"] is False
    assert report["qwen_experiment"]["ab_evaluation_only"] is True
    assert "does not establish" in report["scope"]
    assert "insufficient_evidence" in markdown
    assert "confidence" not in markdown.lower()


def test_qwen_requires_explicit_experimental_feature_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="experimental feature flag"):
        validate_qwen_options(False, tmp_path / "qwen")
    with pytest.raises(ValueError, match="requires --vlm-model"):
        validate_qwen_options(True, None)
    validate_qwen_options(True, tmp_path / "qwen")
