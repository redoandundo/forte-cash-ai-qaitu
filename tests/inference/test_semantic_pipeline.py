from __future__ import annotations

from pathlib import Path

import pytest

from fortebank_audit.inference.semantic_pipeline import apply_semantic_verification
from fortebank_audit.inference.vlm_verifier import Observation, VerificationResult


def first_layer(events: int = 1) -> dict:
    return {"verdict": "possible_violation" if events else "no_violation",
        "score": 0.8, "timecode_sec": 2.0 if events else None,
        "events": [{"score": 0.8, "start_sec": 1.0, "end_sec": 3.0,
            "peak_time_sec": 2.0, "supporting_windows": 4,
            "supporting_window_indices": [0, 1, 2, 3],
            "evidence": [{"timecode_sec": float(i), "jpeg_path": f"e{i}.jpg"}
                         for i in range(3)],
            "_temporal_context": [
                {"role": "before", "timestamp_sec": 0.0, "jpeg_path": "b.jpg"},
                {"role": "start", "timestamp_sec": 1.0, "jpeg_path": "s.jpg"},
                {"role": "peak", "timestamp_sec": 2.0, "jpeg_path": "q.jpg"},
                {"role": "end", "timestamp_sec": 3.0, "jpeg_path": "e.jpg"},
                {"role": "after", "timestamp_sec": 4.0, "jpeg_path": "a.jpg"},
            ]} for _ in range(events)],
        "runtime": {"model_version": "onnx-v4", "processing_sec": 1.0}}


def result(decision: str, *, limitations=(), confidence="high") -> VerificationResult:
    return VerificationResult(decision=decision, category="ordinary_entry_or_presence",
        confidence=confidence, thesis="Тезис",
        observations=(
            Observation("before", 0.0, "Факт до"),
            Observation("start", 1.0, "Факт начала"),
            Observation("peak", 2.0, "Факт"),
            Observation("end", 3.0, "Факт конца"),
            Observation("after", 4.0, "Факт после"),
        ),
        reasoning="Обоснование", risk="Риск не установлен", matched_signal_ids=(),
        rule_candidate=None, limitations=tuple(limitations), model_version="qwen",
        model_checksum="abc", load_sec=1.0, inference_sec=2.0,
        peak_vram_gib=3.0, retry_count=0)


class Verifier:
    def __init__(self, outputs): self.outputs, self.calls = outputs, 0
    def verify(self, context):
        out = self.outputs[self.calls]
        self.calls += 1
        if isinstance(out, Exception): raise out
        return out


def test_normal_activity_removes_candidate() -> None:
    output = apply_semantic_verification(first_layer(), Verifier([result("normal_activity")]), Path("root"))
    assert output["verdict"] == "no_violation"
    assert output["events"] == []
    assert output["diagnostics"]["rejected_candidates"] == 1


def test_low_confidence_normal_activity_is_retained_as_insufficient_evidence() -> None:
    output = apply_semantic_verification(
        first_layer(), Verifier([result("normal_activity", confidence="low")]), Path("root")
    )
    assert output["verdict"] == "possible_violation"
    assert len(output["events"]) == 1
    assert output["events"][0]["semantic_verification"]["decision"] == "insufficient_evidence"
    assert output["diagnostics"]["rejected_candidates"] == 0


@pytest.mark.parametrize("missing_role", ["before", "after"])
def test_normal_activity_without_strict_before_and_after_is_retained(
    missing_role: str,
) -> None:
    layer = first_layer()
    layer["events"][0]["_temporal_context"] = [
        frame for frame in layer["events"][0]["_temporal_context"]
        if frame["role"] != missing_role
    ]
    output = apply_semantic_verification(
        layer, Verifier([result("normal_activity")]), Path("root")
    )
    assert output["verdict"] == "possible_violation"
    assert output["events"][0]["semantic_verification"]["decision"] == "insufficient_evidence"


def test_normal_activity_with_too_few_supporting_windows_is_retained() -> None:
    layer = first_layer()
    layer["events"][0]["supporting_windows"] = 3
    output = apply_semantic_verification(
        layer, Verifier([result("normal_activity")]), Path("root")
    )
    assert output["verdict"] == "possible_violation"
    assert output["events"][0]["semantic_verification"]["decision"] == "insufficient_evidence"


def test_normal_activity_with_limitations_is_retained() -> None:
    output = apply_semantic_verification(
        first_layer(),
        Verifier([result("normal_activity", limitations=("Ракурс ограничен",))]),
        Path("root"),
    )
    assert output["verdict"] == "possible_violation"
    assert output["events"][0]["semantic_verification"]["decision"] == "insufficient_evidence"


@pytest.mark.parametrize("decision", ["possible_violation", "insufficient_evidence"])
def test_possible_and_insufficient_keep_one_event(decision: str) -> None:
    limits = ("Плохой ракурс",) if decision == "insufficient_evidence" else ()
    output = apply_semantic_verification(first_layer(), Verifier([result(decision, limitations=limits)]), Path("root"))
    assert len(output["events"]) == 1
    assert len(output["events"][0]["evidence"]) == 3
    assert output["events"][0]["semantic_verification"]["decision"] == decision


def test_no_candidates_never_call_qwen() -> None:
    verifier = Verifier([])
    output = apply_semantic_verification(first_layer(0), verifier, Path("root"))
    assert verifier.calls == 0
    assert output["verdict"] == "no_violation"


def test_qwen_failure_propagates_instead_of_no_violation() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        apply_semantic_verification(first_layer(), Verifier([RuntimeError("boom")]), Path("root"))


def test_diagnostics_name_both_models() -> None:
    output = apply_semantic_verification(first_layer(), Verifier([result("possible_violation")]), Path("root"))
    assert output["diagnostics"]["models"] == ["onnx-v4", "qwen"]


@pytest.mark.parametrize("candidate_count", [1, 2, 3])
def test_candidate_count_is_not_padded_or_duplicated(candidate_count: int) -> None:
    output = apply_semantic_verification(
        first_layer(candidate_count),
        Verifier([result("possible_violation") for _ in range(candidate_count)]),
        Path("root"),
    )
    assert len(output["events"]) == candidate_count
