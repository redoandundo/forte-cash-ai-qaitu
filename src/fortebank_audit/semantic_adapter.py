"""Teamlead boundary for the inference engineer's local semantic pipeline."""
from __future__ import annotations

import importlib
import inspect
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .inference.contracts import InferenceError
from .rules import rule_catalog, unmatched_rule


class SemanticIntegrationError(InferenceError):
    """Semantic code, model or output is unavailable/incompatible; never fall back."""


Text = Annotated[str, Field(min_length=1, max_length=8000)]
Timestamp = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    frame_role: Literal["before", "start", "peak", "end", "after"]
    timestamp_sec: Timestamp
    visible_fact: Text


class RuleCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    rule_id: Text
    document_name: Text
    section: Text
    applicability_reason: Text
    certainty: Literal["candidate"]


class SemanticEvent(BaseModel):
    # Inference may add diagnostics without changing the API-facing payload.
    model_config = ConfigDict(extra="ignore", strict=True, str_strip_whitespace=True)
    decision: Literal["possible_violation", "insufficient_evidence"]
    category: Literal[
        "ordinary_entry_or_presence", "cash_or_document_handling", "hand_or_object_transfer",
        "unattended_workspace", "prolonged_inactivity_or_distraction", "unclear_activity",
    ]
    confidence: Literal["low", "medium", "high"]
    thesis: Text
    observations: list[Observation] = Field(min_length=1, max_length=30)
    reasoning: Text
    risk: Text
    limitations: list[Text] = Field(max_length=30)
    matched_signal_ids: list[Text] = Field(max_length=30)
    rule_candidate: RuleCandidate | None
    model_version: Text
    model_checksum: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class SemanticAdapter:
    def __init__(self, module: object, verifier: object) -> None:
        self.module = module
        self.verifier = verifier
        backend = getattr(verifier, "backend", None)
        self.model_version = getattr(backend, "model_version", None)
        self.model_checksum = getattr(backend, "model_checksum", None)
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise SemanticIntegrationError("Qwen backend must expose model_version")
        if (not isinstance(self.model_checksum, str) or len(self.model_checksum) != 64
                or any(c not in "0123456789abcdef" for c in self.model_checksum)):
            raise SemanticIntegrationError("Qwen backend must expose a SHA-256 model_checksum")

    @classmethod
    def from_environment(cls) -> "SemanticAdapter":
        try:
            module = importlib.import_module("fortebank_audit.inference.semantic_pipeline")
            build = getattr(module, "build_semantic_verifier_from_env")
            analyze = getattr(module, "analyze_video_with_semantic_verification")
            inspect.signature(build).bind()
            inspect.signature(analyze).bind(
                "video.mov", object(), object(), object(), evidence_root="evidence", run_id="preflight"
            )
            # Catch incomplete merges before loading several GB of weights.
            from .inference.pipeline import analyze_video
            if "include_temporal_context" not in inspect.signature(analyze_video).parameters:
                raise SemanticIntegrationError("Inference pipeline lacks temporal context support")
            verifier = build()  # Backend validates local assets and preloads once at startup.
            if verifier is None:
                raise SemanticIntegrationError("Qwen was requested but verifier is disabled")
            return cls(module, verifier)
        except SemanticIntegrationError:
            raise
        except Exception as exc:
            raise SemanticIntegrationError(
                "Qwen startup failed: check inference handoff, local weights, manifest and CUDA runtime"
            ) from exc

    def analyze(self, video_path, predictor, config, *, evidence_root, run_id):
        try:
            result = self.module.analyze_video_with_semantic_verification(
                video_path, predictor, config, self.verifier,
                evidence_root=evidence_root, run_id=run_id,
            )
            if not isinstance(result, dict) or not isinstance(result.get("events"), list):
                raise SemanticIntegrationError("Qwen pipeline did not return an event list")
            expected = "possible_violation" if result["events"] else "no_violation"
            if result.get("verdict") != expected:
                raise SemanticIntegrationError("Qwen verdict and events disagree")
            diagnostics = result.get("diagnostics", {})
            candidate_count = diagnostics.get("candidate_events")
            rejected_count = diagnostics.get("rejected_candidates")
            retained_count = diagnostics.get("retained_candidates")
            if (any(type(n) is not int or n < 0 for n in
                    (candidate_count, rejected_count, retained_count))
                    or retained_count != len(result["events"])
                    or candidate_count != rejected_count + retained_count
                    or diagnostics.get("qwen_called") is not (candidate_count > 0)):
                raise SemanticIntegrationError("Qwen candidate accounting is invalid")
            if candidate_count:
                identity = result.get("runtime", {}).get("qwen", {})
                if (identity.get("model_version") != self.model_version
                        or identity.get("checksum") != self.model_checksum):
                    raise SemanticIntegrationError("Qwen runtime model identity changed")
            for event in result["events"]:
                parsed = parse_semantic_event(event.get("semantic_verification"))
                if (parsed.model_version != self.model_version
                        or parsed.model_checksum != self.model_checksum):
                    raise SemanticIntegrationError("Semantic output model identity changed")
            return result
        except SemanticIntegrationError:
            raise
        except Exception as exc:
            raise SemanticIntegrationError("Local Qwen verification failed") from exc


def parse_semantic_event(payload: object) -> SemanticEvent:
    try:
        return SemanticEvent.model_validate(payload)
    except ValidationError as exc:
        raise SemanticIntegrationError("Semantic event does not match the handoff contract") from exc


def map_semantic_event(payload: object, event_index: int) -> tuple[str, str, dict]:
    semantic = parse_semantic_event(payload)
    observations = [f"{item.timestamp_sec:.2f} сек. — {item.visible_fact}"
                    for item in semantic.observations]
    reasoning = semantic.reasoning
    if semantic.decision == "insufficient_evidence":
        reasoning = "Недостаточно данных для уверенного вывода; эпизод сохранён для проверки. " + reasoning
    if semantic.limitations:
        reasoning += " Ограничения: " + "; ".join(semantic.limitations)
    rule = unmatched_rule()
    if semantic.rule_candidate is not None:
        candidate = semantic.rule_candidate
        catalog = rule_catalog()
        canonical = next((r for r in catalog["rules"] if r["id"] == candidate.rule_id), None)
        if (canonical is None or candidate.document_name != catalog["document_name"]
                or candidate.section != canonical["section"]):
            raise SemanticIntegrationError("Qwen rule candidate does not match reviewed catalogue")
        # A model suggestion is not human confirmation of all required facts.
        rule.update(label=f"Предложено Qwen для проверки: {canonical['label']}",
                    document_name=catalog["document_name"], section=canonical["section"],
                    source_url=catalog["source_url"],
                    applicability=f"Требуется подтверждение наблюдений проверяющим. {candidate.applicability_reason}")
    detail = dict(event_index=event_index, thesis=semantic.thesis, observations=observations,
                  reasoning=reasoning, risk=semantic.risk, rule=rule)
    description = (f"{semantic.thesis} Наблюдения: {'; '.join(observations)} "
                   f"Обоснование: {reasoning} Риск: {semantic.risk} "
                   "Вывод локальной модели требует проверки человеком.")
    return semantic.thesis, description, detail
