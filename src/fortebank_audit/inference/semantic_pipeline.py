from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

from .contracts import PipelineConfig
from .pipeline import analyze_video
from .predictor import Predictor
from .vlm_verifier import (
    EventContext, EventVerifier, TemporalFrame, get_qwen_backend, load_signal_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "inference" / "qwen_verifier_manifest.json"
DEFAULT_SIGNALS = PROJECT_ROOT / "configs" / "inference" / "observable_signals.json"
MIN_NORMAL_REJECTION_SUPPORTING_WINDOWS = 4
CONTEXT_TIME_EPSILON_SEC = 0.05


def build_semantic_verifier_from_env() -> EventVerifier | None:
    mode = os.environ.get("QAITU_VLM_MODE", "off").strip().lower()
    if mode in {"", "off", "disabled"}:
        return None
    if mode != "qwen":
        raise ValueError("QAITU_VLM_MODE must be off or qwen")
    model_path = os.environ.get("QAITU_VLM_MODEL_PATH")
    if not model_path:
        raise ValueError("QAITU_VLM_MODEL_PATH is required when QAITU_VLM_MODE=qwen")
    manifest = Path(os.environ.get("QAITU_VLM_MANIFEST_PATH", str(DEFAULT_MANIFEST)))
    signals = Path(os.environ.get("QAITU_VLM_SIGNALS_PATH", str(DEFAULT_SIGNALS)))
    concurrency = int(os.environ.get("QAITU_VLM_MAX_CONCURRENT_JOBS", "1"))
    backend = get_qwen_backend(model_path, manifest, max_concurrent_jobs=concurrency)
    backend.preload()
    return EventVerifier(
        backend,
        load_signal_policy(signals),
        max_new_tokens=int(backend.manifest["max_new_tokens"]),
    )


def _event_context(event: dict[str, Any], evidence_root: Path) -> EventContext:
    frames = tuple(
        TemporalFrame(
            role=str(item["role"]),
            timestamp_sec=float(item["timestamp_sec"]),
            jpeg_path=evidence_root / str(item["jpeg_path"]),
        )
        for item in event.get("_temporal_context", [])
    )
    return EventContext(
        start_sec=float(event["start_sec"]),
        end_sec=float(event["end_sec"]),
        peak_time_sec=float(event["peak_time_sec"]),
        ranking_score=float(event["score"]),
        supporting_windows=int(event["supporting_windows"]),
        activity_duration_sec=max(0.0, float(event["end_sec"]) - float(event["start_sec"])),
        frames=frames,
    )


def _normal_rejection_safety(
    context: EventContext, verified: Any,
) -> tuple[bool, str | None]:
    """Allow a destructive rejection only with strong temporal evidence."""
    if verified.decision != "normal_activity":
        return False, None
    if verified.confidence != "high":
        return False, "normal_activity confidence is not high"
    if context.supporting_windows < MIN_NORMAL_REJECTION_SUPPORTING_WINDOWS:
        return False, "too few supporting ONNX windows for safe rejection"
    frames = {frame.role: frame.timestamp_sec for frame in context.frames}
    required_roles = {"before", "start", "peak", "end", "after"}
    if not required_roles.issubset(frames):
        return False, "complete before/start/peak/end/after context is unavailable"
    if frames["before"] >= context.start_sec - CONTEXT_TIME_EPSILON_SEC:
        return False, "no distinct frame before the candidate interval"
    if frames["after"] <= context.end_sec + CONTEXT_TIME_EPSILON_SEC:
        return False, "no distinct frame after the candidate interval"
    observed_roles = {item.frame_role for item in verified.observations}
    if not required_roles.issubset(observed_roles):
        return False, "the verifier did not observe the complete temporal context"
    if verified.limitations or "view_quality_limitation" in verified.matched_signal_ids:
        return False, "the verifier reported material context limitations"
    return True, None


def _retained_insufficient_payload(verified: Any, reason: str) -> dict[str, Any]:
    payload = verified.as_dict()
    payload["decision"] = "insufficient_evidence"
    payload["confidence"] = "low"
    payload["reasoning"] = (
        f"Автоматическое исключение кандидата запрещено: {reason}. "
        f"Исходное наблюдение модели: {verified.reasoning}"
    )
    payload["risk"] = (
        "Кандидат сохранён для ручной проверки, потому что безопасно исключить "
        "возможный риск по доступному временному контексту нельзя."
    )
    payload["limitations"] = list(dict.fromkeys([*verified.limitations, reason]))
    return payload


def apply_semantic_verification(
    first_layer: dict[str, Any], verifier: EventVerifier, evidence_root: str | Path,
) -> dict[str, Any]:
    """Filter ONNX candidates. Verification failures deliberately propagate to job failure."""
    result = copy.deepcopy(first_layer)
    candidates = result.get("events", [])
    onnx_model = str(result.get("runtime", {}).get("model_version", "unknown-onnx"))
    if not candidates:
        result["diagnostics"] = {
            "candidate_events": 0, "rejected_candidates": 0, "retained_candidates": 0,
            "rejection_reasons": [], "models": [onnx_model], "qwen_called": False,
        }
        return result

    retained: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    verification_sec = 0.0
    peak_vram: float | None = None
    qwen_model = "unknown-qwen"
    qwen_checksum = ""
    qwen_load_sec = 0.0
    for index, event in enumerate(candidates):
        context = _event_context(event, Path(evidence_root))
        verified = verifier.verify(context)
        verification_sec += verified.inference_sec
        qwen_load_sec = max(qwen_load_sec, verified.load_sec)
        peak_vram = max(peak_vram or 0.0, verified.peak_vram_gib or 0.0)
        qwen_model, qwen_checksum = verified.model_version, verified.model_checksum
        event.pop("_temporal_context", None)
        safe_rejection, retention_reason = _normal_rejection_safety(context, verified)
        if safe_rejection:
            rejections.append({
                "candidate_index": index,
                "decision": verified.decision,
                "category": verified.category,
                "confidence": verified.confidence,
                "reason": verified.thesis,
            })
            continue
        if verified.decision == "normal_activity":
            event["semantic_verification"] = _retained_insufficient_payload(
                verified, retention_reason or "normal activity was not safe to reject"
            )
        else:
            event["semantic_verification"] = verified.as_dict()
        retained.append(event)

    result["events"] = retained
    result["verdict"] = "possible_violation" if retained else "no_violation"
    result["timecode_sec"] = (
        float(max(retained, key=lambda item: float(item["score"]))["peak_time_sec"])
        if retained else None
    )
    result["diagnostics"] = {
        "candidate_events": len(candidates),
        "rejected_candidates": len(rejections),
        "retained_candidates": len(retained),
        "rejection_reasons": rejections,
        "models": [onnx_model, qwen_model],
        "qwen_called": True,
    }
    result["runtime"]["qwen"] = {
        "backend_class": type(getattr(verifier, "backend", verifier)).__name__,
        "model_version": qwen_model,
        "checksum": qwen_checksum,
        "load_sec": qwen_load_sec,
        "inference_sec": verification_sec,
        "peak_vram_gib": peak_vram,
    }
    return result


def analyze_video_with_semantic_verification(
    video_path: str | Path, predictor: Predictor, config: PipelineConfig,
    verifier: EventVerifier, *, evidence_root: str | Path = "data/raw/evidence",
    run_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_layer = analyze_video(
        video_path, predictor, config, evidence_root=evidence_root,
        run_id=run_id, include_temporal_context=True,
    )
    result = apply_semantic_verification(first_layer, verifier, evidence_root)
    result["runtime"]["total_processing_sec"] = time.perf_counter() - started
    result["runtime"]["realtime_factor"] = (
        result["runtime"]["total_processing_sec"] / result["runtime"]["duration_sec"]
    )
    return result
