from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark import PROJECT_ROOT, build_predictor, load_pipeline_config
from .pipeline import analyze_video
from .semantic_pipeline import apply_semantic_verification
from .vlm_verifier import EventVerifier, get_qwen_backend, load_signal_policy


def _resolve_video(data_root: Path, aliases: list[str]) -> Path:
    path = next((data_root / alias for alias in aliases if (data_root / alias).is_file()), None)
    if path is None:
        raise FileNotFoundError("A holdout video is unavailable locally")
    return path


def evaluate_qwen_ab(
    *,
    data_root: Path,
    reference_report: Path,
    qwen_model: Path,
    evidence_root: Path,
    signal_policy_path: Path,
    bundle_root: Path | None = None,
    allow_development_model: bool = False,
) -> dict[str, Any]:
    """Measure semantic filtering quality and technical reliability on a fixed holdout."""
    reference = json.loads(reference_report.read_text(encoding="utf-8"))
    source_ids = {str(item["source_id"]) for item in reference["videos"]}
    dataset = json.loads(
        (PROJECT_ROOT / "analysis" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    sources = [item for item in dataset["sources"] if str(item["source_id"]) in source_ids]
    if {str(item["source_id"]) for item in sources} != source_ids:
        raise RuntimeError("Dataset manifest does not contain the complete accepted holdout")

    _, config, manifest = load_pipeline_config(
        bundle_root,
        allow_development_model=allow_development_model,
    )
    predictor = build_predictor(
        "onnx",
        bundle_root,
        allow_development_model=allow_development_model,
    )
    policy = load_signal_policy(signal_policy_path)
    verifier = EventVerifier(
        get_qwen_backend(
            qwen_model,
            PROJECT_ROOT / "configs" / "inference" / "qwen_verifier_manifest.json",
        ),
        policy,
    )

    qwen_statuses: Counter[str] = Counter()
    qwen_failures: Counter[str] = Counter()
    baseline_processing_sec = 0.0
    qwen_processing_sec = 0.0
    candidate_events = 0
    public_output_changes = 0
    false_positives_removed = 0
    true_positives_removed = 0

    for source in sources:
        video_path = _resolve_video(data_root, list(source["aliases"]))
        run_id = f"ab-{source['source_id']}"
        result = analyze_video(
            video_path,
            predictor,
            config,
            evidence_root=evidence_root,
            run_id=run_id,
            include_temporal_context=True,
        )
        baseline_processing_sec += float(result["runtime"]["processing_sec"])
        public_events_before = [
            {key: value for key, value in event.items() if key != "_temporal_context"}
            for event in result["events"]
        ]
        public_projection_before = {
            "verdict": result["verdict"],
            "score": result["score"],
            "timecode_sec": result["timecode_sec"],
            "events": public_events_before,
        }
        candidate_events += len(result["events"])
        started = time.perf_counter()
        try:
            filtered = apply_semantic_verification(result, verifier, evidence_root)
        except Exception as exc:
            qwen_failures[type(exc).__name__] += 1
            continue
        finally:
            qwen_processing_sec += time.perf_counter() - started
        for rejection in filtered["diagnostics"]["rejection_reasons"]:
            qwen_statuses[rejection["decision"]] += 1
        for event in filtered["events"]:
            qwen_statuses[event["semantic_verification"]["decision"]] += 1
        public_projection_after = {
            "verdict": filtered["verdict"],
            "score": filtered["score"],
            "timecode_sec": filtered["timecode_sec"],
            "events": filtered["events"],
        }
        public_output_changes += int(public_projection_before != public_projection_after)
        removed = len(result["events"]) - len(filtered["events"])
        if removed and source.get("label") in {0, "negative", False}:
            false_positives_removed += removed
        if removed and source.get("label") in {1, "positive", True}:
            true_positives_removed += removed

    return {
        "report_version": "qwen-semantic-ab-v2",
        "model_version": manifest["model_version"],
        "holdout_sources": len(sources),
        "candidate_events": candidate_events,
        "baseline_processing_sec": baseline_processing_sec,
        "qwen_processing_sec": qwen_processing_sec,
        "qwen_statuses": dict(sorted(qwen_statuses.items())),
        "qwen_failures": dict(sorted(qwen_failures.items())),
        "public_output_changes": public_output_changes,
        "public_verdict_influence": True,
        "timecode_influence": True,
        "false_positives_removed": false_positives_removed,
        "true_positives_removed": true_positives_removed,
        "quality_gain_measured": not qwen_failures,
        "promotion_allowed": False,
        "promotion_blocker": (
            "Promotion requires zero schema/technical failures and review of every removed true positive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated Qwen holdout A/B")
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "open" / "videos"
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "mobilenet_pair_safe_v2_full_video.json",
    )
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, help="Canonical versioned ML bundle directory")
    parser.add_argument(
        "--allow-development-model",
        action="store_true",
        help="Allow a bundle whose metrics mark it development-only",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "evidence" / "qwen-ab",
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "observable_signals.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = evaluate_qwen_ab(
        data_root=args.data_root,
        reference_report=args.reference_report,
        qwen_model=args.qwen_model,
        evidence_root=args.evidence_root,
        signal_policy_path=args.signals,
        bundle_root=args.bundle,
        allow_development_model=args.allow_development_model,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
