from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark import PROJECT_ROOT, build_predictor, load_pipeline_config
from .pipeline import analyze_video
from .vlm_verifier import EventVerifier, QwenLocalBackend, load_signal_policy


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
) -> dict[str, Any]:
    """Measure Qwen overhead/contract reliability without changing detector outputs."""
    reference = json.loads(reference_report.read_text(encoding="utf-8"))
    source_ids = {str(item["source_id"]) for item in reference["videos"]}
    dataset = json.loads(
        (PROJECT_ROOT / "analysis" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    sources = [item for item in dataset["sources"] if str(item["source_id"]) in source_ids]
    if {str(item["source_id"]) for item in sources} != source_ids:
        raise RuntimeError("Dataset manifest does not contain the complete accepted holdout")

    _, config, manifest = load_pipeline_config()
    predictor = build_predictor("onnx")
    policy = load_signal_policy(signal_policy_path)
    verifier = EventVerifier(QwenLocalBackend(qwen_model), policy)

    qwen_statuses: Counter[str] = Counter()
    evidence_qualities: Counter[str] = Counter()
    qwen_failures: Counter[str] = Counter()
    baseline_processing_sec = 0.0
    qwen_processing_sec = 0.0
    candidate_events = 0
    public_output_changes = 0

    for source in sources:
        video_path = _resolve_video(data_root, list(source["aliases"]))
        run_id = f"ab-{source['source_id']}"
        result = analyze_video(
            video_path,
            predictor,
            config,
            evidence_root=evidence_root,
            run_id=run_id,
        )
        baseline_processing_sec += float(result["runtime"]["processing_sec"])
        public_projection_before = {
            "verdict": result["verdict"],
            "score": result["score"],
            "timecode_sec": result["timecode_sec"],
            "events": result["events"],
        }
        for event in result["events"]:
            candidate_events += 1
            paths = [evidence_root / item["jpeg_path"] for item in event["evidence"]]
            started = time.perf_counter()
            try:
                description = verifier.verify(
                    paths,
                    score=float(event["score"]),
                    peak_time_sec=float(event["peak_time_sec"]),
                )
            except Exception as exc:  # Experimental arm failure is recorded, never mapped to a verdict.
                qwen_failures[type(exc).__name__] += 1
            else:
                qwen_statuses[description.status] += 1
                evidence_qualities[description.evidence_quality] += 1
            finally:
                qwen_processing_sec += time.perf_counter() - started
        public_projection_after = {
            "verdict": result["verdict"],
            "score": result["score"],
            "timecode_sec": result["timecode_sec"],
            "events": result["events"],
        }
        public_output_changes += int(public_projection_before != public_projection_after)

    return {
        "report_version": "qwen-ab-v1",
        "model_version": manifest["model_version"],
        "holdout_sources": len(sources),
        "candidate_events": candidate_events,
        "baseline_processing_sec": baseline_processing_sec,
        "qwen_processing_sec": qwen_processing_sec,
        "qwen_statuses": dict(sorted(qwen_statuses.items())),
        "evidence_qualities": dict(sorted(evidence_qualities.items())),
        "qwen_failures": dict(sorted(qwen_failures.items())),
        "public_output_changes": public_output_changes,
        "public_verdict_influence": False,
        "timecode_influence": False,
        "quality_gain_measured": False,
        "promotion_allowed": False,
        "promotion_blocker": (
            "The holdout has detector labels/timecodes but no human labels for description quality "
            "or review-time reduction. Qwen remains experimental until a blinded human A/B shows gain."
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
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
