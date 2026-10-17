from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from .benchmark import PROJECT_ROOT, build_predictor
from .contracts import PipelineConfig
from .experience_store import ExperienceStore
from .meta_analysis import MetaAnalyzer, MetaFeatures
from .pipeline import analyze_video
from .preprocessing import PreprocessingConfig
from .reporting import build_report, save_report
from .vlm_verifier import EventVerifier, QwenLocalBackend, load_signal_policy


DEFAULT_DB = PROJECT_ROOT / "data" / "raw" / "meta" / "experience.sqlite3"


def build_config() -> tuple[PipelineConfig, str]:
    preprocessing = PreprocessingConfig.from_file(PROJECT_ROOT / "models" / "preprocessing.json")
    threshold = float(
        json.loads((PROJECT_ROOT / "models" / "thresholds.json").read_text(encoding="utf-8"))[
            "violation"
        ]
    )
    manifest = json.loads(
        (PROJECT_ROOT / "configs" / "inference" / "model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    config = PipelineConfig(
        window_sec=preprocessing.window_sec,
        stride_sec=preprocessing.stride_sec,
        num_frames=preprocessing.num_frames,
        threshold=threshold,
    )
    return config, str(manifest["sha256"])


def analyze(args: argparse.Namespace) -> None:
    store = ExperienceStore(args.db)
    config, model_sha256 = build_config()
    predictor = build_predictor("onnx")
    run_id = args.run_id or uuid.uuid4().hex
    result = analyze_video(
        args.video,
        predictor,
        config,
        evidence_root=args.evidence_root,
        run_id=run_id,
    )
    features = MetaFeatures.from_result(result)
    decision = MetaAnalyzer(store).evaluate(result, threshold=config.threshold)
    store.record_run(
        run_id=run_id,
        video_path=args.video,
        model_version=predictor.model_version,
        model_sha256=model_sha256,
        config=asdict(config),
        result=result,
        decision=decision,
        features=features,
    )
    payload: dict[str, object] = {
        "run_id": run_id,
        "first_layer": result,
        "second_layer": decision.as_dict(),
    }
    if args.vlm_model is not None:
        policy = load_signal_policy(args.signals)
        verifier = EventVerifier(
            QwenLocalBackend(args.vlm_model, max_image_side=args.max_image_side),
            policy,
            max_new_tokens=args.max_new_tokens,
        )
        verifications = []
        for event in result["events"][: args.max_vlm_events]:
            evidence_paths = [
                Path(args.evidence_root) / item["jpeg_path"] for item in event["evidence"]
            ]
            verifications.append(
                verifier.verify(
                    evidence_paths,
                    score=float(event["score"]),
                    peak_time_sec=float(event["peak_time_sec"]),
                )
            )
        report = build_report(
            run_id=run_id,
            first_layer=result,
            second_layer=decision.as_dict(),
            verifications=verifications,
            signals_version=str(policy["signals_version"]),
        )
        saved = save_report(report, args.report_root)
        payload["visual_verifications"] = [item.as_dict() for item in verifications]
        payload["report_paths"] = {
            key: path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
            for key, path in saved.items()
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def review(args: argparse.Namespace) -> None:
    store = ExperienceStore(args.db)
    store.add_review(
        args.run_id,
        outcome=args.outcome,
        corrected_timecode_sec=args.corrected_timecode_sec,
        error_type=args.error_type,
        notes=args.notes,
    )
    print(json.dumps({"run_id": args.run_id, "review_saved": True}, ensure_ascii=False))


def stats(args: argparse.Namespace) -> None:
    print(json.dumps(ExperienceStore(args.db).stats(), ensure_ascii=False, indent=2))


def plan(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            ExperienceStore(args.db).improvement_plan(
                minimum_unique_videos=args.minimum_unique_videos
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and review the local contextual analysis layer")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--video", type=Path, required=True)
    analyze_parser.add_argument(
        "--evidence-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "evidence" / "meta"
    )
    analyze_parser.add_argument("--run-id")
    analyze_parser.add_argument(
        "--vlm-model",
        type=Path,
        help="Optional local Qwen3-VL model directory for visual verification",
    )
    analyze_parser.add_argument(
        "--signals",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "observable_signals.json",
    )
    analyze_parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "reports",
    )
    analyze_parser.add_argument("--max-vlm-events", type=int, default=5)
    analyze_parser.add_argument("--max-image-side", type=int, default=448)
    analyze_parser.add_argument("--max-new-tokens", type=int, default=192)
    analyze_parser.set_defaults(handler=analyze)

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--run-id", required=True)
    review_parser.add_argument(
        "--outcome",
        choices=("confirmed_positive", "confirmed_negative", "uncertain"),
        required=True,
    )
    review_parser.add_argument("--corrected-timecode-sec", type=float)
    review_parser.add_argument("--error-type")
    review_parser.add_argument("--notes")
    review_parser.set_defaults(handler=review)

    stats_parser = subparsers.add_parser("stats")
    stats_parser.set_defaults(handler=stats)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--minimum-unique-videos", type=int, default=20)
    plan_parser.set_defaults(handler=plan)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
