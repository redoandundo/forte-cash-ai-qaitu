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
    print(
        json.dumps(
            {"run_id": run_id, "first_layer": result, "second_layer": decision.as_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )


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
