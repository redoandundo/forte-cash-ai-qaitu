from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from .benchmark import PROJECT_ROOT, build_predictor, load_pipeline_config
from .contracts import PipelineConfig
from .experience_store import ExperienceStore
from .meta_analysis import MetaAnalyzer, MetaFeatures
from .pipeline import analyze_video
from .reporting import build_report, save_report
from .semantic_pipeline import analyze_video_with_semantic_verification
from .vlm_verifier import EventVerifier, get_qwen_backend, load_signal_policy


DEFAULT_DB = PROJECT_ROOT / "data" / "raw" / "meta" / "experience.sqlite3"


def build_config(
    bundle_root: Path | None = None,
    *,
    allow_development_model: bool = False,
) -> tuple[PipelineConfig, str]:
    _, config, manifest = load_pipeline_config(
        bundle_root,
        allow_development_model=allow_development_model,
    )
    return config, str(manifest["sha256"])


def validate_qwen_options(enabled: bool, model_path: Path | None) -> None:
    if model_path is not None and not enabled:
        raise ValueError("Qwen requires explicit QAITU_VLM_MODE=qwen or --enable-qwen")
    if enabled and model_path is None:
        raise ValueError("Qwen mode requires --vlm-model or QAITU_VLM_MODEL_PATH")


def analyze(args: argparse.Namespace) -> None:
    env_mode = os.environ.get("QAITU_VLM_MODE", "off").strip().lower()
    enabled = bool(args.enable_qwen or env_mode == "qwen")
    model_path = args.vlm_model or (
        Path(os.environ["QAITU_VLM_MODEL_PATH"])
        if os.environ.get("QAITU_VLM_MODEL_PATH") else None
    )
    validate_qwen_options(enabled, model_path)
    store = ExperienceStore(args.db)
    config, model_sha256 = build_config(
        args.bundle,
        allow_development_model=args.allow_development_model,
    )
    predictor = build_predictor(
        "onnx",
        args.bundle,
        allow_development_model=args.allow_development_model,
    )
    run_id = args.run_id or uuid.uuid4().hex
    verifier = None
    policy = None
    if enabled:
        policy = load_signal_policy(args.signals)
        backend = get_qwen_backend(
            model_path, args.vlm_manifest,
            max_concurrent_jobs=args.max_concurrent_jobs,
        )
        backend.preload()
        verifier = EventVerifier(backend, policy, max_new_tokens=args.max_new_tokens)
        result = analyze_video_with_semantic_verification(
            args.video, predictor, config, verifier,
            evidence_root=args.evidence_root, run_id=run_id,
        )
    else:
        result = analyze_video(
            args.video, predictor, config,
            evidence_root=args.evidence_root, run_id=run_id,
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
    if enabled:
        report = build_report(
            run_id=run_id,
            first_layer=result,
            second_layer=decision.as_dict(),
            verifications=[],
            signals_version=str(policy["signals_version"]),
        )
        saved = save_report(report, args.report_root)
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
    analyze_parser.add_argument("--bundle", type=Path, help="Canonical versioned ML bundle directory")
    analyze_parser.add_argument(
        "--allow-development-model",
        action="store_true",
        help="Allow a bundle whose metrics mark it development-only",
    )
    analyze_parser.add_argument(
        "--enable-qwen", "--enable-experimental-qwen", dest="enable_qwen",
        action="store_true",
        help="Enable mandatory local Qwen semantic verification",
    )
    analyze_parser.add_argument(
        "--vlm-model",
        type=Path,
        help="Optional local Qwen3-VL model directory for visual verification",
    )
    analyze_parser.add_argument(
        "--vlm-manifest", type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "qwen_verifier_manifest.json",
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
    analyze_parser.add_argument("--max-new-tokens", type=int, default=768)
    analyze_parser.add_argument("--max-concurrent-jobs", type=int, default=1)
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
