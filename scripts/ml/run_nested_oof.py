from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*parts: object) -> None:
    command = [str(part) for part in parts]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested scenario-group OOF with inner-fold threshold calibration.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--experiment", default="nested_oof")
    parser.add_argument("--model", default="mobilenet_v3_small_temporal")
    parser.add_argument("--pair-ranking", action="store_true")
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--hard-negative-mining", action="store_true")
    args = parser.parse_args()
    build_command: list[object] = [sys.executable, "scripts/ml/build_oof_manifests.py"]
    if args.hard_negative_mining:
        hard_negative_manifest = ROOT / "analysis" / "hard_negative_clips.json"
        if not hard_negative_manifest.is_file():
            run(sys.executable, "scripts/ml/mine_hard_negatives.py", "--output", hard_negative_manifest)
        build_command += ["--hard-negative-clips", hard_negative_manifest]
    run(*build_command)
    outputs = []
    for outer in range(5):
        run_id = f"{args.model}_{args.experiment}_{outer}"
        checkpoint = ROOT / "models" / "checkpoints" / run_id / "best.pt"
        training_metrics = checkpoint.with_name("metrics.json")
        report_dir = ROOT / "reports" / "ml" / args.experiment
        calibration = report_dir / f"outer_{outer}_calibration.json"
        output = report_dir / f"outer_{outer}_evaluation.json"
        outputs.append(output)
        if not (args.skip_existing and checkpoint.is_file() and training_metrics.is_file()):
            train_command: list[object] = [sys.executable, "scripts/ml/train_video.py", "--model", args.model, "--run-id", run_id, "--epochs", args.epochs, "--checkpoint-policy", "last-epoch", "--manifest", f"analysis/oof/outer_{outer}_train_calibration_clips.json"]
            if args.pair_ranking:
                train_command += ["--pair-ranking", "--ranking-weight", args.ranking_weight, "--ranking-margin", args.ranking_margin]
            run(*train_command)
        if not (args.skip_existing and calibration.is_file()):
            run(sys.executable, "scripts/ml/evaluate_full_videos.py", "--checkpoint", checkpoint, "--metrics", training_metrics, "--manifest", f"analysis/oof/outer_{outer}_calibration_dataset.json", "--output", calibration)
        if not (args.skip_existing and output.is_file()):
            run(sys.executable, "scripts/ml/evaluate_full_videos.py", "--checkpoint", checkpoint, "--metrics", training_metrics, "--manifest", f"analysis/oof/fold_{outer}_dataset.json", "--threshold-from", calibration, "--output", output)
    run(
        sys.executable, "scripts/ml/summarize_nested_oof.py", *outputs,
        "--output", ROOT / "reports" / "ml" / f"{args.experiment}_summary.json",
        "--markdown", ROOT / "reports" / "ml" / f"{args.experiment}_summary.md",
    )


if __name__ == "__main__":
    main()
