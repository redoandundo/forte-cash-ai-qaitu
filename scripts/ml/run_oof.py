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
    parser = argparse.ArgumentParser(description="Run leakage-safe five-fold full-video OOF evaluation.")
    parser.add_argument("--epochs", type=int, default=1, help="Fixed before looking at outer folds; v3 selected epoch 1.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    run(sys.executable, "scripts/ml/build_oof_manifests.py")
    outputs = []
    for fold in range(5):
        run_id = f"mobilenet_oof_fold_{fold}"
        checkpoint = ROOT / "models" / "checkpoints" / run_id / "best.pt"
        training_metrics = checkpoint.with_name("metrics.json")
        output = ROOT / "reports" / "ml" / "oof" / f"fold_{fold}_full_video.json"
        outputs.append(output)
        if not (args.skip_existing and checkpoint.is_file() and training_metrics.is_file()):
            run(
                sys.executable, "scripts/ml/train_video.py",
                "--model", "mobilenet_v3_small_temporal",
                "--run-id", run_id,
                "--epochs", args.epochs,
                "--checkpoint-policy", "last-epoch",
                "--manifest", f"analysis/oof/fold_{fold}_clips.json",
            )
        if not (args.skip_existing and output.is_file()):
            run(
                sys.executable, "scripts/ml/evaluate_full_videos.py",
                "--checkpoint", checkpoint,
                "--metrics", training_metrics,
                "--manifest", f"analysis/oof/fold_{fold}_dataset.json",
                "--output", output,
            )
    run(sys.executable, "scripts/ml/summarize_oof.py", *outputs)


if __name__ == "__main__":
    main()
