from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.bundle import write_json
from fortebank_audit.ml.evaluation import calibration_report, evaluation_report
from fortebank_audit.ml.experiment_data import fingerprint, load_sources, source_identity


def main():
    parser = argparse.ArgumentParser(description="Separate threshold calibration from held-out evaluation.")
    parser.add_argument("--mode", choices=["calibrate", "evaluate"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/ml/cache")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite evaluation evidence")
    if (args.mode == "evaluate") != bool(args.calibration):
        parser.error("--calibration is required only for evaluate")
    sources = load_sources(args.manifest, args.data_root, args.review)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identity = fingerprint([source_identity(row) for row in sorted(sources, key=lambda row: row["source_id"])])
    if identity != plan["dataset_sha256"]:
        raise ValueError("Dataset changed after the frozen experiment")
    selected = [row for row in sources if plan["folds"][row["source_id"]] == args.fold]
    if args.mode == "calibrate":
        report = calibration_report(args.checkpoint, selected, args.cache)
    else:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        report = evaluation_report(args.checkpoint, selected, args.cache, calibration)
    write_json(args.output, report)
    print(json.dumps(report["metrics"]))


if __name__ == "__main__":
    main()
