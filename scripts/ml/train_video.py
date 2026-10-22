from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.bundle import write_json
from fortebank_audit.ml.experiment_data import assert_disjoint, fingerprint, load_sources, source_identity
from fortebank_audit.ml.training import train_model


def main():
    parser = argparse.ArgumentParser(description="Train final checkpoint on four frozen folds; reserve fold 4 for calibration.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--variant", choices=["baseline", "temporal_conv1d"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/ml/cache")
    args = parser.parse_args()
    sources = load_sources(args.manifest, args.data_root, args.review)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identity = fingerprint([source_identity(row) for row in sorted(sources, key=lambda row: row["source_id"])])
    if identity != plan["dataset_sha256"]:
        raise ValueError("Dataset changed after the frozen experiment")
    config = yaml.safe_load((ROOT / "configs/ml" / (args.variant + ".yaml")).read_text())
    if config != plan["configs"][args.variant]:
        raise ValueError("Recipe changed after the experiment")
    folds = plan["folds"]
    if set(folds) != {row["source_id"] for row in sources}:
        raise ValueError("Fold plan does not cover the dataset")
    train = [row for row in sources if folds[row["source_id"]] != 4]
    calibration = [row for row in sources if folds[row["source_id"]] == 4]
    assert_disjoint(train, calibration)
    if any({row["label"] for row in part} != {0, 1} for part in (train, calibration)):
        raise ValueError("Train and calibration must both contain both classes")
    metrics = train_model(train, config, args.output, args.cache)
    write_json(args.output / "training_metrics.json", metrics)
    print(json.dumps({"checkpoint": str(args.output / "best.pt"), "train_sources": len(train), "calibration_sources": len(calibration), "calibration_fold": 4}))


if __name__ == "__main__":
    main()
