"""Run one fixed MIL recipe on five pre-existing scenario-group folds."""
import argparse
import json
from pathlib import Path

import yaml

from fortebank_audit.ml.bundle import write_json
from fortebank_audit.ml.evaluation import calibration_report, evaluation_report, score_sources
from fortebank_audit.ml.experiment_data import load_sources, split_sources
from fortebank_audit.ml.quality import detailed_metrics, evaluated_rows
from fortebank_audit.ml.training import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "plan", "config", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new directory for each fixed experiment")
    sources = load_sources(args.manifest)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if set(plan["folds"].values()) != set(range(5)):
        raise ValueError("Expected five frozen folds")
    # Validate every partition before training the first model.
    splits = [split_sources(sources, plan["folds"], outer, (outer + 1) % 5) for outer in range(5)]
    write_json(args.output / "recipe.json", {"folds": plan["folds"], "config": config})
    rows, folds = [], []
    for outer, (train, cal, test) in enumerate(splits):
        folder = args.output / f"outer_{outer}"
        checkpoint = folder / "training/best.pt"
        write_json(folder / "training_metrics.json", train_model(train, config, folder / "training", args.cache))
        calibration = calibration_report(checkpoint, cal, args.cache)
        write_json(folder / "calibration.json", calibration)
        raw, _, spec = score_sources(checkpoint, test, args.cache)
        evaluated = evaluated_rows(raw, calibration["threshold"], spec.stride_sec)
        details = detailed_metrics(evaluated, test)
        write_json(folder / "details.json", {**details, "videos": evaluated})
        write_json(folder / "evaluation.json", evaluation_report(checkpoint, test, args.cache, calibration))
        rows.extend(evaluated)
        folds.append(details["metrics"])
        print({"outer": outer, "metrics": details["metrics"]}, flush=True)
    if len(rows) != len(sources) or len({row["source_id"] for row in rows}) != len(sources):
        raise ValueError("Every source must appear exactly once in outer predictions")
    summary = detailed_metrics(rows, sources)
    summary["folds"] = folds
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
