"""Reproducible small experiments; private artifacts never belong in Git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .bundle import write_json
from .evaluation import calibration_report, evaluation_report, score_sources
from .experiment_data import fingerprint, load_sources, source_identity, split_sources, assert_disjoint
from .quality import detailed_metrics, evaluated_rows
from .training import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["development", "oof", "final", "rescore"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoints", type=Path, help="Existing outer_0...outer_4/training/best.pt for rescore")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/ml/cache"))
    args = parser.parse_args()
    sources = load_sources(args.manifest, args.data_root, args.review)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identity = fingerprint([source_identity(row) for row in sorted(sources, key=lambda row: row["source_id"])])
    if identity != plan["dataset_sha256"]:
        raise ValueError("Data differ from the frozen split plan")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else None
    if args.stage == "rescore" and args.checkpoints is None:
        raise ValueError("Rescore requires existing checkpoints")
    if args.stage != "rescore" and config is None:
        raise ValueError("Training requires an explicit frozen config")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {"dataset_sha256": identity, "folds": plan["folds"], "config": config,
               "stage": args.stage, "development_outer": 0, "development_calibration": 1,
               "final_calibration": 4, "epoch_selection": "fixed_before_evaluation",
               "limitation": "reused_development_dataset_not_a_new_external_test"})
    if args.stage == "final":
        train = [row for row in sources if plan["folds"][row["source_id"]] != 4]
        cal = [row for row in sources if plan["folds"][row["source_id"]] == 4]
        assert_disjoint(train, cal)
        metrics = train_model(train, config, args.output / "training", args.cache)
        write_json(args.output / "training_metrics.json", metrics)
        calibration = calibration_report(args.output / "training/best.pt", cal, args.cache)
        write_json(args.output / "calibration.json", calibration)
        print(json.dumps({"calibration_is_test": False, "metrics": calibration["metrics"]}))
        return
    all_rows, folds = [], []
    for outer in ([0] if args.stage == "development" else range(5)):
        train, cal, test = split_sources(sources, plan["folds"], outer, (outer + 1) % 5)
        folder = args.output / f"outer_{outer}"
        checkpoint = (args.checkpoints / f"outer_{outer}/training/best.pt") if args.stage == "rescore" else folder / "training/best.pt"
        if args.stage != "rescore":
            write_json(folder / "training_metrics.json", train_model(train, config, folder / "training", args.cache))
        else:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if set(saved["provenance"]["train_source_ids"]) != {row["source_id"] for row in train}:
                raise ValueError("Checkpoint training partition differs from frozen plan")
        calibration = calibration_report(checkpoint, cal, args.cache)
        write_json(folder / "calibration.json", calibration)
        if args.stage == "development":
            raw, _, spec = score_sources(checkpoint, cal, args.cache)
            rows = evaluated_rows(raw, calibration["threshold"], spec.stride_sec)
            result = detailed_metrics(rows, cal)
            result["calibration_is_test"] = False
            write_json(folder / "development.json", result)
        else:
            evaluation = evaluation_report(checkpoint, test, args.cache, calibration)
            write_json(folder / "evaluation.json", evaluation)
            raw, _, spec = score_sources(checkpoint, test, args.cache)
            rows = evaluated_rows(raw, calibration["threshold"], spec.stride_sec)
            result = detailed_metrics(rows, test)
            write_json(folder / "details.json", {**result, "videos": rows})
            all_rows.extend(rows)
            folds.append(result["metrics"])
        print(json.dumps({"outer": outer, "stage": args.stage, "metrics": result["metrics"]}), flush=True)
    if all_rows:
        if len(all_rows) != len(sources) or len({r["source_id"] for r in all_rows}) != len(sources):
            raise ValueError("OOF must cover every source exactly once")
        summary = detailed_metrics(all_rows, sources)
        summary["folds"] = folds
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"aggregate": summary["metrics"], "errors": summary["error_counts"]}), flush=True)


if __name__ == "__main__":
    main()
