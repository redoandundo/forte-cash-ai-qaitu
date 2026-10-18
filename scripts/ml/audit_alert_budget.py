"""Check fixed quality constraints on separate calibration splits."""
import argparse
import json
from pathlib import Path

from fortebank_audit.ml.bundle import write_json
from fortebank_audit.ml.evaluation import file_hash, score_sources
from fortebank_audit.ml.experiment_data import load_sources, split_sources
from fortebank_audit.ml.operating_point import calibrate_alert_budget
from fortebank_audit.ml.quality import detailed_metrics, evaluated_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "plan", "run", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an operating point audit")
    sources = load_sources(args.manifest)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    folds, all_rows = [], []
    for outer in range(5):
        _, cal, test = split_sources(sources, plan["folds"], outer, (outer + 1) % 5)
        folder = args.run / f"outer_{outer}"
        checkpoint = folder / "training/best.pt"
        calibration = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
        if (calibration["checkpoint_sha256"] != file_hash(checkpoint)
                or set(calibration["calibration_source_ids"]) != {row["source_id"] for row in cal}):
            raise ValueError("Changed calibration/checkpoint provenance")
        raw, weights, spec = score_sources(checkpoint, cal, args.cache)
        if set(weights["provenance"]["train_source_ids"]) & {row["source_id"] for row in cal + test}:
            raise ValueError("Training overlaps calibration or test")
        result = calibrate_alert_budget(raw, spec.stride_sec)
        write_json(args.output / f"outer_{outer}_calibration_scores.json", {"videos": raw})
        if result["feasible"]:
            outer_raw, _, _ = score_sources(checkpoint, test, args.cache)
            evaluated = evaluated_rows(outer_raw, result["selected"]["threshold"], spec.stride_sec)
            result["outer_metrics"] = detailed_metrics(evaluated, test)["metrics"]
            all_rows.extend(evaluated)
        folds.append(result)
        print({"fold": outer, "feasible": result["feasible"],
               "best_calibration_recall_under_budget": result["best_recall_under_false_alert_budget"]}, flush=True)
    summary = {"scope": "reused_development_data_calibration_only_selection",
               "policy": "recall_at_least_0.75_and_false_events_per_minute_at_most_0.6",
               "folds": folds, "all_folds_feasible": all(row["feasible"] for row in folds),
               "deployment_changed": False}
    if len(all_rows) == len(sources):
        summary["outer_metrics"] = detailed_metrics(all_rows, sources)["metrics"]
    else:
        summary["outer_metrics"] = None
        summary["reason"] = "Do not report a cherry-picked aggregate omitting infeasible folds"
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
