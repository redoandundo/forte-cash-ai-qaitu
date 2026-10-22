from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.evaluation import calibration_report, evaluation_report, file_hash, pipeline_hash, promotion_decision
from fortebank_audit.ml.experiment_data import fingerprint, load_sources, make_folds, source_identity, split_sources
from fortebank_audit.ml.metrics import summarize_rows
from fortebank_audit.ml.training import train_model
from fortebank_audit.ml.preprocessing import VideoSpec


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Fixed baseline and one Conv1D candidate; separate calibration and outer folds.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "ml" / "cache")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--variants", nargs="+", choices=["baseline", "temporal_conv1d"], default=["baseline", "temporal_conv1d"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.folds) != len(set(args.folds)) or any(fold not in range(5) for fold in args.folds):
        raise ValueError("Invalid outer folds")
    sources = load_sources(args.manifest, args.data_root, args.review)
    assignments = make_folds(sources)
    configs = {name: yaml.safe_load((ROOT / "configs" / "ml" / (name + ".yaml")).read_text()) for name in ("baseline", "temporal_conv1d")}
    plan = {"dataset_sha256": fingerprint([source_identity(row) for row in sorted(sources, key=lambda row: row["source_id"])]), "folds": assignments, "configs": configs, "seed": 42, "epoch_selection": "fixed_before_outer"}
    plan_path = args.output / "frozen_plan.json"
    if plan_path.exists():
        if not args.resume or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing experiment or changed frozen plan")
    else:
        write_json(plan_path, plan)
    for variant in args.variants:
        for outer in args.folds:
            run_dir = args.output / variant / f"outer_{outer}"
            output = run_dir / "evaluation.json"
            train, cal, test = split_sources(sources, assignments, outer, (outer + 1) % 5)
            checkpoint = run_dir / "training" / "best.pt"
            if output.exists():
                if args.resume:
                    saved = json.loads(output.read_text(encoding="utf-8"))
                    cal_report = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
                    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    provenance = weights["provenance"]
                    checks = [
                        saved["checkpoint_sha256"] == cal_report["checkpoint_sha256"] == file_hash(checkpoint),
                        saved["pipeline_sha256"] == cal_report["pipeline_sha256"] == pipeline_hash(VideoSpec(**weights["spec"])),
                        provenance["config_sha256"] == fingerprint(configs[variant]),
                        provenance["dataset_sha256"] == fingerprint([source_identity(row) for row in sorted(train, key=lambda row: row["source_id"])]),
                        {row["source_id"] for row in saved["videos"]} == {row["source_id"] for row in test},
                        set(cal_report["calibration_source_ids"]) == {row["source_id"] for row in cal},
                    ]
                    if not all(checks):
                        raise ValueError("Cannot resume changed or inconsistent experiment evidence")
                    print(f"Already complete: {variant} outer {outer}", flush=True)
                    continue
                raise FileExistsError(output)
            if not checkpoint.exists():
                training = train_model(train, configs[variant], run_dir / "training", args.cache)
                write_json(run_dir / "training_metrics.json", training)
            calibration = calibration_report(checkpoint, cal, args.cache)
            write_json(run_dir / "calibration.json", calibration)
            evaluation = evaluation_report(checkpoint, test, args.cache, calibration)
            write_json(output, evaluation)
            print(json.dumps({"variant": variant, "outer": outer, **evaluation["metrics"]}), flush=True)
    summaries = {}
    for variant in args.variants:
        paths = [args.output / variant / f"outer_{fold}" / "evaluation.json" for fold in range(5)]
        if all(path.exists() for path in paths):
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            rows = [row for report in reports for row in report["videos"]]
            if len(rows) != len(sources) or len({row["source_id"] for row in rows}) != len(sources):
                raise ValueError("OOF sources must appear exactly once")
            summaries[variant] = {"metrics": summarize_rows(rows), "folds": [r["metrics"] for r in reports]}
            write_json(args.output / variant / "summary.json", summaries[variant])
    if set(summaries) == {"baseline", "temporal_conv1d"}:
        decision = promotion_decision(summaries["baseline"]["metrics"], summaries["temporal_conv1d"]["metrics"], summaries["temporal_conv1d"]["folds"])
        write_json(args.output / "decision.json", decision)
        print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
