from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_predictions import metrics, stratified_bootstrap

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize nested OOF decisions made with inner-fold thresholds.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "analysis" / "dataset_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "ml" / "nested_oof_summary.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "reports" / "ml" / "nested_oof_summary.md")
    args = parser.parse_args()

    rows = []
    folds = []
    seen = set()
    for outer_fold, path in enumerate(args.inputs):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not str(report.get("threshold_selected_on", "")).startswith("external_calibration:"):
            raise ValueError(f"Outer fold {outer_fold} did not use an external calibration threshold")
        current = report["videos"]
        overlap = seen.intersection(row["source_id"] for row in current)
        if overlap:
            raise ValueError(f"Sources occur in multiple outer folds: {sorted(overlap)}")
        seen.update(row["source_id"] for row in current)
        rows.extend({**row, "outer_fold": outer_fold} for row in current)
        labels = [int(row["label"]) for row in current]
        decisions = [int(row["prediction"]) for row in current]
        fold_metrics = metrics(labels, [float(value) for value in decisions], 0.5)
        fold_metrics.pop("threshold")
        folds.append({
            **fold_metrics,
            "outer_fold": outer_fold,
            "calibration_threshold": report["threshold"],
            "source_count": len(current),
        })

    source_map = {source["source_id"]: source for source in json.loads(args.dataset.read_text(encoding="utf-8"))["sources"]}
    if set(source_map) != seen:
        raise ValueError(f"Expected {len(source_map)} unique sources, got {len(seen)}")
    labels = [int(row["label"]) for row in rows]
    decisions = [int(row["prediction"]) for row in rows]
    decision_scores = [float(value) for value in decisions]
    result_metrics = metrics(labels, decision_scores, 0.5)
    result_metrics.pop("threshold")
    ci = stratified_bootstrap(labels, decision_scores, 0.5, 10_000, 42)
    negative_minutes = sum(source_map[row["source_id"]]["duration_sec"] for row in rows if row["label"] == 0) / 60
    false_positives = sum(row["label"] == 0 and row["prediction"] == 1 for row in rows)
    timestamp_errors = [float(row["timestamp_error_sec"]) for row in rows if row.get("timestamp_error_sec") is not None]
    result = {
        "evaluation": "five-fold nested scenario-group OOF",
        "source_count": len(rows),
        "oof_folds": len(args.inputs),
        "threshold_selection_leak": 0,
        "metrics": {**result_metrics, "false_positives_per_minute": false_positives / negative_minutes},
        "bootstrap_95_ci": ci,
        "timestamp_metrics": {
            "mae_sec": sum(timestamp_errors) / len(timestamp_errors),
            "within_5_sec": sum(error <= 5 for error in timestamp_errors) / len(timestamp_errors),
        },
        "folds": folds,
        "videos": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Nested OOF report", "", "Date: 2026-10-17", "",
        "Each outer fold was evaluated once with a threshold selected on a different inner calibration fold. Scenario groups never cross train, calibration, and outer evaluation.", "",
        "## Result", "", "| Metric | Value |", "|---|---:|",
        f"| Sources | {len(rows)} |", f"| Macro F1 | {result_metrics['macro_f1']:.3f} |",
        f"| Macro F1 bootstrap 95% interval | {ci['macro_f1'][0]:.3f}–{ci['macro_f1'][2]:.3f} |",
        f"| Precision | {result_metrics['precision']:.3f} |", f"| Recall | {result_metrics['recall']:.3f} |",
        f"| False positives/minute | {result['metrics']['false_positives_per_minute']:.3f} |",
        f"| Timestamp MAE | {result['timestamp_metrics']['mae_sec']:.2f} s |",
        f"| Within ±5 seconds | {result['timestamp_metrics']['within_5_sec']:.3f} |", "",
        "## Fold stability", "", "| Outer fold | Calibration threshold | Macro F1 | Precision | Recall |", "|---:|---:|---:|---:|---:|",
        *[f"| {row['outer_fold']} | {row['calibration_threshold']:.6f} | {row['macro_f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} |" for row in folds], "",
        "## Interpretation", "",
        "This is the current honest generalization estimate. The five thresholds differ because each fold has a separately trained model. A deployable final threshold still needs calibration for the single final model.", "",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"], "ci": ci}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
