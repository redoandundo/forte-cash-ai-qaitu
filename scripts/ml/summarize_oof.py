from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_predictions import audit, best_threshold, markdown, metrics

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool outer-fold full-video predictions and audit the resulting threshold.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "analysis" / "dataset_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "ml" / "oof_summary.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "reports" / "ml" / "oof_summary.md")
    args = parser.parse_args()

    rows = []
    fold_rows = []
    seen = set()
    for fold, path in enumerate(args.inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = payload["videos"]
        overlap = seen.intersection(row["source_id"] for row in current)
        if overlap:
            raise ValueError(f"Sources occur in multiple outer folds: {sorted(overlap)}")
        seen.update(row["source_id"] for row in current)
        for row in current:
            rows.append({**row, "outer_fold": fold})

    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    threshold = best_threshold(labels, scores)
    for fold in range(len(args.inputs)):
        fold_data = [row for row in rows if row["outer_fold"] == fold]
        fold_rows.append({"fold": fold, "source_count": len(fold_data), **metrics(
            [int(row["label"]) for row in fold_data],
            [float(row["score"]) for row in fold_data],
            threshold,
        )})

    source_map = {source["source_id"]: source for source in json.loads(args.dataset.read_text(encoding="utf-8"))["sources"]}
    missing = set(source_map).difference(seen)
    if missing:
        raise ValueError(f"OOF predictions are incomplete: {sorted(missing)}")
    predictions = [int(score >= threshold) for score in scores]
    negative_minutes = sum(source_map[row["source_id"]]["duration_sec"] for row in rows if row["label"] == 0) / 60
    false_positives = sum(row["label"] == 0 and prediction == 1 for row, prediction in zip(rows, predictions))
    timestamp_errors = [float(row["timestamp_error_sec"]) for row in rows if row.get("timestamp_error_sec") is not None]
    payload = {
        "model": "mobilenet_v3_small_temporal_oof",
        "evaluation": "raw pooled score diagnostic",
        "diagnostic_only": True,
        "warning": "Scores from independently trained folds are not on a common calibrated scale. Use nested_oof_summary.json for promotion decisions.",
        "threshold": threshold,
        "threshold_selected_on": "pooled_outer_fold_predictions",
        "video_metrics": {
            **metrics(labels, scores, threshold),
            "false_positives_per_minute": false_positives / negative_minutes if negative_minutes else None,
        },
        "timestamp_metrics": {
            "mae_sec": sum(timestamp_errors) / len(timestamp_errors),
            "within_5_sec": sum(error <= 5 for error in timestamp_errors) / len(timestamp_errors),
        },
        "folds": fold_rows,
        "videos": rows,
    }
    evidence = audit(payload, 10_000, 42)
    payload["evidence_audit"] = evidence
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = markdown(evidence).replace("Evidence audit for model v3", "Raw pooled OOF diagnostic")
    report = "> **Diagnostic only.** Independently trained fold scores are not on a common calibrated scale. Use `nested_oof_summary.md` for the promotion decision.\n\n" + report
    report += "\n## Fold results at the pooled threshold\n\n| Fold | Sources | Macro F1 | Precision | Recall |\n|---:|---:|---:|---:|---:|\n"
    report += "\n".join(f"| {row['fold']} | {row['source_count']} | {row['macro_f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} |" for row in fold_rows) + "\n"
    args.markdown.write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sources": len(rows), "threshold": threshold, "metrics": payload["video_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
