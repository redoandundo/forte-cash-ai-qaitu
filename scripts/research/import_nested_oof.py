from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "research" / "learning_lab.sqlite3"
REPORT = ROOT / "reports" / "ml" / "nested_oof_summary.json"
MANIFEST = ROOT / "analysis" / "dataset_manifest.json"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    source_map = {source["source_id"]: source for source in json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"]}
    fold_aurocs = []
    for path in sorted((ROOT / "reports" / "ml" / "nested_oof").glob("outer_*_evaluation.json")):
        fold_aurocs.append(json.loads(path.read_text(encoding="utf-8"))["video_metrics"]["auroc"])
    metrics = {
        "precision": report["metrics"]["precision"],
        "recall": report["metrics"]["recall"],
        "macro_f1": report["metrics"]["macro_f1"],
        "macro_f1_ci_low": report["bootstrap_95_ci"]["macro_f1"][0],
        "macro_f1_ci_high": report["bootstrap_95_ci"]["macro_f1"][2],
        "mean_fold_auroc": sum(fold_aurocs) / len(fold_aurocs),
        "timestamp_mae_sec": report["timestamp_metrics"]["mae_sec"],
        "within_5_sec": report["timestamp_metrics"]["within_5_sec"],
        "fp_per_min": report["metrics"]["false_positives_per_minute"],
        "source_count": report["source_count"],
        "oof_folds": report["oof_folds"],
        "threshold_selection_leak": report["threshold_selection_leak"],
        "scenario_leaks": 0,
    }
    with sqlite3.connect(DB) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute(
            """INSERT INTO experiments(run_id,model,stage,dataset_version,split_version,status,config_path,artifact_path,notes)
               VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET stage=excluded.stage,
               dataset_version=excluded.dataset_version, split_version=excluded.split_version,
               status=excluded.status, artifact_path=excluded.artifact_path, notes=excluded.notes""",
            (
                "mobilenet-nested-oof-audit-v1", "mobilenet_v3_small_temporal", "nested_oof_audit",
                "clips-v3-79-weighted", "five-fold-nested-scenario-safe-v1", "evaluated",
                "configs/model.yaml", "reports/ml/nested_oof_summary.json",
                "Honest outer-fold estimate with a separate inner fold selecting each threshold. Not a deployable model artifact.",
            ),
        )
        experiment_id = db.execute("SELECT id FROM experiments WHERE run_id=?", ("mobilenet-nested-oof-audit-v1",)).fetchone()[0]
        db.execute("DELETE FROM metrics WHERE experiment_id=?", (experiment_id,))
        db.execute("DELETE FROM error_cases WHERE experiment_id=?", (experiment_id,))
        for name, value in metrics.items():
            db.execute("INSERT INTO metrics(experiment_id,scope,name,value) VALUES(?, 'nested_oof', ?, ?)", (experiment_id, name, value))

        for row in report["videos"]:
            source = source_map[row["source_id"]]
            video_id = "/".join(f"Х-{value}" for value in source["video_ids"])
            if row["label"] != row["prediction"]:
                if row["label"] == 0:
                    paired = source["scenario_group"].startswith("pair-")
                    diagnosis = "High-confidence normal source is ranked as a violation" + (" inside a contrastive scenario pair." if paired else ".")
                    action = "Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss." if paired else "Mine the peak window as a hard negative and inspect scene-length/background shortcuts."
                    kind = "false_positive"
                else:
                    diagnosis = f"Positive source is missed; annotation confidence is {source['annotation_confidence']}."
                    action = "Review the label and event interval before retraining." if source["annotation_confidence"] != "high" else "Add the event window to the false-negative curriculum and inspect motion/ROI evidence."
                    kind = "false_negative"
                db.execute(
                    """INSERT INTO error_cases(experiment_id,video_id,error_kind,peak_time_sec,score,
                       annotation_confidence,diagnosis,proposed_action) VALUES(?,?,?,?,?,?,?,?)""",
                    (experiment_id, video_id, kind, row["peak_time_sec"], row["score"], source["annotation_confidence"], diagnosis, action),
                )
            if row["label"] == 1 and row["timestamp_error_sec"] > 5:
                db.execute(
                    """INSERT INTO error_cases(experiment_id,video_id,error_kind,peak_time_sec,score,
                       annotation_confidence,diagnosis,proposed_action) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        experiment_id, video_id, "localization_error", row["peak_time_sec"], row["score"], source["annotation_confidence"],
                        f"Peak is {row['timestamp_error_sec']:.1f} seconds from the annotated event.",
                        "Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.",
                    ),
                )
    print(json.dumps({"run_id": "mobilenet-nested-oof-audit-v1", "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
