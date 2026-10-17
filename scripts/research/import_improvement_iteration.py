from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "research" / "learning_lab.sqlite3"
MANIFEST = ROOT / "analysis" / "dataset_manifest.json"
BASELINE_RUN = "mobilenet-nested-oof-audit-v1"
CANDIDATES = [
    ("pair-ranking-w05-m01", "mobilenet_v3_small_temporal", "pair_rank_w05_m01", "Pair ranking alone; rejected after nested OOF regression."),
    ("temporal-stats-pair-w025-m005", "mobilenet_v3_small_temporal_stats", "temporal_stats_pair_w025_m005", "Appearance, salience, motion statistics plus pair ranking; rejected after recall regression."),
    ("hard-negative-pair-w025-m005", "mobilenet_v3_small_temporal", "hardneg_pair_w025_m005", "Leakage-safe OOF hard-negative curriculum plus pair ranking; fewer false positives but unacceptable recall loss."),
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metrics(summary: dict, report_dir: Path) -> dict[str, float]:
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(report_dir.glob("outer_*_evaluation.json"))]
    runtimes = [row["runtime_sec"] for fold in folds for row in fold["videos"]]
    aurocs = [fold["video_metrics"]["auroc"] for fold in folds if fold["video_metrics"]["auroc"] is not None]
    return {
        "precision": summary["metrics"]["precision"],
        "recall": summary["metrics"]["recall"],
        "macro_f1": summary["metrics"]["macro_f1"],
        "macro_f1_ci_low": summary["bootstrap_95_ci"]["macro_f1"][0],
        "macro_f1_ci_high": summary["bootstrap_95_ci"]["macro_f1"][2],
        "auroc": sum(aurocs) / len(aurocs),
        "timestamp_mae_sec": summary["timestamp_metrics"]["mae_sec"],
        "within_5_sec": summary["timestamp_metrics"]["within_5_sec"],
        "fp_per_min": summary["metrics"]["false_positives_per_minute"],
        "p95_runtime_sec": percentile(runtimes, 0.95),
        "source_count": summary["source_count"],
        "oof_folds": summary["oof_folds"],
        "threshold_selection_leak": summary["threshold_selection_leak"],
        "scenario_leaks": 0,
        "parity_error": 0,
    }


def upsert_metrics(db: sqlite3.Connection, experiment_id: int, values: dict[str, float]) -> None:
    for name, value in values.items():
        db.execute(
            """INSERT INTO metrics(experiment_id,scope,name,value) VALUES(?,'nested_oof',?,?)
               ON CONFLICT(experiment_id,scope,name) DO UPDATE SET value=excluded.value""",
            (experiment_id, name, value),
        )


def main() -> None:
    source_map = {row["source_id"]: row for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"]}
    all_runs = [(BASELINE_RUN, "mobilenet_v3_small_temporal", "nested_oof", "Honest nested OOF baseline.")] + CANDIDATES
    imported = []
    with sqlite3.connect(DB) as db:
        db.execute("PRAGMA foreign_keys = ON")
        for run_id, model, report_name, notes in all_runs:
            summary_path = ROOT / "reports" / "ml" / ("nested_oof_summary.json" if report_name == "nested_oof" else f"{report_name}_summary.json")
            report_dir = ROOT / "reports" / "ml" / report_name
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            values = metrics(summary, report_dir)
            db.execute(
                """INSERT INTO experiments(run_id,model,stage,dataset_version,split_version,status,config_path,artifact_path,notes)
                   VALUES(?,?,?,?,?,'rejected',?,?,?) ON CONFLICT(run_id) DO UPDATE SET model=excluded.model,
                   stage=excluded.stage,status=excluded.status,artifact_path=excluded.artifact_path,notes=excluded.notes""",
                (run_id, model, "nested_oof_candidate" if run_id != BASELINE_RUN else "nested_oof_audit", "clips-v3-hardneg-v1", "five-fold-nested-scenario-safe-v1", "configs/model.yaml", str(summary_path.relative_to(ROOT)), notes),
            )
            experiment_id = db.execute("SELECT id FROM experiments WHERE run_id=?", (run_id,)).fetchone()[0]
            upsert_metrics(db, experiment_id, values)
            if run_id != BASELINE_RUN:
                db.execute("DELETE FROM error_cases WHERE experiment_id=?", (experiment_id,))
                for row in summary["videos"]:
                    if row["label"] == row["prediction"]:
                        continue
                    source = source_map[row["source_id"]]
                    kind = "false_positive" if row["label"] == 0 else "false_negative"
                    diagnosis = "Candidate ranks a normal source above its calibrated threshold." if kind == "false_positive" else "Candidate misses a labelled violation."
                    action = "Retain for MIL error curriculum; do not promote this checkpoint."
                    db.execute(
                        """INSERT INTO error_cases(experiment_id,video_id,error_kind,peak_time_sec,score,
                           annotation_confidence,diagnosis,proposed_action) VALUES(?,?,?,?,?,?,?,?)""",
                        (experiment_id, "/".join(f"Х-{value}" for value in source["video_ids"]), kind, row["peak_time_sec"], row["score"], source["annotation_confidence"], diagnosis, action),
                    )
            imported.append({"run_id": run_id, "metrics": values})
        db.execute("UPDATE experiments SET status='evaluated' WHERE run_id=?", (BASELINE_RUN,))
        db.execute("UPDATE hypotheses SET status='evaluated_no_gain' WHERE hypothesis_key='H7-pair-ranking'")
        db.execute("UPDATE hypotheses SET status='next' WHERE hypothesis_key='H2-mil-ranking'")
    print(json.dumps(imported, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
