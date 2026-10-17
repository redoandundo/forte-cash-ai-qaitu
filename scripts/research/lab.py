from __future__ import annotations

import argparse
import html
import json
import math
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
DB_PATH = RESEARCH / "learning_lab.sqlite3"
SEED = RESEARCH / "seed"


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def load(name: str):
    return json.loads((SEED / name).read_text(encoding="utf-8"))


def initialize() -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript((RESEARCH / "schema.sql").read_text(encoding="utf-8"))
        for source in load("sources.json"):
            db.execute(
                """INSERT INTO sources(source_key,title,year,kind,url,evidence_level,relevance,notes)
                   VALUES(:source_key,:title,:year,:kind,:url,:evidence_level,:relevance,:notes)
                   ON CONFLICT(source_key) DO UPDATE SET title=excluded.title, year=excluded.year,
                   kind=excluded.kind, url=excluded.url, evidence_level=excluded.evidence_level,
                   relevance=excluded.relevance, notes=excluded.notes""",
                source,
            )
        for hypothesis in load("hypotheses.json"):
            source_keys = hypothesis.pop("sources")
            db.execute(
                """INSERT INTO hypotheses(hypothesis_key,title,mechanism,expected_effect,priority,
                   estimated_gpu_hours,risk,status,success_criteria)
                   VALUES(:hypothesis_key,:title,:mechanism,:expected_effect,:priority,
                   :estimated_gpu_hours,:risk,:status,:success_criteria)
                   ON CONFLICT(hypothesis_key) DO UPDATE SET title=excluded.title,
                   mechanism=excluded.mechanism, expected_effect=excluded.expected_effect,
                   priority=excluded.priority, estimated_gpu_hours=excluded.estimated_gpu_hours,
                   risk=excluded.risk, status=excluded.status, success_criteria=excluded.success_criteria""",
                hypothesis,
            )
            hypothesis_id = db.execute("SELECT id FROM hypotheses WHERE hypothesis_key=?", (hypothesis["hypothesis_key"],)).fetchone()[0]
            db.execute("DELETE FROM hypothesis_sources WHERE hypothesis_id=?", (hypothesis_id,))
            for source_key in source_keys:
                source_id = db.execute("SELECT id FROM sources WHERE source_key=?", (source_key,)).fetchone()[0]
                db.execute("INSERT INTO hypothesis_sources(hypothesis_id,source_id) VALUES(?,?)", (hypothesis_id, source_id))
        for cycle in load("cycles.json"):
            db.execute(
                """INSERT INTO learning_cycles(cycle_key,title,objective,entry_criteria,exit_criteria,status,order_index)
                   VALUES(:cycle_key,:title,:objective,:entry_criteria,:exit_criteria,:status,:order_index)
                   ON CONFLICT(cycle_key) DO UPDATE SET title=excluded.title, objective=excluded.objective,
                   entry_criteria=excluded.entry_criteria, exit_criteria=excluded.exit_criteria,
                   status=excluded.status, order_index=excluded.order_index""",
                cycle,
            )
        for experiment in load("experiments.json"):
            metrics = experiment.pop("metrics")
            errors = experiment.pop("errors", [])
            db.execute(
                """INSERT INTO experiments(run_id,model,stage,dataset_version,split_version,status,config_path,artifact_path,notes)
                   VALUES(:run_id,:model,:stage,:dataset_version,:split_version,:status,:config_path,:artifact_path,:notes)
                   ON CONFLICT(run_id) DO UPDATE SET model=excluded.model, stage=excluded.stage,
                   dataset_version=excluded.dataset_version, split_version=excluded.split_version,
                   status=excluded.status, config_path=excluded.config_path,
                   artifact_path=excluded.artifact_path, notes=excluded.notes""",
                experiment,
            )
            experiment_id = db.execute("SELECT id FROM experiments WHERE run_id=?", (experiment["run_id"],)).fetchone()[0]
            for name, value in metrics.items():
                db.execute(
                    """INSERT INTO metrics(experiment_id,scope,name,value) VALUES(?, 'full_video', ?, ?)
                       ON CONFLICT(experiment_id,scope,name) DO UPDATE SET value=excluded.value""",
                    (experiment_id, name, value),
                )
            db.execute("DELETE FROM error_cases WHERE experiment_id=?", (experiment_id,))
            for error in errors:
                db.execute(
                    """INSERT INTO error_cases(experiment_id,video_id,error_kind,peak_time_sec,score,
                       annotation_confidence,diagnosis,proposed_action)
                       VALUES(:experiment_id,:video_id,:error_kind,:peak_time_sec,:score,
                       :annotation_confidence,:diagnosis,:proposed_action)""",
                    {"experiment_id": experiment_id, **error},
                )
    print(f"Initialized {DB_PATH}")


def metric_map(db: sqlite3.Connection, run_id: str) -> dict[str, float]:
    rows = db.execute(
        """SELECT m.name,m.value FROM metrics m JOIN experiments e ON e.id=m.experiment_id
           WHERE e.run_id=? ORDER BY CASE m.scope WHEN 'nested_oof' THEN 0 ELSE 1 END""",
        (run_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"Unknown run or no metrics: {run_id}")
    result = {}
    for row in rows:
        result.setdefault(row["name"], row["value"])
    return result


def relative_improvement(baseline: float, candidate: float) -> float:
    return max(-1.0, min(1.0, (baseline - candidate) / max(abs(baseline), 1e-9)))


def calculate_reward(baseline_run: str, candidate_run: str) -> dict:
    policy = json.loads((RESEARCH / "reward_policy.json").read_text(encoding="utf-8"))
    weights = policy["weights"]
    hard_penalties = policy["hard_penalties"]
    with connect() as db:
        baseline = metric_map(db, baseline_run)
        candidate = metric_map(db, candidate_run)
        components = {
            "macro_f1": weights["macro_f1_delta"] * (candidate["macro_f1"] - baseline["macro_f1"]),
            "auroc": weights["auroc_delta"] * (candidate["auroc"] - baseline["auroc"]),
            "recall": weights["recall_delta"] * (candidate["recall"] - baseline["recall"]),
            "within_5_sec": weights["within_5_sec_delta"] * (candidate["within_5_sec"] - baseline["within_5_sec"]),
            "timestamp_mae": weights["timestamp_mae_relative_improvement"] * relative_improvement(baseline["timestamp_mae_sec"], candidate["timestamp_mae_sec"]),
            "false_positives": weights["fp_per_min_relative_improvement"] * relative_improvement(baseline["fp_per_min"], candidate["fp_per_min"]),
            "latency": weights["p95_runtime_relative_improvement"] * relative_improvement(baseline["p95_runtime_sec"], candidate["p95_runtime_sec"]),
        }
        penalties = {}
        if candidate.get("scenario_leaks", 0) > 0:
            penalties["scenario_leakage"] = hard_penalties["scenario_leaks_above_zero"]
        if candidate.get("parity_error", 1) > 1e-4:
            penalties["onnx_parity"] = hard_penalties["onnx_parity_error_above_1e-4"]
        if candidate.get("source_count", 0) < 8:
            penalties["validation_size"] = hard_penalties["validation_sources_below_8"]
        if candidate.get("oof_folds", 0) < 5:
            penalties["oof_coverage"] = hard_penalties["oof_folds_below_5"]
        if candidate.get("threshold_selection_leak", 1) > 0:
            penalties["threshold_selection_leak"] = hard_penalties["threshold_selected_on_evaluation_set"]
        raw_reward = sum(components.values())
        sample_confidence = min(1.0, math.sqrt(candidate.get("source_count", 0) / 40))
        oof_confidence = min(1.0, candidate.get("oof_folds", 0) / 5)
        confidence = sample_confidence * oof_confidence
        final_reward = raw_reward * confidence + sum(penalties.values())
        gates = {
            "no_scenario_leakage": candidate.get("scenario_leaks", 0) == 0,
            "onnx_parity": candidate.get("parity_error", 1) <= 1e-4,
            "macro_f1_not_worse": candidate["macro_f1"] >= baseline["macro_f1"],
            "fp_per_min_not_worse": candidate["fp_per_min"] <= baseline["fp_per_min"],
            "recall_at_least_0_75": candidate["recall"] >= 0.75,
            "validation_sources_at_least_8": candidate.get("source_count", 0) >= 8,
            "five_outer_folds": candidate.get("oof_folds", 0) >= 5,
            "threshold_not_selected_on_evaluation_set": candidate.get("threshold_selection_leak", 1) == 0,
            "macro_f1_ci_low_at_least_0_5": candidate.get("macro_f1_ci_low", 0) >= 0.5,
        }
        promoted = all(gates.values()) and final_reward > 0
        explanation = {"components": components, "penalties": penalties, "gates": gates, "policy": policy["version"]}
        db.execute(
            """INSERT INTO reward_events(baseline_run_id,candidate_run_id,formula_version,raw_reward,
               confidence_factor,final_reward,promoted,explanation_json) VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(baseline_run_id,candidate_run_id,formula_version) DO UPDATE SET
               raw_reward=excluded.raw_reward, confidence_factor=excluded.confidence_factor,
               final_reward=excluded.final_reward, promoted=excluded.promoted,
               explanation_json=excluded.explanation_json, created_at=CURRENT_TIMESTAMP""",
            (baseline_run, candidate_run, policy["version"], raw_reward, confidence, final_reward, int(promoted), json.dumps(explanation, ensure_ascii=False)),
        )
    result = {"baseline": baseline_run, "candidate": candidate_run, "raw_reward": raw_reward, "confidence_factor": confidence, "final_reward": final_reward, "promoted": promoted, **explanation}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def recommend() -> Path:
    with connect() as db:
        cycle = db.execute("SELECT * FROM learning_cycles WHERE status='next' ORDER BY order_index LIMIT 1").fetchone()
        hypotheses = db.execute(
            """SELECT h.*, COUNT(hs.source_id) evidence_count FROM hypotheses h
               LEFT JOIN hypothesis_sources hs ON hs.hypothesis_id=h.id
               WHERE h.status IN ('next','proposed') GROUP BY h.id
               ORDER BY CASE h.status WHEN 'next' THEN 0 ELSE 1 END, h.priority DESC,
               h.estimated_gpu_hours ASC LIMIT 4"""
        ).fetchall()
        errors = db.execute(
            """SELECT ec.* FROM error_cases ec JOIN experiments e ON e.id=ec.experiment_id
               WHERE e.status IN ('selected','evaluated') AND ec.resolved=0 ORDER BY ec.error_kind,ec.video_id"""
        ).fetchall()
    lines = ["# Next learning stage", ""]
    if cycle:
        lines += [f"## {cycle['title']}", "", cycle["objective"], "", f"**Exit criteria:** {cycle['exit_criteria']}", ""]
    lines += ["## Ranked experiments", ""]
    for index, item in enumerate(hypotheses, 1):
        lines += [f"{index}. **{item['title']}** (`{item['hypothesis_key']}`)", f"   - Mechanism: {item['mechanism']}", f"   - Success: {item['success_criteria']}", f"   - Evidence links: {item['evidence_count']}; estimate: {item['estimated_gpu_hours']} GPU h", ""]
    lines += ["## Unresolved error curriculum", ""]
    for item in errors:
        lines += [f"- `{item['video_id']}` — **{item['error_kind']}**: {item['diagnosis']}", f"  Next action: {item['proposed_action']}"]
    lines += ["", "## Promotion rule", "", "A candidate replaces the current model only when reward is positive and every guardrail passes: five scenario-safe outer folds, no threshold-selection leakage, bootstrap macro-F1 lower bound ≥ 0.50, no scenario leakage, ONNX parity, macro F1 and false-positive non-regression, and recall ≥ 0.75."]
    target = RESEARCH / "NEXT_STAGE.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)
    return target


def dashboard() -> Path:
    with connect() as db:
        runs = db.execute("SELECT * FROM experiment_scorecard ORDER BY macro_f1 DESC").fetchall()
        rewards = db.execute("SELECT * FROM reward_events ORDER BY created_at DESC").fetchall()
        hypotheses = db.execute("SELECT hypothesis_key,title,priority,status,estimated_gpu_hours FROM hypotheses ORDER BY priority DESC").fetchall()
        cycles = db.execute("SELECT cycle_key,title,status,objective FROM learning_cycles ORDER BY order_index").fetchall()
        errors = db.execute("SELECT video_id,error_kind,diagnosis,proposed_action FROM error_cases WHERE resolved=0").fetchall()

    def table(rows, columns):
        head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[key] if row[key] is not None else ''))}</td>" for key, _ in columns) + "</tr>" for row in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    page = f"""<!doctype html><html lang='ru'><meta charset='utf-8'><title>Qaitu ML Learning Lab</title>
<style>body{{font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#0d1117;color:#e6edf3}}h1,h2{{color:#58a6ff}}table{{width:100%;border-collapse:collapse;margin:12px 0 28px}}th,td{{border:1px solid #30363d;padding:8px;text-align:left}}th{{background:#161b22}}.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin:16px 0}}code{{color:#79c0ff}}</style>
<body><h1>Qaitu ML Learning Lab</h1><p>Локальное пространство доказательств, экспериментов, ошибок и наград.</p>
<div class='card'><h2>Experiment scorecard</h2>{table(runs,[('run_id','Run'),('model','Model'),('macro_f1','Macro F1'),('macro_f1_ci_low','F1 CI low'),('threshold_loo_macro_f1','LOO threshold F1'),('oof_folds','OOF folds'),('auroc','AUROC'),('precision','Precision'),('recall','Recall'),('fp_per_min','FP/min'),('timestamp_mae_sec','Timestamp MAE')])}</div>
<div class='card'><h2>Reward ledger</h2>{table(rewards,[('baseline_run_id','Baseline'),('candidate_run_id','Candidate'),('raw_reward','Raw'),('confidence_factor','Confidence'),('final_reward','Final'),('promoted','Promoted')])}</div>
<div class='card'><h2>Learning cycles</h2>{table(cycles,[('cycle_key','Cycle'),('title','Title'),('status','Status'),('objective','Objective')])}</div>
<div class='card'><h2>Hypothesis queue</h2>{table(hypotheses,[('hypothesis_key','ID'),('title','Hypothesis'),('priority','Priority'),('status','Status'),('estimated_gpu_hours','GPU h')])}</div>
<div class='card'><h2>Error curriculum</h2>{table(errors,[('video_id','Video'),('error_kind','Error'),('diagnosis','Diagnosis'),('proposed_action','Next action')])}</div></body></html>"""
    target = RESEARCH / "dashboard.html"
    target.write_text(page, encoding="utf-8")
    print(target)
    return target


def summary() -> None:
    with connect() as db:
        counts = {name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in ("sources", "hypotheses", "experiments", "metrics", "error_cases", "reward_events", "learning_cycles")}
        latest = db.execute("SELECT baseline_run_id,candidate_run_id,final_reward,promoted FROM reward_events ORDER BY created_at DESC LIMIT 1").fetchone()
    print(json.dumps({"database": str(DB_PATH), "counts": counts, "latest_reward": dict(latest) if latest else None}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local evidence, experiment, reward, and curriculum database.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    reward_parser = sub.add_parser("reward")
    reward_parser.add_argument("--baseline", required=True)
    reward_parser.add_argument("--candidate", required=True)
    sub.add_parser("recommend")
    sub.add_parser("dashboard")
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.command == "init":
        initialize()
    elif args.command == "reward":
        calculate_reward(args.baseline, args.candidate)
    elif args.command == "recommend":
        recommend()
    elif args.command == "dashboard":
        dashboard()
    else:
        summary()


if __name__ == "__main__":
    main()
