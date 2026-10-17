PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    year INTEGER,
    kind TEXT NOT NULL,
    url TEXT NOT NULL,
    evidence_level INTEGER NOT NULL CHECK(evidence_level BETWEEN 1 AND 5),
    relevance TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY,
    hypothesis_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    mechanism TEXT NOT NULL,
    expected_effect TEXT NOT NULL,
    priority INTEGER NOT NULL,
    estimated_gpu_hours REAL NOT NULL DEFAULT 0,
    risk TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed',
    success_criteria TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypothesis_sources (
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    PRIMARY KEY (hypothesis_id, source_id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    model TEXT NOT NULL,
    stage TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    split_version TEXT NOT NULL,
    status TEXT NOT NULL,
    config_path TEXT,
    artifact_path TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    UNIQUE(experiment_id, scope, name)
);

CREATE TABLE IF NOT EXISTS error_cases (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    video_id TEXT NOT NULL,
    error_kind TEXT NOT NULL,
    peak_time_sec REAL,
    score REAL,
    annotation_confidence TEXT,
    diagnosis TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reward_events (
    id INTEGER PRIMARY KEY,
    baseline_run_id TEXT NOT NULL,
    candidate_run_id TEXT NOT NULL,
    formula_version TEXT NOT NULL,
    raw_reward REAL NOT NULL,
    confidence_factor REAL NOT NULL,
    final_reward REAL NOT NULL,
    promoted INTEGER NOT NULL,
    explanation_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(baseline_run_id, candidate_run_id, formula_version)
);

CREATE TABLE IF NOT EXISTS learning_cycles (
    id INTEGER PRIMARY KEY,
    cycle_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    entry_criteria TEXT NOT NULL,
    exit_criteria TEXT NOT NULL,
    status TEXT NOT NULL,
    order_index INTEGER NOT NULL
);

DROP VIEW IF EXISTS experiment_scorecard;
CREATE VIEW experiment_scorecard AS
SELECT e.run_id, e.model, e.stage, e.dataset_version, e.split_version,
       MAX(CASE WHEN m.name='macro_f1' THEN m.value END) AS macro_f1,
       MAX(CASE WHEN m.name='auroc' THEN m.value END) AS auroc,
       MAX(CASE WHEN m.name='precision' THEN m.value END) AS precision,
       MAX(CASE WHEN m.name='recall' THEN m.value END) AS recall,
       MAX(CASE WHEN m.name='timestamp_mae_sec' THEN m.value END) AS timestamp_mae_sec,
       MAX(CASE WHEN m.name='fp_per_min' THEN m.value END) AS fp_per_min,
       MAX(CASE WHEN m.name='p95_runtime_sec' THEN m.value END) AS p95_runtime_sec,
       MAX(CASE WHEN m.name='oof_folds' THEN m.value END) AS oof_folds,
       MAX(CASE WHEN m.name='macro_f1_ci_low' THEN m.value END) AS macro_f1_ci_low,
       MAX(CASE WHEN m.name='threshold_loo_macro_f1' THEN m.value END) AS threshold_loo_macro_f1
FROM experiments e LEFT JOIN metrics m ON m.experiment_id=e.id
GROUP BY e.id;
