from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .meta_analysis import MetaDecision, MetaFeatures


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    video_sha256 TEXT NOT NULL,
    video_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    features_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    created_at TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (
        outcome IN ('confirmed_positive', 'confirmed_negative', 'uncertain')
    ),
    corrected_timecode_sec REAL,
    error_type TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS reviews_run_id_idx ON reviews(run_id);
"""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperienceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def record_run(
        self,
        *,
        run_id: str,
        video_path: str | Path,
        model_version: str,
        model_sha256: str,
        config: dict[str, Any],
        result: dict[str, Any],
        decision: MetaDecision,
        features: MetaFeatures,
    ) -> None:
        video_path = Path(video_path)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, video_sha256, video_name, model_version,
                    model_sha256, config_json, result_json, decision_json, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    file_sha256(video_path),
                    video_path.name,
                    model_version,
                    model_sha256,
                    _canonical_json(config),
                    _canonical_json(result),
                    _canonical_json(decision.as_dict()),
                    _canonical_json(features.as_dict()),
                ),
            )

    def add_review(
        self,
        run_id: str,
        *,
        outcome: str,
        corrected_timecode_sec: float | None = None,
        error_type: str | None = None,
        notes: str | None = None,
    ) -> None:
        if outcome not in {"confirmed_positive", "confirmed_negative", "uncertain"}:
            raise ValueError("Unsupported review outcome")
        if corrected_timecode_sec is not None and corrected_timecode_sec < 0:
            raise ValueError("corrected_timecode_sec must be non-negative")
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            connection.execute(
                """
                INSERT INTO reviews (
                    run_id, created_at, outcome, corrected_timecode_sec, error_type, notes
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    outcome,
                    corrected_timecode_sec,
                    error_type,
                    notes,
                ),
            )

    def nearest_reviewed(
        self,
        features: MetaFeatures,
        *,
        model_version: str,
        limit: int,
        max_distance: float,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.run_id, r.features_json, v.outcome
                FROM runs r
                JOIN reviews v ON v.run_id = r.run_id
                WHERE r.model_version = ?
                  AND v.review_id IN (
                      SELECT MAX(v2.review_id)
                      FROM reviews v2
                      JOIN runs r2 ON r2.run_id = v2.run_id
                      WHERE r2.model_version = ?
                      GROUP BY r2.video_sha256
                  )
                """,
                (model_version, model_version),
            ).fetchall()
        target = features.as_dict()
        ranked = []
        for row in rows:
            candidate = json.loads(row["features_json"])
            distance = math.sqrt(
                sum((float(target[key]) - float(candidate[key])) ** 2 for key in target)
            )
            if distance <= max_distance:
                ranked.append(
                    {"run_id": row["run_id"], "outcome": row["outcome"], "distance": distance}
                )
        return sorted(ranked, key=lambda item: (item["distance"], item["run_id"]))[:limit]

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            runs = int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
            reviewed = int(
                connection.execute("SELECT COUNT(DISTINCT run_id) FROM reviews").fetchone()[0]
            )
            outcomes = {
                row["outcome"]: int(row["count"])
                for row in connection.execute(
                    "SELECT outcome, COUNT(*) AS count FROM reviews GROUP BY outcome"
                )
            }
        return {"runs": runs, "reviewed_runs": reviewed, "review_outcomes": outcomes}

    def improvement_plan(self, *, minimum_unique_videos: int = 20) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.video_sha256, v.outcome, v.error_type
                FROM reviews v
                JOIN runs r ON r.run_id = v.run_id
                WHERE v.review_id IN (
                    SELECT MAX(v2.review_id)
                    FROM reviews v2
                    JOIN runs r2 ON r2.run_id = v2.run_id
                    GROUP BY r2.video_sha256
                )
                """
            ).fetchall()
        error_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        for row in rows:
            outcome = str(row["outcome"])
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            if row["error_type"]:
                error_type = str(row["error_type"])
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        action_map = {
            "false_positive": "mine_reviewed_hard_negative_windows",
            "localization_error": "train_or_calibrate_temporal_localization",
            "false_negative": "inspect_missed_positive_context_and_recall",
            "decode_error": "repair_video_ingestion_before_model_work",
        }
        ranked_actions = [
            {"action": action_map.get(error_type, "inspect_unclassified_error_slice"), "evidence": count}
            for error_type, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        unique_videos = len(rows)
        positive_count = outcome_counts.get("confirmed_positive", 0)
        negative_count = outcome_counts.get("confirmed_negative", 0)
        ready = (
            unique_videos >= minimum_unique_videos
            and positive_count >= 5
            and negative_count >= 5
        )
        return {
            "unique_reviewed_videos": unique_videos,
            "minimum_unique_videos": minimum_unique_videos,
            "outcomes": outcome_counts,
            "error_types": error_counts,
            "ranked_actions": ranked_actions,
            "dataset_ready_for_challenger_training": ready,
            "automatic_weight_update_allowed": False,
            "next_gate": (
                "train_and_evaluate_offline_challenger"
                if ready
                else "collect_more_human_verified_context"
            ),
        }
