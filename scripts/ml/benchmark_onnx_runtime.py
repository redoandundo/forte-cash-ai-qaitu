from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.onnx_runtime import provider, score_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the production ONNX path on all unique source videos.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best_model.onnx")
    parser.add_argument("--manifest", type=Path, default=ROOT / "analysis" / "dataset_manifest.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "reports" / "mobilenet_balanced_v3_full_video.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "ml" / "onnx_speed_benchmark.json")
    args = parser.parse_args()

    sources = json.loads(args.manifest.read_text(encoding="utf-8"))["sources"]
    sources = sorted(sources, key=lambda row: row["duration_sec"], reverse=True)
    rows = []
    wall_started = time.perf_counter()
    for source in sources:
        started = time.perf_counter()
        windows, threshold = score_video(Path(source["path"]), args.model)
        runtime = time.perf_counter() - started
        peak = max(windows, key=lambda item: item.score)
        row = {
            "source_id": source["source_id"],
            "video_ids": source["video_ids"],
            "duration_sec": source["duration_sec"],
            "runtime_sec": runtime,
            "realtime_factor": source["duration_sec"] / runtime,
            "windows": len(windows),
            "label": source["label"],
            "score": peak.score,
            "prediction": int(peak.score >= threshold),
            "peak_time_sec": peak.timestamp_sec,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    runtimes = np.asarray([row["runtime_sec"] for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    predictions = np.asarray([row["prediction"] for row in rows])
    by_source = {row["source_id"]: row for row in rows}
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    comparisons = []
    for old in reference["videos"]:
        new = by_source[old["source_id"]]
        comparisons.append({
            "source_id": old["source_id"],
            "old_score": old["score"],
            "new_score": new["score"],
            "absolute_score_delta": abs(old["score"] - new["score"]),
            "old_prediction": old["prediction"],
            "new_prediction": new["prediction"],
        })
    result = {
        "model": args.model.name,
        "provider": provider(args.model),
        "source_count": len(rows),
        "total_video_minutes": sum(row["duration_sec"] for row in rows) / 60,
        "total_wall_sec": time.perf_counter() - wall_started,
        "mean_runtime_sec": float(runtimes.mean()),
        "p95_runtime_sec": float(np.percentile(runtimes, 95)),
        "max_runtime_sec": float(runtimes.max()),
        "slowest_video_ids": rows[int(runtimes.argmax())]["video_ids"],
        "five_minute_limit_sec": 300,
        "meets_five_minute_limit": bool(runtimes.max() < 300),
        "margin_to_five_minute_limit": float(300 / runtimes.max()),
        "all_source_metrics": {
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        },
        "reference_holdout": {
            "prediction_changes": sum(row["old_prediction"] != row["new_prediction"] for row in comparisons),
            "max_absolute_score_delta": max(row["absolute_score_delta"] for row in comparisons),
            "comparisons": comparisons,
        },
        "videos": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"videos", "reference_holdout"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
