from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.bundle import write_json
from fortebank_audit.ml.evaluation import file_hash
from fortebank_audit.ml.experiment_data import load_sources
from fortebank_audit.ml.runtime import predict_video


def main():
    parser = argparse.ArgumentParser(description="Uncached CPU ML pipeline timing; this is not a held-out accuracy evaluation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite benchmark evidence")
    rows = []
    for source in load_sources(args.manifest, args.data_root):
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise ValueError("Source checksum mismatch")
        result = predict_video(Path(source["path"]), args.bundle)
        rows.append({"source_id": source["source_id"], "duration_sec": result["duration_sec"], "runtime_sec": result["runtime_sec"], "windows": result["windows"]})
        print(json.dumps({"completed": len(rows), "duration_sec": result["duration_sec"], "runtime_sec": result["runtime_sec"]}), flush=True)
    values = [row["runtime_sec"] for row in rows]
    summary = {
        "model_version": result["model_version"], "sources": len(rows), "failures": 0,
        "total_video_sec": sum(row["duration_sec"] for row in rows),
        "longest_video_sec": max(row["duration_sec"] for row in rows),
        "total_wall_sec": sum(values), "mean_sec": float(np.mean(values)),
        "p95_sec": float(np.percentile(values, 95)), "max_sec": max(values),
        "device": "CPU", "cache": False, "scope": result["runtime_scope"],
        "all_observed_under_300_sec": max(values) < 300,
        "full_service_under_300_sec_verified": False, "accuracy_evaluation": False,
    }
    write_json(args.output, {"summary": summary, "videos": rows})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
