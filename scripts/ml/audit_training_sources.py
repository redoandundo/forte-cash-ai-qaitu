from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.bundle import write_json
from fortebank_audit.ml.evaluation import file_hash
from fortebank_audit.ml.experiment_data import load_sources, make_folds, split_sources


def main():
    parser = argparse.ArgumentParser(description="Validate local source data; produce a private annotation-review queue.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a data audit")
    sources = load_sources(args.manifest, args.data_root, args.review)
    folds = make_folds(sources)
    for fold in range(5):
        split_sources(sources, folds, fold, (fold + 1) % 5)
    for row in sources:
        if file_hash(Path(row["path"])) != row["sha256"]:
            raise ValueError("Dataset checksum mismatch")
    review_queue = [{
        "source_id": row["source_id"], "video_ids": row.get("video_ids", []),
        "annotation_confidence": row.get("annotation_confidence", "high"),
        "candidate_points_sec": row.get("events_sec", []),
        "needs": ["confirm_label", "confirm_event_intervals", "confirm_event_count", "confirm_scenario_group"],
    } for row in sources if row["label"] and (not row.get("events") or row.get("annotation_confidence") != "high")]
    summary = {
        "sources": len(sources), "labels": dict(Counter(row["label"] for row in sources)),
        "scenario_groups": len({row["scenario_group"] for row in sources}),
        "confidence": dict(Counter(row.get("annotation_confidence", "high") for row in sources)),
        "confirmed_interval_sources": sum(bool(row.get("events")) for row in sources),
        "duplicate_alias_groups": sum(len(row.get("video_ids", [])) > 1 for row in sources),
        "review_queue_count": len(review_queue), "checksum_and_split_checks": "passed",
    }
    write_json(args.output, {"summary": summary, "review_queue": review_queue})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
