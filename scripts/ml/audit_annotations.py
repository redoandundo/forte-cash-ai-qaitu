from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fortebank_audit.dataset import build_index

SCENARIO_PAIRS = [(11, 12), (15, 16), (18, 19), (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (34, 35), (38, 39), (44, 45), (47, 48)]
VALIDATION_IDS = {15, 16, 22, 23, 34, 35, 44, 45}
CONFIDENCE_BY_ID = {
    15: "high", 24: "high", 30: "high", 33: "high", 34: "high", 38: "high", 40: "high",
    4: "medium", 13: "medium", 17: "medium", 18: "medium", 26: "medium", 28: "medium", 32: "medium", 44: "medium",
    1: "low", 2: "low", 11: "low", 20: "low", 22: "low", 46: "low", 47: "low",
}


def scenario_group(video_ids: list[int]) -> str:
    all_ids = set(video_ids)
    for left, right in SCENARIO_PAIRS:
        if all_ids.intersection({left, right}):
            return f"pair-{left}-{right}"
    return f"source-{min(video_ids)}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_info(path: Path) -> tuple[float, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return (frames / fps if fps else 0.0), fps, width, height


def assign_splits(groups: list[dict], validation_fraction: float, seed: int) -> None:
    del validation_fraction, seed
    other_scenarios = sorted({item["scenario_group"] for item in groups if not set(item["video_ids"]).intersection(VALIDATION_IDS)})
    fold_by_scenario = {name: 1 + index % 4 for index, name in enumerate(other_scenarios)}
    for item in groups:
        is_validation = bool(set(item["video_ids"]).intersection(VALIDATION_IDS))
        item["fold"] = 0 if is_validation else fold_by_scenario[item["scenario_group"]]
        item["split"] = "validation" if is_validation else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit labels, collapse duplicate videos, and build a leakage-free manifest.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "open")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "analysis" / "dataset_manifest.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build_index(args.data_root)
    hashed: dict[str, list] = defaultdict(list)
    for record in records:
        hashed[sha256(record.path)].append(record)

    groups: list[dict] = []
    conflicts: list[dict] = []
    for digest, duplicate_records in sorted(hashed.items(), key=lambda item: min(row.video_id for row in item[1])):
        labels = {row.has_violation for row in duplicate_records}
        if len(labels) != 1:
            conflicts.append({"sha256": digest, "video_ids": [row.video_id for row in duplicate_records], "reason": "label_conflict"})
            continue
        duration, fps, width, height = video_info(duplicate_records[0].path)
        events = sorted({round(row.timecode_sec, 3) for row in duplicate_records if row.timecode_sec is not None})
        outside = [value for value in events if value < 0 or value > duration]
        if outside:
            conflicts.append({"sha256": digest, "video_ids": [row.video_id for row in duplicate_records], "reason": "timecode_outside_video", "values": outside})
        ids = sorted(row.video_id for row in duplicate_records)
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        annotation_confidence = max((CONFIDENCE_BY_ID.get(value, "high") for value in ids), key=confidence_order.get) if labels == {True} else "high"
        groups.append(
            {
                "source_id": digest[:16],
                "sha256": digest,
                "video_ids": ids,
                "aliases": sorted(row.path.name for row in duplicate_records),
                "path": str(duplicate_records[0].path.resolve()),
                "label": int(duplicate_records[0].has_violation),
                "events_sec": events,
                "duration_sec": round(duration, 3),
                "fps": round(fps, 3),
                "width": width,
                "height": height,
                "scenario_group": scenario_group(ids),
                "annotation_confidence": annotation_confidence,
            }
        )

    assign_splits(groups, args.validation_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "split_unit": "scenario_group containing SHA-256 sources",
        "duplicate_policy": {
            "Х-4/Х-47": "one positive source with candidate events at 57 and 68 seconds",
            "Х-5/Х-48": "one negative source",
        },
        "summary": {
            "label_rows": len(records),
            "unique_sources": len(groups),
            "positive_sources": sum(item["label"] == 1 for item in groups),
            "negative_sources": sum(item["label"] == 0 for item in groups),
            "duplicate_sets": sum(len(item["video_ids"]) > 1 for item in groups),
            "blocking_conflicts": len(conflicts),
        },
        "conflicts": conflicts,
        "sources": groups,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
