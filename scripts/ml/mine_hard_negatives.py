from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from prepare_clips import extract_clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract leakage-safe hard negatives from baseline OOF false-positive peaks.")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "ml" / "nested_oof_summary.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "analysis" / "dataset_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "hard_negatives")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "hard_negative_clips.json")
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    sources = {row["source_id"]: row for row in json.loads(args.dataset.read_text(encoding="utf-8"))["sources"]}
    clips = []
    for row in report["videos"]:
        if row["label"] != 0 or row["prediction"] != 1:
            continue
        source = sources[row["source_id"]]
        clip_id = f"{row['source_id']}_hard_oof"
        target = args.output_dir / f"{clip_id}.mp4"
        extraction = extract_clip(Path(source["path"]), target, float(row["peak_time_sec"]), args.seconds, 12.0, 320)
        clips.append({
            "clip_id": clip_id,
            "source_id": row["source_id"],
            "source_video_ids": source["video_ids"],
            "path": str(target.resolve()),
            "label": 0,
            "split": "train",
            "event_center_sec": None,
            "sample_weight": 1.0,
            "scenario_group": source["scenario_group"],
            "annotation_confidence": source.get("annotation_confidence", "high"),
            "mined_from": "baseline_nested_oof_false_positive_peak",
            "mining_score": row["score"],
            **extraction,
        })
    payload = {"schema_version": 1, "leakage_policy": "Each nested split assigns mined clips by source scenario fold; outer and calibration sources remain excluded from training.", "clips": clips}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"hard_negative_clips": len(clips), "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
