from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fortebank_audit import build_index, extract_frame, probe_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index videos and extract evidence frames.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "dataset_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_index(args.data_root)
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for record in records:
        metadata = probe_video(record.path)
        row = record.as_row() | metadata
        rows.append(row)

        center = record.timecode_sec if record.timecode_sec is not None else float(metadata["duration_sec"]) / 2
        for offset in (-4.0, -2.0, 0.0, 2.0, 4.0):
            timestamp = min(max(0.0, center + offset), max(0.0, float(metadata["duration_sec"]) - 0.05))
            output = args.out / "frames" / f"{record.video_id:03d}_{timestamp:07.2f}.jpg"
            extract_frame(record.path, timestamp, output)

    report_path = args.out / "videos.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positive = sum(bool(row["has_violation"]) for row in rows)
    print(f"Indexed {len(rows)} videos: {positive} positive, {len(rows) - positive} negative")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

