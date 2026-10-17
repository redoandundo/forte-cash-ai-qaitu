from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_audit"
FRAME_SIZE = (240, 135)
LABEL_WIDTH = 130


def labeled_tile(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read {path}")
    tile = cv2.resize(image, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    timestamp = path.stem.split("_", 1)[1]
    cv2.putText(tile, f"{float(timestamp):.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 255, 255), 2)
    return tile


def build_sheet(positive: bool) -> Path:
    with (AUDIT_ROOT / "videos.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    frames_by_id: dict[int, list[Path]] = defaultdict(list)
    for path in sorted((AUDIT_ROOT / "frames").glob("*.jpg")):
        frames_by_id[int(path.stem.split("_", 1)[0])].append(path)

    selected = [row for row in rows if (row["has_violation"].lower() == "true") is positive]
    contact_rows: list[np.ndarray] = []
    for row in selected:
        current_id = int(row["video_id"])
        tiles = [labeled_tile(path) for path in frames_by_id[current_id]]
        label = np.full((FRAME_SIZE[1], LABEL_WIDTH, 3), 245, dtype=np.uint8)
        status = "violation" if positive else "no violation"
        cv2.putText(label, f"ID {current_id}", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.putText(label, status, (8, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
        contact_rows.append(np.hstack([label, *tiles]))

    sheet = np.vstack(contact_rows)
    output = AUDIT_ROOT / ("positive_contact_sheet.jpg" if positive else "negative_contact_sheet.jpg")
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"Could not write {output}")
    return output


if __name__ == "__main__":
    print(build_sheet(positive=True))
    print(build_sheet(positive=False))
