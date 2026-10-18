from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_clip(source: Path, target: Path, center: float, seconds: float, output_fps: float, max_width: int) -> dict:
    capture = cv2.VideoCapture(str(source))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = source_frames / source_fps if source_fps else 0.0
    start = max(0.0, min(center - seconds / 2, max(0.0, duration - seconds)))
    end = min(duration, start + seconds)
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or max_width)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or max_width)
    scale = min(1.0, max_width / max(width, 1))
    size = (max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size)
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create {target}")

    step = source_fps / output_fps
    next_source_index = start * source_fps
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current_index = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
        current_sec = current_index / source_fps
        if current_sec > end:
            break
        if current_index + 1e-6 >= next_source_index:
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
            next_source_index += step
    capture.release()
    writer.release()
    if written < 2:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Too few frames extracted from {source} at {center:.2f}s")
    return {"start_sec": round(start, 3), "end_sec": round(end, 3), "frames": written, "fps": output_fps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed windows around annotations and matched negative clips.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "analysis" / "dataset_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "clips")
    parser.add_argument("--output-manifest", type=Path, default=PROJECT_ROOT / "analysis" / "clip_manifest.json")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--max-width", type=int, default=320)
    args = parser.parse_args()

    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips: list[dict] = []
    for source in source_manifest["sources"]:
        centers = source["events_sec"] if source["label"] else [source["duration_sec"] * position for position in (0.25, 0.5, 0.75)]
        source_weight = 1.0 / len(centers)
        confidence_weight = {"high": 1.0, "medium": 0.75, "low": 0.5}[source.get("annotation_confidence", "high")]
        for event_index, center in enumerate(centers):
            clip_id = f"{source['source_id']}_{event_index}_{source['label']}"
            target = args.output_dir / source["split"] / f"{clip_id}.mp4"
            extraction = extract_clip(project_path(source["path"]), target, center, args.seconds, args.fps, args.max_width)
            clips.append(
                {
                    "clip_id": clip_id,
                    "source_id": source["source_id"],
                    "source_video_ids": source["video_ids"],
                    "path": target.resolve().relative_to(PROJECT_ROOT).as_posix(),
                    "label": source["label"],
                    "split": source["split"],
                    "event_center_sec": round(center, 3) if source["label"] else None,
                    "sample_weight": round(source_weight * confidence_weight, 4),
                    "scenario_group": source.get("scenario_group"),
                    "annotation_confidence": source.get("annotation_confidence", "high"),
                    **extraction,
                }
            )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "clip_seconds": args.seconds,
        "output_fps": args.fps,
        "summary": {
            "clips": len(clips),
            "positive": sum(item["label"] == 1 for item in clips),
            "negative": sum(item["label"] == 0 for item in clips),
            "train": sum(item["split"] == "train" for item in clips),
            "validation": sum(item["split"] == "validation" for item in clips),
        },
        "clips": clips,
    }
    args.output_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output_manifest.resolve())


if __name__ == "__main__":
    main()
