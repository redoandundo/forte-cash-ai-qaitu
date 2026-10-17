from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fortebank_audit.ml.data import preprocess
from fortebank_audit.ml.models import build_model


def sample_window(path: Path, center: float, seconds: float, frame_count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps
    start = max(0.0, min(center - seconds / 2, max(0.0, duration - seconds)))
    times = np.linspace(start, min(duration, start + seconds), frame_count, endpoint=False)
    frames = []
    for moment in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(moment * 1000))
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not decode {path} at {moment:.2f}s")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sliding-window predictions on complete validation videos.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "analysis" / "dataset_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=float, default=2.0)
    parser.add_argument("--window", type=float, default=8.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    clip_threshold = json.loads(args.metrics.read_text(encoding="utf-8"))["source_metrics"]["threshold"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["model"], pretrained=False, train_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = [source for source in payload["sources"] if source["split"] == "validation"]
    rows = []
    durations = []
    for source in sources:
        started = time.perf_counter()
        centers = np.arange(args.window / 2, max(args.window / 2 + 0.01, source["duration_sec"] - args.window / 2 + 0.01), args.stride)
        if len(centers) == 0:
            centers = np.array([source["duration_sec"] / 2])
        scores = []
        for offset in range(0, len(centers), 8):
            batch = torch.stack([
                preprocess(sample_window(Path(source["path"]), float(center), args.window * checkpoint.get("temporal_span", 1.0), checkpoint["frames"]), checkpoint["image_size"], False, checkpoint["normalization"])
                for center in centers[offset : offset + 8]
            ]).to(device)
            with torch.inference_mode():
                scores.extend(torch.sigmoid(model(batch)).cpu().tolist())
        best_index = int(np.argmax(scores))
        peak = float(centers[best_index])
        maximum = float(scores[best_index])
        error = min((abs(peak - event) for event in source["events_sec"]), default=None)
        duration = time.perf_counter() - started
        durations.append(duration)
        rows.append({"source_id": source["source_id"], "video_ids": source["video_ids"], "label": source["label"], "score": maximum, "prediction": int(maximum >= clip_threshold), "peak_time_sec": peak, "timestamp_error_sec": error, "runtime_sec": duration})
        print(json.dumps(rows[-1]))

    labels = np.array([row["label"] for row in rows])
    scores = np.array([row["score"] for row in rows])
    ordered = np.sort(np.unique(scores))
    candidates = np.concatenate(([0.0], (ordered[:-1] + ordered[1:]) / 2, [1.0]))
    threshold, best_f2 = 0.5, -1.0
    for candidate in candidates:
        candidate_predictions = scores >= candidate
        candidate_precision = precision_score(labels, candidate_predictions, zero_division=0)
        candidate_recall = recall_score(labels, candidate_predictions, zero_division=0)
        candidate_f2 = 5 * candidate_precision * candidate_recall / (4 * candidate_precision + candidate_recall) if candidate_precision + candidate_recall else 0.0
        if candidate_f2 > best_f2:
            threshold, best_f2 = float(candidate), float(candidate_f2)
    predictions = scores >= threshold
    for row, prediction in zip(rows, predictions):
        row["prediction"] = int(prediction)
    errors = np.array([row["timestamp_error_sec"] for row in rows if row["timestamp_error_sec"] is not None])
    total_negative_minutes = sum(source["duration_sec"] for source in sources if source["label"] == 0) / 60
    false_positives = sum(row["label"] == 0 and row["prediction"] == 1 for row in rows)
    result = {
        "model": checkpoint["model"],
        "threshold": threshold,
        "clip_threshold": clip_threshold,
        "threshold_selected_on": "full_validation_videos",
        "video_metrics": {
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f2": best_f2,
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
            "false_positives_per_minute": false_positives / total_negative_minutes if total_negative_minutes else None,
        },
        "timestamp_metrics": {
            "mae_sec": float(errors.mean()) if len(errors) else None,
            "within_2_sec": float((errors <= 2).mean()) if len(errors) else None,
            "within_5_sec": float((errors <= 5).mean()) if len(errors) else None,
        },
        "performance": {"mean_runtime_sec": float(np.mean(durations)), "p95_runtime_sec": float(np.percentile(durations, 95))},
        "videos": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
