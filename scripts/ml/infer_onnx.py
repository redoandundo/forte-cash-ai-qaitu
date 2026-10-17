from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fortebank_audit.ml.data import preprocess


def sample(path: Path, center: float, seconds: float, count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps
    start = max(0.0, min(center - seconds / 2, max(0.0, duration - seconds)))
    frames = []
    for moment in np.linspace(start, min(duration, start + seconds), count, endpoint=False):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(moment * 1000))
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not decode {path} at {moment:.2f}s")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exported ONNX model on one complete video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "best_model.onnx")
    parser.add_argument("--stride", type=float, default=2.0)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    preprocessing = json.loads((PROJECT_ROOT / "models" / "preprocessing.json").read_text(encoding="utf-8"))
    threshold = json.loads((PROJECT_ROOT / "models" / "thresholds.json").read_text(encoding="utf-8"))["violation"]
    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25)
    duration = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    capture.release()
    window = float(preprocessing["window_sec"])
    centers = np.arange(window / 2, max(window / 2 + 0.01, duration - window / 2 + 0.01), args.stride)
    if len(centers) == 0:
        centers = np.array([duration / 2])

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    scores = []
    for offset in range(0, len(centers), 8):
        batch = np.stack([
            preprocess(
                sample(args.video, float(center), window * preprocessing.get("temporal_span_fraction", 1.0), preprocessing["num_frames"]),
                preprocessing["image_size"],
                False,
                preprocessing["normalization"],
            ).numpy()
            for center in centers[offset : offset + 8]
        ]).astype(np.float32)
        logits = session.run([preprocessing["output_name"]], {preprocessing["input_name"]: batch})[0].reshape(-1)
        scores.extend((1 / (1 + np.exp(-logits))).tolist())
    peak_index = int(np.argmax(scores))
    score = float(scores[peak_index])
    print(json.dumps({
        "video": str(args.video.resolve()),
        "label": "violation" if score >= threshold else "no_violation",
        "score": score,
        "threshold": threshold,
        "peak_time_sec": float(centers[peak_index]),
        "runtime_sec": time.perf_counter() - started,
        "windows": len(scores),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
