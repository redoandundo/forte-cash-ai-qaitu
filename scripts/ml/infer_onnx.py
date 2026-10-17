from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fortebank_audit.onnx_runtime import provider, score_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exported ONNX model on one complete video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "best_model.onnx")
    parser.add_argument("--stride", type=float, default=2.0)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    started = time.perf_counter()
    windows, threshold = score_video(args.video, args.model, args.stride)
    peak = max(windows, key=lambda item: item.score)
    print(json.dumps({
        "video": str(args.video.resolve()),
        "label": "violation" if peak.score >= threshold else "no_violation",
        "score": peak.score,
        "threshold": threshold,
        "peak_time_sec": peak.timestamp_sec,
        "runtime_sec": time.perf_counter() - started,
        "windows": len(windows),
        "provider": provider(args.model),
        "aggregation": "three_window_weighted_smoothing_then_peak",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
