from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.bundle import export_bundle
from fortebank_audit.ml.preprocessing import VideoSpec, normalize_rgb
from fortebank_audit.ml.video import sample_video


def main():
    parser = argparse.ArgumentParser(description="Atomically export a versioned ONNX bundle; never overwrite fallback.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--code-commit", default="uncommitted")
    parser.add_argument("--parity-video", type=Path, nargs="+", required=True)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    spec = VideoSpec(**checkpoint["spec"])
    inputs = []
    for path in args.parity_video:
        samples = sample_video(path, spec)
        indices = np.unique(np.linspace(0, len(samples.frames) - 1, min(8, len(samples.frames))).astype(int))
        inputs.extend(normalize_rgb(samples.frames[index], spec.image_size) for index in indices)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics else None
    result = export_bundle(args.checkpoint, calibration, args.output_dir, np.stack(inputs), args.model_version, metrics, args.code_commit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
