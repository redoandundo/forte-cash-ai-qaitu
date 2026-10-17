from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import PipelineConfig
from .pipeline import analyze_video
from .predictor import MockPredictor, OnnxPredictor
from .preprocessing import PreprocessingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_predictor(kind: str):
    if kind == "mock":
        return MockPredictor([0.1, 0.2, 0.8, 0.9, 0.3])
    preprocessing = PreprocessingConfig.from_file(PROJECT_ROOT / "models" / "preprocessing.json")
    return OnnxPredictor(
        PROJECT_ROOT / "models" / "best_model.onnx",
        preprocessing,
        PROJECT_ROOT / "configs" / "inference" / "model_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the local video inference pipeline")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--predictor", choices=("mock", "onnx"), default="mock")
    parser.add_argument("--evidence-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "evidence")
    args = parser.parse_args()

    preprocessing = PreprocessingConfig.from_file(PROJECT_ROOT / "models" / "preprocessing.json")
    threshold = json.loads((PROJECT_ROOT / "models" / "thresholds.json").read_text(encoding="utf-8"))["violation"]
    config = PipelineConfig(
        window_sec=preprocessing.window_sec,
        stride_sec=preprocessing.stride_sec,
        num_frames=preprocessing.num_frames,
        threshold=float(threshold),
    )
    result = analyze_video(args.video, build_predictor(args.predictor), config, evidence_root=args.evidence_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
