from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundle import load_bundle_contract
from .contracts import PipelineConfig
from .pipeline import analyze_video
from .predictor import MockPredictor, OnnxPredictor
from .preprocessing import PreprocessingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_pipeline_config(
    bundle_root: str | Path | None = None,
    *,
    allow_development_model: bool = False,
) -> tuple[PreprocessingConfig, PipelineConfig, dict]:
    if bundle_root is not None:
        bundle = load_bundle_contract(
            bundle_root,
            allow_development_model=allow_development_model,
        )
        return bundle.preprocessing, bundle.pipeline_config, bundle.manifest
    bundle_root = PROJECT_ROOT / "models"
    preprocessing = PreprocessingConfig.from_file(bundle_root / "preprocessing.json")
    manifest = json.loads((bundle_root / "model_manifest.json").read_text(encoding="utf-8"))
    thresholds = json.loads((bundle_root / "thresholds.json").read_text(encoding="utf-8"))
    if float(manifest["threshold"]) != float(thresholds["violation"]):
        raise ValueError("Model threshold disagrees with threshold contract")
    if manifest["aggregation"] != thresholds["aggregation"]:
        raise ValueError("Model aggregation disagrees with threshold contract")
    config = PipelineConfig(
        window_sec=preprocessing.window_sec,
        stride_sec=preprocessing.stride_sec,
        num_frames=preprocessing.num_frames,
        threshold=float(thresholds["violation"]),
        aggregation=str(thresholds["aggregation"]),
        sampling=str(manifest["sampling"]),
    )
    return preprocessing, config, manifest


def build_predictor(
    kind: str,
    bundle_root: str | Path | None = None,
    *,
    allow_development_model: bool = False,
    prefer_cuda: bool = True,
):
    if kind == "mock":
        return MockPredictor([0.1, 0.2, 0.8, 0.9, 0.3])
    if bundle_root is not None:
        bundle = load_bundle_contract(
            bundle_root,
            allow_development_model=allow_development_model,
        )
        return OnnxPredictor(
            bundle.model_path,
            bundle.preprocessing,
            bundle.manifest_path,
            allow_development_model=allow_development_model,
            prefer_cuda=prefer_cuda,
        )
    preprocessing, _, _ = load_pipeline_config()
    return OnnxPredictor(
        PROJECT_ROOT / "models" / "best_model.onnx",
        preprocessing,
        PROJECT_ROOT / "models" / "model_manifest.json",
        prefer_cuda=prefer_cuda,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the local video inference pipeline")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--predictor", choices=("mock", "onnx"), default="mock")
    parser.add_argument("--bundle", type=Path, help="Canonical versioned ML bundle directory")
    parser.add_argument(
        "--allow-development-model",
        action="store_true",
        help="Allow a bundle whose metrics mark it development-only",
    )
    parser.add_argument("--evidence-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "evidence")
    args = parser.parse_args()

    _, config, _ = load_pipeline_config(
        args.bundle,
        allow_development_model=args.allow_development_model,
    )
    predictor = build_predictor(
        args.predictor,
        args.bundle,
        allow_development_model=args.allow_development_model,
    )
    result = analyze_video(args.video, predictor, config, evidence_root=args.evidence_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
