from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.bundle import cpu_session, validate_bundle, write_json
from fortebank_audit.ml.preprocessing import VideoSpec, normalize_rgb
from fortebank_audit.ml.video import sample_video


def main():
    parser = argparse.ArgumentParser(description="Verify a supplied inference package against the ML bundle without editing that package.")
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite adapter verification")
    package = args.reference_package.resolve()
    sys.path.insert(0, str(package.parent))
    contracts = importlib.import_module(package.name + ".contracts")
    preprocessing = importlib.import_module(package.name + ".preprocessing")
    predictor_module = importlib.import_module(package.name + ".predictor")
    manifest = validate_bundle(args.bundle)
    pre = preprocessing.PreprocessingConfig.from_file(args.bundle / "preprocessing.json")
    spec = VideoSpec(pre.num_frames, pre.image_size, pre.window_sec, pre.stride_sec, True)
    samples = sample_video(args.video, spec)
    indices = sorted(set([0, len(samples.frames) // 2, len(samples.frames) - 1]))
    windows = [contracts.ClipWindow(i, float(samples.starts[i]), float(samples.ends[i]), float(samples.centers[i]), samples.frames[i]) for i in indices]
    predictor = predictor_module.OnnxPredictor(args.bundle / "compatibility/model_dynamic.onnx", pre, args.bundle / "compatibility/inference/model_manifest.json", prefer_cuda=False)
    actual = predictor.predict_batch(windows)
    session = cpu_session(args.bundle / manifest["artifact"])
    expected = []
    for window in windows:
        ml_input = normalize_rgb(window.frames_rgb, spec.image_size)
        assert np.array_equal(ml_input, preprocessing.preprocess_frames(window.frames_rgb, pre))
        logits = session.run(["logits"], {"video": ml_input[None]})[0]
        expected.append(float(1 / (1 + np.exp(-float(logits[0])))))
    error = max(abs(a.score - e) for a, e in zip(actual, expected, strict=True))
    if error > 1e-5:
        raise ValueError("Reference predictor score parity failed")
    if predictor.predict_batch([]) != []:
        raise ValueError("Empty batch handling failed")
    # Exercise actual CPU fallback when CUDA is requested but not installed.
    fallback = predictor_module.OnnxPredictor(args.bundle / "compatibility/model_dynamic.onnx", pre, args.bundle / "compatibility/inference/model_manifest.json", prefer_cuda=True)
    fallback.predict_batch(windows[:1])
    report = {"reference_commit": args.reference_commit, "model_version": manifest["model_version"], "max_score_error": error,
              "preprocessing_exact": True, "windows": len(windows), "device": predictor.device, "fallback_device": fallback.device,
              "scope": "predictor_only; tail, smoothing, event aggregation and JPEG integration still require pipeline verification"}
    write_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
