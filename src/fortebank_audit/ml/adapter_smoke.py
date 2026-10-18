"""Execute a pinned local Git revision of the inference adapter without editing it."""
from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .bundle import write_json
from .diagnostics import trace_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--video", type=Path, nargs="+", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, help="Optional local JPEG directory (ASCII path for affected Windows OpenCV builds)")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.reference_commit):
        raise ValueError("Pin a complete local Git commit SHA")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = args.output / "reference_inference"
    reference.mkdir()
    files = ("contracts", "preprocessing", "bundle", "predictor", "aggregation", "video_reader", "windowing", "evidence", "pipeline")
    for name in files:
        code = subprocess.check_output(["git", "show", f"{args.reference_commit}:src/fortebank_audit/inference/{name}.py"])
        (reference / (name + ".py")).write_bytes(code)
    (reference / "__init__.py").write_text("", encoding="utf-8")
    sys.path.insert(0, str(args.output.resolve()))
    loader = importlib.import_module("reference_inference.bundle")
    predictor_module = importlib.import_module("reference_inference.predictor")
    pipeline = importlib.import_module("reference_inference.pipeline")
    contract = loader.load_bundle_contract(args.bundle, allow_development_model=True)
    predictor = predictor_module.OnnxPredictor(contract.model_path, contract.preprocessing, contract.manifest_path,
                                               allow_development_model=True, prefer_cuda=True)
    results = []
    for index, video in enumerate(args.video):
        reference_trace = trace_video(video, args.bundle, check_order=False)
        _, predictions, seconds, rss = pipeline.score_video(video, predictor, contract.pipeline_config)
        if len(predictions) != reference_trace["window_count"]:
            raise ValueError("Inference and ML window counts differ")
        error = max(abs(p.score - row["sigmoid_score"]) for p, row in zip(predictions, reference_trace["windows"]))
        timestamp_error = max(abs(p.center_sec - row["center_sec"]) for p, row in zip(predictions, reference_trace["windows"]))
        if error > 1e-6 or timestamp_error > 1e-6:
            raise ValueError("Inference and ML window scores/timestamps differ")
        evidence_root = args.evidence_root or args.output / "evidence"
        result = pipeline.analyze_video(video, predictor, contract.pipeline_config, evidence_root=evidence_root)
        actual_events = result["events"]
        expected_events = reference_trace["events"]
        if len(actual_events) != len(expected_events):
            raise ValueError("Inference and ML event counts differ")
        for actual, expected in zip(actual_events, expected_events):
            for key in ("start_sec", "end_sec", "peak_time_sec"):
                if abs(actual[key] - expected[key]) > 1e-6:
                    raise ValueError("Inference and ML event timestamps differ")
            if actual["supporting_windows"] != expected["supporting_windows"]:
                raise ValueError("Inference and ML event support differs")
            if not 1 <= len(actual["evidence"]) <= 3:
                raise ValueError("Invalid JPEG count")
            for evidence in actual["evidence"]:
                jpeg = evidence_root / evidence["jpeg_path"]
                if cv2.imdecode(np.frombuffer(jpeg.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR) is None:
                    raise ValueError("Evidence is not decodable JPEG")
        results.append({"video_index": index, "score_max_error": error, "timestamp_max_error": timestamp_error,
                        "windows": len(predictions), "scoring_sec": seconds, "rss_delta_mb": rss, "result": result})
        print(json.dumps({"video_index": index, "score_max_error": error, "windows": len(predictions),
                          "events": len(actual_events), "pipeline_sec": result["runtime"]["processing_sec"]}), flush=True)
    corrupt = args.output / "corrupt.mov"
    corrupt.write_bytes(b"not a video")
    try:
        pipeline.analyze_video(corrupt, predictor, contract.pipeline_config, evidence_root=args.output / "evidence")
    except Exception as error:
        failure_type = type(error).__name__
    else:
        raise AssertionError("Corrupt video produced a verdict")
    write_json(args.output / "summary.json", {"reference_commit": args.reference_commit, "device": predictor.device,
               "corrupt_video_exception": failure_type, "videos": results,
               "scope": "real_model_decode_aggregation_JPEG_no_API"})


if __name__ == "__main__":
    main()
