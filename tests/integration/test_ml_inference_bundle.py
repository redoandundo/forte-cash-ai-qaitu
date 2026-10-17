from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import av
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fortebank_audit.inference.benchmark import build_predictor, load_pipeline_config
from fortebank_audit.inference.bundle import load_bundle_contract
from fortebank_audit.inference.contracts import PredictorLoadError
from fortebank_audit.inference.pipeline import analyze_video


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fixed_batch_onnx(path: Path) -> str:
    input_info = helper.make_tensor_value_info(
        "video", TensorProto.FLOAT, [1, 3, 4, 160, 160]
    )
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                ["video"],
                ["logits"],
                axes=[1, 2, 3, 4],
                keepdims=0,
            )
        ],
        "canonical-fixed-batch-smoke",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_canonical_bundle(root: Path) -> Path:
    root.mkdir()
    model_sha = _write_fixed_batch_onnx(root / "model.onnx")
    _write_json(
        root / "model_manifest.json",
        {
            "contract_version": 1,
            "model_name": "canonical_fixed_batch_smoke",
            "model_version": "integration-development",
            "artifact": "model.onnx",
            "sha256": model_sha,
            "runtime": "onnxruntime",
            "input": {
                "name": "video",
                "layout": "NCTHW",
                "shape": [1, 3, 4, 160, 160],
                "dtype": "float32",
                "color": "RGB",
            },
            "output": {
                "name": "logits",
                "semantics": "possible_violation_score",
                "activation": "logit",
                "positive_label": "possible_violation",
            },
            "preprocessing_file": "preprocessing.json",
            "thresholds_file": "thresholds.json",
            "labels_file": "labels.json",
            "metrics_file": "metrics.json",
        },
    )
    _write_json(
        root / "preprocessing.json",
        {
            "input_layout": "NCTHW",
            "input_name": "video",
            "output_name": "logits",
            "dtype": "float32",
            "color": "RGB",
            "normalization": "imagenet",
            "num_frames": 4,
            "image_size": 160,
            "window_sec": 6.0,
            "recommended_stride_sec": 2.0,
            "temporal_span_fraction": 1.0,
            "resize": "opencv_inter_linear_stretch",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "sampling": "uniform_endpoint_false_nearest_pts",
            "include_tail_window": True,
            "timestamp_origin": "stream_start_time",
            "pipeline_version": "pts-tail-smooth-v1",
        },
    )
    _write_json(
        root / "thresholds.json",
        {
            "violation": 0.45335930958390236,
            "aggregation": "three_window_weighted_smoothing_then_peak",
            "smoothing_weights": [0.25, 0.5, 0.25],
            "smoothing_edges": "replicate",
            "short_sequence_rule": "identity_if_fewer_than_three",
            "event_merge": "adjacent_centers_within_stride",
            "max_events": 3,
            "pipeline_sha256": "a" * 64,
        },
    )
    _write_json(root / "labels.json", {"0": "no_violation", "1": "possible_violation"})
    _write_json(root / "metrics.json", {"release_status": "development_only_not_promoted"})
    (root / "ml-to-inference.md").write_text(
        "Synthetic development-only integration fixture.\n", encoding="utf-8"
    )
    files = sorted(path for path in root.iterdir() if path.is_file())
    (root / "SHA256SUMS.txt").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in files
        ),
        encoding="ascii",
    )
    return root


def _make_unicode_video(path: Path) -> Path:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for _ in range(90):
            image = np.full((48, 64, 3), 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_canonical_ml_bundle_runs_full_inference_pipeline(tmp_path: Path) -> None:
    bundle = _make_canonical_bundle(tmp_path / "bundle")
    video = _make_unicode_video(tmp_path / "касса-тест.mov")

    with pytest.raises(PredictorLoadError, match="explicit opt-in"):
        load_bundle_contract(bundle)

    preprocessing, config, manifest = load_pipeline_config(
        bundle, allow_development_model=True
    )
    predictor = build_predictor(
        "onnx",
        bundle,
        allow_development_model=True,
        prefer_cuda=False,
    )
    result = analyze_video(
        video,
        predictor,
        config,
        evidence_root=tmp_path / "evidence",
        run_id="canonical-bundle",
    )

    assert preprocessing.input_layout == "NCTHW"
    assert manifest["input"]["shape"] == [1, 3, 4, 160, 160]
    assert result["verdict"] == "possible_violation"
    assert result["runtime"]["device"] == "CPU"
    assert result["runtime"]["sampling"] == "pts_nearest"
    assert result["runtime"]["windows"] == 3
    assert len(result["events"]) == 1
    evidence = result["events"][0]["evidence"]
    assert len(evidence) == 3
    assert all((tmp_path / "evidence" / item["jpeg_path"]).is_file() for item in evidence)
