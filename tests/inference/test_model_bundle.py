from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fortebank_audit.inference.bundle import load_bundle_contract
from fortebank_audit.inference.contracts import ClipWindow, PredictorLoadError
from fortebank_audit.inference.predictor import OnnxPredictor


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_bundle(root: Path) -> Path:
    root.mkdir()
    model = root / "model.onnx"
    model.write_bytes(b"synthetic-model-for-session-double")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = {
        "contract_version": 1,
        "model_name": "mobilenet_v3_small_temporal",
        "model_version": "baseline-repair-v2-development",
        "artifact": "model.onnx",
        "sha256": digest,
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
    }
    preprocessing = {
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
    }
    thresholds = {
        "violation": 0.45335930958390236,
        "aggregation": "three_window_weighted_smoothing_then_peak",
        "smoothing_weights": [0.25, 0.5, 0.25],
        "smoothing_edges": "replicate",
        "short_sequence_rule": "identity_if_fewer_than_three",
        "event_merge": "adjacent_centers_within_stride",
        "max_events": 3,
        "pipeline_sha256": "a" * 64,
    }
    write_json(root / "model_manifest.json", manifest)
    write_json(root / "preprocessing.json", preprocessing)
    write_json(root / "thresholds.json", thresholds)
    write_json(root / "labels.json", {"0": "no_violation", "1": "possible_violation"})
    write_json(root / "metrics.json", {"release_status": "development_only_not_promoted"})
    (root / "ml-to-inference.md").write_text("development bundle\n", encoding="utf-8")
    files = sorted(path for path in root.iterdir() if path.is_file())
    (root / "SHA256SUMS.txt").write_text(
        "".join(
            hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
            for path in files
        ),
        encoding="ascii",
    )
    return root


def test_development_bundle_requires_explicit_opt_in(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    with pytest.raises(PredictorLoadError, match="explicit opt-in"):
        load_bundle_contract(bundle)
    contract = load_bundle_contract(bundle, allow_development_model=True)
    assert contract.pipeline_config.include_tail_window is True
    assert contract.pipeline_config.sampling == "pts_nearest"
    assert contract.pipeline_config.max_events == 3
    assert contract.pipeline_config.batch_size == 1


def test_canonical_fixed_batch_bundle_loads_with_session_double(tmp_path: Path) -> None:
    class Value:
        def __init__(self, name: str, shape: list[object]) -> None:
            self.name = name
            self.shape = shape
            self.type = "tensor(float)"

    class Session:
        def get_inputs(self):
            return [Value("video", [1, 3, 4, 160, 160])]

        def get_outputs(self):
            return [Value("logits", [1])]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, output_names, inputs):
            assert output_names == ["logits"]
            assert inputs["video"].shape == (1, 3, 4, 160, 160)
            return [np.asarray([0.0], dtype=np.float32)]

    bundle = make_bundle(tmp_path / "bundle")
    contract = load_bundle_contract(bundle, allow_development_model=True)
    with pytest.raises(PredictorLoadError, match="explicit opt-in"):
        OnnxPredictor(
            contract.model_path,
            contract.preprocessing,
            contract.manifest_path,
            session=Session(),
        )
    predictor = OnnxPredictor(
        contract.model_path,
        contract.preprocessing,
        contract.manifest_path,
        session=Session(),
        allow_development_model=True,
    )
    assert predictor.model_version == "baseline-repair-v2-development"
    assert predictor.device == "CPU"
    predictions = predictor.predict_batch(
        [
            ClipWindow(
                index=0,
                start_sec=0.0,
                end_sec=6.0,
                center_sec=3.0,
                frames_rgb=np.zeros((4, 16, 16, 3), dtype=np.uint8),
            )
        ]
    )
    assert predictions[0].score == pytest.approx(0.5)
