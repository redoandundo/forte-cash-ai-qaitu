from __future__ import annotations

from pathlib import Path

import pytest

from fortebank_audit.inference.contracts import PredictorLoadError
from fortebank_audit.inference.predictor import OnnxPredictor
from fortebank_audit.inference.preprocessing import PreprocessingConfig


ROOT = Path(__file__).resolve().parents[2]


def test_current_onnx_bundle_loads_and_checksum_matches() -> None:
    preprocessing = PreprocessingConfig.from_file(ROOT / "models" / "preprocessing.json")
    predictor = OnnxPredictor(
        ROOT / "models" / "best_model.onnx",
        preprocessing,
        ROOT / "models" / "model_manifest.json",
        prefer_cuda=False,
    )
    assert predictor.device == "CPU"
    assert predictor.model_version == "mobilenet-temporal-pairsafe-v3"


def test_fixed_batch_model_is_rejected_by_dynamic_batch_manifest() -> None:
    class Value:
        def __init__(self, name: str, shape: list[object]) -> None:
            self.name = name
            self.shape = shape
            self.type = "tensor(float)"

    class FixedBatchSession:
        def get_inputs(self):
            return [Value("video", [1, 3, 4, 160, 160])]

        def get_outputs(self):
            return [Value("logits", [1])]

    preprocessing = PreprocessingConfig.from_file(ROOT / "models" / "preprocessing.json")
    with pytest.raises(PredictorLoadError, match="dynamic dimension"):
        OnnxPredictor(
            ROOT / "models" / "best_model.onnx",
            preprocessing,
            ROOT / "models" / "model_manifest.json",
            prefer_cuda=False,
            session=FixedBatchSession(),
        )
