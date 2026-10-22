from __future__ import annotations

import pytest

from fortebank_audit.inference.benchmark import build_predictor
from fortebank_audit.inference.contracts import PredictorLoadError


def test_onnx_predictor_requires_an_explicit_external_bundle() -> None:
    with pytest.raises(PredictorLoadError, match="explicit canonical bundle"):
        build_predictor("onnx", prefer_cuda=False)
