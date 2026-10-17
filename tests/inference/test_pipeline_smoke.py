from __future__ import annotations

from pathlib import Path

import pytest

from fortebank_audit.inference.contracts import EvidenceExtractionError, PipelineConfig, PredictionError
from fortebank_audit.inference.pipeline import analyze_video
from fortebank_audit.inference.predictor import MockPredictor


ROOT = Path(__file__).resolve().parents[2]


def test_mock_pipeline_is_deterministic_on_generated_video(
    tmp_path: Path, synthetic_video: Path
) -> None:
    config = PipelineConfig(window_sec=8, stride_sec=2, num_frames=4, threshold=0.5)
    result = analyze_video(
        synthetic_video,
        MockPredictor([0.8]),
        config,
        evidence_root=tmp_path,
        run_id="smoke",
    )
    assert result["verdict"] == "possible_violation"
    assert result["events"]
    assert result["window_summary"]["positive_fraction"] == 1.0
    assert result["runtime"]["windows"] > 0
    for evidence in result["events"][0]["evidence"]:
        assert (tmp_path / evidence["jpeg_path"]).is_file()


def test_predictor_cannot_return_an_empty_batch(synthetic_video: Path, tmp_path: Path) -> None:
    class EmptyPredictor:
        device = "CPU"
        model_version = "broken"

        def predict_batch(self, windows):
            return []

    config = PipelineConfig(window_sec=8, stride_sec=2, num_frames=4, threshold=0.5)
    with pytest.raises(PredictionError, match="incomplete batch"):
        analyze_video(synthetic_video, EmptyPredictor(), config, evidence_root=tmp_path)


def test_model_window_contract_cannot_be_bypassed(synthetic_video: Path, tmp_path: Path) -> None:
    class Contract:
        num_frames = 4
        window_sec = 8.0
        stride_sec = 2.0

    class ContractPredictor:
        device = "CPU"
        model_version = "contract"
        preprocessing = Contract()

        def predict_batch(self, windows):
            raise AssertionError("contract must be checked before inference")

    config = PipelineConfig(window_sec=1, stride_sec=1, num_frames=4, threshold=0.5)
    with pytest.raises(PredictionError, match="does not match"):
        analyze_video(synthetic_video, ContractPredictor(), config, evidence_root=tmp_path)


def test_existing_evidence_run_is_rejected(synthetic_video: Path, tmp_path: Path) -> None:
    config = PipelineConfig(window_sec=8, stride_sec=2, num_frames=4, threshold=0.5)
    analyze_video(synthetic_video, MockPredictor([0.8]), config, evidence_root=tmp_path, run_id="a/b")
    with pytest.raises(EvidenceExtractionError, match="already exists"):
        analyze_video(synthetic_video, MockPredictor([0.8]), config, evidence_root=tmp_path, run_id="a/b")
