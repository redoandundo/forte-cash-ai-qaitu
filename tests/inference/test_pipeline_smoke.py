from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fortebank_audit.inference.contracts import EvidenceExtractionError, PipelineConfig, PredictionError
from fortebank_audit.inference.pipeline import analyze_video
from fortebank_audit.inference.predictor import MockPredictor


ROOT = Path(__file__).resolve().parents[2]


def test_temporal_context_and_progress_coexist(synthetic_video, tmp_path):
    stages = []
    result = analyze_video(synthetic_video, MockPredictor([.8]),
                           PipelineConfig(window_sec=8, stride_sec=2, num_frames=4, threshold=.5),
                           evidence_root=tmp_path, run_id="semantic-context", progress=stages.append,
                           include_temporal_context=True)
    assert stages == ["video_decode", "temporal_windows", "model_inference", "aggregation", "evidence"]
    for event in result["events"]:
        assert 1 <= len(event["evidence"]) <= 3
        context = event["_temporal_context"]
        assert 1 <= len(context) <= 5
        assert [f["timestamp_sec"] for f in context] == sorted(f["timestamp_sec"] for f in context)
        assert all((tmp_path / f["jpeg_path"]).is_file() for f in context)


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
    assert len(result["events"][0]["supporting_window_indices"]) == result["events"][0][
        "supporting_windows"
    ]
    assert result["window_summary"]["positive_fraction"] == 1.0
    assert len(result["window_scores"]) == result["runtime"]["windows"]
    assert {
        "window_index",
        "start_sec",
        "end_sec",
        "center_sec",
        "raw_score",
        "score",
    } <= result["window_scores"][0].keys()
    assert result["runtime"]["windows"] > 0
    for evidence in result["events"][0]["evidence"]:
        assert (tmp_path / evidence["jpeg_path"]).is_file()


def test_progress_stages_are_monotonic_and_contain_no_payload(
    tmp_path: Path, synthetic_video: Path
) -> None:
    stages: list[str] = []
    config = PipelineConfig(window_sec=8, stride_sec=2, num_frames=4, threshold=0.5)

    analyze_video(
        synthetic_video,
        MockPredictor([0.8]),
        config,
        evidence_root=tmp_path,
        run_id="progress",
        progress=stages.append,
    )

    assert stages == [
        "video_decode",
        "temporal_windows",
        "model_inference",
        "aggregation",
        "evidence",
    ]
    assert all("/" not in stage and "\\" not in stage for stage in stages)


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


def test_evidence_is_written_and_readable_under_unicode_root(
    synthetic_video: Path, tmp_path: Path
) -> None:
    evidence_root = tmp_path / "доказательства"
    config = PipelineConfig(window_sec=8, stride_sec=2, num_frames=4, threshold=0.5)
    result = analyze_video(
        synthetic_video,
        MockPredictor([0.8]),
        config,
        evidence_root=evidence_root,
        run_id="unicode",
    )
    paths = [evidence_root / item["jpeg_path"] for item in result["events"][0]["evidence"]]
    assert 1 <= len(paths) <= 3
    assert all(
        path.is_file()
        and cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) is not None
        for path in paths
    )
