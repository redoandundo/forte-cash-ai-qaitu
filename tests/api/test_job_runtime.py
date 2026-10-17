from __future__ import annotations

import base64
import sys
from pathlib import Path

import av
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fortebank_audit.inference.contracts import PipelineConfig
from fortebank_audit.inference.predictor import MockPredictor
from fortebank_audit.job_runtime import (
    InferenceRuntime,
    RuntimeConfigurationError,
    RuntimeSettings,
    get_runtime,
)


def _video(path: Path) -> Path:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(24):
            image = np.full((48, 64, 3), 80 + index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_mock_runtime_maps_real_evidence_and_summary(tmp_path: Path) -> None:
    settings = RuntimeSettings("mock", None, False, tmp_path / "evidence")
    runtime = InferenceRuntime(
        settings,
        predictor=MockPredictor([0.9]),
        config=PipelineConfig(
            window_sec=1.0,
            stride_sec=1.0,
            num_frames=4,
            threshold=0.5,
            evidence_count=3,
        ),
    )
    stages: list[str] = []
    result = runtime.analyze(
        _video(tmp_path / "касса.mov"),
        job_id="job-1",
        progress=stages.append,
    )

    assert result.verdict == "violation"
    assert len(result.violations) == 1
    assert "автоматический анализ" in result.summary.lower()
    assert "ручн" not in result.summary.lower()
    assert result.runtime["mode_label"] == "Demo predictor"
    assert result.runtime["predictor_class"] == "MockPredictor"
    assert result.runtime["model_version"] == "mock-v1"
    assert result.runtime["runtime_providers"] == ["MockExecutionProvider"]
    assert result.runtime["model_sha256"] is None
    assert result.runtime["diagnostics"]["window_scores"]
    assert result.runtime["diagnostics"]["events"][0]["supporting_window_indices"]
    assert stages[-1] == "summary"
    detail = result.analysis_details[0]
    assert detail["event_index"] == 1
    assert detail["thesis"].startswith("Вероятно")
    assert len(detail["observations"]) >= 2
    assert detail["reasoning"]
    assert detail["risk"]
    assert detail["rule"]["status"] == "not_established"
    assert detail["rule"]["label"] == "Нормативное основание не установлено"
    frames = result.violations[0]["evidence_frames"]
    assert len(frames) == 3
    for frame in frames:
        payload = base64.b64decode(frame["image_base64"], validate=True)
        assert payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
    assert not (settings.evidence_root / "job-1").exists()


def test_onnx_environment_without_bundle_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QAITU_INFERENCE_MODE", "onnx")
    monkeypatch.delenv("QAITU_MODEL_BUNDLE", raising=False)

    with pytest.raises(RuntimeConfigurationError, match="requires QAITU_MODEL_BUNDLE"):
        RuntimeSettings.from_environment()


def test_inference_mode_must_be_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QAITU_INFERENCE_MODE", raising=False)
    monkeypatch.delenv("QAITU_MODEL_BUNDLE", raising=False)

    with pytest.raises(RuntimeConfigurationError, match="explicitly set"):
        RuntimeSettings.from_environment()


def test_runtime_is_loaded_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QAITU_INFERENCE_MODE", "mock")
    monkeypatch.setenv("QAITU_EVIDENCE_ROOT", str(tmp_path / "evidence"))
    get_runtime.cache_clear()
    try:
        assert get_runtime() is get_runtime()
    finally:
        get_runtime.cache_clear()


def test_injected_predictor_and_config_must_be_provided_together(tmp_path: Path) -> None:
    settings = RuntimeSettings("mock", None, False, tmp_path)

    with pytest.raises(RuntimeConfigurationError, match="injected together"):
        InferenceRuntime(settings, predictor=MockPredictor([0.5]))
