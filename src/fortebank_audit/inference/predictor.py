from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import onnxruntime as ort

from .contracts import ClipWindow, PredictionError, PredictorLoadError, WindowPrediction
from .preprocessing import PreprocessingConfig, preprocess_frames


class Predictor(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def device(self) -> str: ...

    def predict_batch(self, windows: Sequence[ClipWindow]) -> list[WindowPrediction]: ...


class MockPredictor:
    def __init__(self, scores: Sequence[float] | None = None) -> None:
        self._scores = tuple(scores) if scores is not None else None

    @property
    def model_version(self) -> str:
        return "mock-v1"

    @property
    def device(self) -> str:
        return "CPU"

    def predict_batch(self, windows: Sequence[ClipWindow]) -> list[WindowPrediction]:
        predictions = []
        for window in windows:
            if self._scores:
                score = float(self._scores[window.index % len(self._scores)])
            else:
                score = float(np.clip(window.frames_rgb.mean() / 255.0, 0.0, 1.0))
            predictions.append(
                WindowPrediction(window.index, window.start_sec, window.end_sec, window.center_sec, score)
            )
        return predictions


class OnnxPredictor:
    def __init__(
        self,
        model_path: str | Path,
        preprocessing: PreprocessingConfig,
        manifest_path: str | Path,
        *,
        prefer_cuda: bool = True,
        session: object | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        manifest_path = Path(manifest_path)
        self.preprocessing = preprocessing
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PredictorLoadError("Model manifest is missing or invalid") from exc
        required = {
            "model_version",
            "sha256",
            "output_semantics",
            "input_shape",
            "input_name",
            "output_name",
            "output_shape",
            "input_layout",
            "dtype",
            "color",
            "contract_version",
            "runtime",
            "artifact",
            "positive_label",
            "preprocessing_file",
            "preprocessing_sha256",
            "threshold_file",
            "threshold_sha256",
        }
        missing = sorted(required - manifest.keys())
        if missing:
            raise PredictorLoadError(f"Model manifest missing fields: {', '.join(missing)}")
        if manifest["output_semantics"] not in {"logit", "probability"}:
            raise PredictorLoadError("output_semantics must be logit or probability")
        if manifest["contract_version"] != 1 or manifest["runtime"] != "onnxruntime":
            raise PredictorLoadError("Unsupported model manifest contract")
        if manifest["positive_label"] != "possible_violation":
            raise PredictorLoadError("Unsupported positive label")
        preprocessing_contract = {
            "input_name": preprocessing.input_name,
            "output_name": preprocessing.output_name,
            "input_layout": preprocessing.input_layout,
            "dtype": preprocessing.dtype,
            "color": preprocessing.color,
        }
        mismatches = [
            field for field, value in preprocessing_contract.items() if manifest[field] != value
        ]
        if mismatches:
            raise PredictorLoadError(
                f"Model manifest disagrees with preprocessing: {', '.join(mismatches)}"
            )
        project_root = manifest_path.resolve().parents[2]
        artifact_path = (project_root / str(manifest["artifact"])).resolve()
        if artifact_path != self.model_path.resolve():
            raise PredictorLoadError("Model artifact path does not match the manifest")
        for file_field, hash_field in (
            ("preprocessing_file", "preprocessing_sha256"),
            ("threshold_file", "threshold_sha256"),
        ):
            contract_file = (project_root / str(manifest[file_field])).resolve()
            if not contract_file.is_file():
                raise PredictorLoadError(f"Missing model contract file: {file_field}")
            actual_contract_hash = hashlib.sha256(contract_file.read_bytes()).hexdigest()
            if actual_contract_hash.lower() != str(manifest[hash_field]).lower():
                raise PredictorLoadError(f"Model contract checksum mismatch: {file_field}")
        if not self.model_path.is_file():
            raise PredictorLoadError("Model artifact does not exist")
        actual_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if actual_hash.lower() != str(manifest["sha256"]).lower():
            raise PredictorLoadError("Model checksum mismatch")
        self._model_version = str(manifest["model_version"])
        self._output_semantics = str(manifest["output_semantics"])
        self._expected_shape = list(manifest["input_shape"])
        self._expected_output_shape = list(manifest["output_shape"])

        if session is None:
            available = ort.get_available_providers()
            providers = []
            if prefer_cuda and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            try:
                session = ort.InferenceSession(str(self.model_path), providers=providers)
            except Exception as exc:  # onnxruntime exposes multiple exception types
                raise PredictorLoadError("Could not load ONNX model") from exc
        self.session = session
        self._validate_session()

    def _validate_session(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != self.preprocessing.input_name:
            raise PredictorLoadError("ONNX input name does not match preprocessing contract")
        if not outputs or self.preprocessing.output_name not in {output.name for output in outputs}:
            raise PredictorLoadError("ONNX output name does not match preprocessing contract")
        if inputs[0].type != "tensor(float)":
            raise PredictorLoadError("ONNX input dtype must be float32")
        selected_output = next(output for output in outputs if output.name == self.preprocessing.output_name)
        if selected_output.type != "tensor(float)":
            raise PredictorLoadError("ONNX output dtype must be float32")
        shape = list(inputs[0].shape)
        expected = self._expected_shape
        if len(shape) != len(expected):
            raise PredictorLoadError("ONNX input rank does not match manifest")
        for actual, wanted in zip(shape, expected, strict=True):
            if wanted is None and isinstance(actual, int):
                raise PredictorLoadError("ONNX dynamic dimension does not match manifest")
            if wanted is not None and actual != wanted:
                raise PredictorLoadError("ONNX input shape does not match manifest")
        output_shape = list(selected_output.shape)
        if len(output_shape) != len(self._expected_output_shape):
            raise PredictorLoadError("ONNX output rank does not match manifest")
        for actual, wanted in zip(output_shape, self._expected_output_shape, strict=True):
            if wanted is None and isinstance(actual, int):
                raise PredictorLoadError("ONNX output dynamic dimension does not match manifest")
            if wanted is not None and actual != wanted:
                raise PredictorLoadError("ONNX output shape does not match manifest")

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def device(self) -> str:
        providers = self.session.get_providers()
        return "CUDA" if "CUDAExecutionProvider" in providers else "CPU"

    def predict_batch(self, windows: Sequence[ClipWindow]) -> list[WindowPrediction]:
        if not windows:
            return []
        batch = np.stack([preprocess_frames(window.frames_rgb, self.preprocessing) for window in windows])
        try:
            raw = self.session.run(
                [self.preprocessing.output_name],
                {self.preprocessing.input_name: batch.astype(np.float32, copy=False)},
            )[0]
        except Exception as exc:
            raise PredictionError("ONNX prediction failed") from exc
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(values) != len(windows) or not np.all(np.isfinite(values)):
            raise PredictionError("ONNX output contains invalid scores")
        if self._output_semantics == "logit":
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
        if np.any(values < 0) or np.any(values > 1):
            raise PredictionError("Prediction scores must be probabilities in [0, 1]")
        return [
            WindowPrediction(window.index, window.start_sec, window.end_sec, window.center_sec, float(score))
            for window, score in zip(windows, values, strict=True)
        ]
