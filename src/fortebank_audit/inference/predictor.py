from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import onnxruntime as ort

from .bundle import load_bundle_contract
from .contracts import ClipWindow, PredictionError, PredictorLoadError, WindowPrediction
from .preprocessing import PreprocessingConfig, preprocess_frames


class Predictor(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def runtime_providers(self) -> tuple[str, ...]: ...

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

    @property
    def runtime_providers(self) -> tuple[str, ...]:
        return ("MockExecutionProvider",)

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


def _resolve_manifest_contract(
    model_path: Path,
    preprocessing: PreprocessingConfig,
    manifest_path: Path,
    manifest: dict,
    *,
    allow_development_model: bool,
) -> tuple[str, str, list[object], list[object]]:
    if manifest.get("contract_version") == 1 and "input" in manifest:
        contract = load_bundle_contract(
            manifest_path.parent,
            allow_development_model=allow_development_model,
        )
        if contract.model_path != model_path.resolve():
            raise PredictorLoadError("Model artifact path does not match the bundle manifest")
        if contract.preprocessing != preprocessing:
            raise PredictorLoadError("Bundle preprocessing does not match the predictor")
        return (
            str(manifest["model_version"]),
            str(manifest["output"]["activation"]),
            list(manifest["input"]["shape"]),
            [1],
        )

    required = {
        "model_version",
        "artifact",
        "sha256",
        "input",
        "output",
        "window_sec",
        "stride_sec",
        "threshold",
        "aggregation",
        "sampling",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise PredictorLoadError(f"Model manifest missing fields: {', '.join(missing)}")
    input_contract = manifest["input"]
    output_contract = manifest["output"]
    if not isinstance(input_contract, dict) or not isinstance(output_contract, dict):
        raise PredictorLoadError("Model input/output contracts must be objects")
    if output_contract.get("activation") != "sigmoid":
        raise PredictorLoadError("The fallback model must expose one binary logit")
    if output_contract.get("classes") != ["no_violation", "violation"]:
        raise PredictorLoadError("Unsupported output classes")
    preprocessing_contract = {
        "input_name": preprocessing.input_name,
        "output_name": preprocessing.output_name,
        "input_layout": preprocessing.input_layout,
        "dtype": preprocessing.dtype,
        "color": preprocessing.color,
    }
    manifest_contract = {
        "input_name": input_contract.get("name"),
        "output_name": output_contract.get("name"),
        "input_layout": input_contract.get("layout"),
        "dtype": input_contract.get("dtype"),
        "color": input_contract.get("color"),
    }
    mismatches = [
        field for field, value in preprocessing_contract.items() if manifest_contract[field] != value
    ]
    if mismatches:
        raise PredictorLoadError(
            f"Model manifest disagrees with preprocessing: {', '.join(mismatches)}"
        )
    expected_input_shape = input_contract.get("shape")
    if expected_input_shape != [
        None,
        3,
        preprocessing.num_frames,
        preprocessing.image_size,
        preprocessing.image_size,
    ]:
        raise PredictorLoadError("Model manifest input shape disagrees with preprocessing")
    if not math.isclose(float(manifest["window_sec"]), preprocessing.window_sec, abs_tol=1e-9):
        raise PredictorLoadError("Model manifest window disagrees with preprocessing")
    if not math.isclose(float(manifest["stride_sec"]), preprocessing.stride_sec, abs_tol=1e-9):
        raise PredictorLoadError("Model manifest stride disagrees with preprocessing")
    bundle_root = manifest_path.resolve().parent
    artifact_path = (bundle_root / str(manifest["artifact"])).resolve()
    if artifact_path != model_path.resolve():
        raise PredictorLoadError("Model artifact path does not match the manifest")
    try:
        thresholds = json.loads((bundle_root / "thresholds.json").read_text(encoding="utf-8"))
        checksum_entries = {
            name: digest.lower()
            for line in (bundle_root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
            if line.strip()
            for digest, name in [line.split(maxsplit=1)]
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PredictorLoadError("Model threshold or checksum contract is invalid") from exc
    if not math.isclose(
        float(thresholds.get("violation", -1)), float(manifest["threshold"]), abs_tol=1e-12
    ):
        raise PredictorLoadError("Model threshold disagrees with manifest")
    if thresholds.get("aggregation") != manifest["aggregation"]:
        raise PredictorLoadError("Model aggregation disagrees with manifest")
    if manifest["sampling"] != "opencv_frame_index_round":
        raise PredictorLoadError("Unsupported fallback-model sampling strategy")
    if not model_path.is_file():
        raise PredictorLoadError("Model artifact does not exist")
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash.lower() != str(manifest["sha256"]).lower():
        raise PredictorLoadError("Model checksum mismatch")
    if checksum_entries.get(model_path.name) != actual_hash.lower():
        raise PredictorLoadError("Model checksum file disagrees with artifact")
    return str(manifest["model_version"]), "logit", list(expected_input_shape), [None]


class OnnxPredictor:
    def __init__(
        self,
        model_path: str | Path,
        preprocessing: PreprocessingConfig,
        manifest_path: str | Path,
        *,
        prefer_cuda: bool = True,
        session: object | None = None,
        allow_development_model: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        manifest_path = Path(manifest_path)
        self.preprocessing = preprocessing
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PredictorLoadError("Model manifest is missing or invalid") from exc
        (
            self._model_version,
            self._output_semantics,
            self._expected_shape,
            self._expected_output_shape,
        ) = _resolve_manifest_contract(
            self.model_path,
            preprocessing,
            manifest_path,
            manifest,
            allow_development_model=allow_development_model,
        )

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

    @property
    def runtime_providers(self) -> tuple[str, ...]:
        return tuple(str(provider) for provider in self.session.get_providers())

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
