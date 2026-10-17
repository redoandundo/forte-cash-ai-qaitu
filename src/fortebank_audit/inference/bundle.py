from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import PipelineConfig, PredictorLoadError
from .preprocessing import PreprocessingConfig


POSTPROCESSING = {
    "aggregation": "three_window_weighted_smoothing_then_peak",
    "smoothing_weights": [0.25, 0.5, 0.25],
    "smoothing_edges": "replicate",
    "short_sequence_rule": "identity_if_fewer_than_three",
    "event_merge": "adjacent_centers_within_stride",
    "max_events": 3,
}


@dataclass(frozen=True)
class ModelBundleContract:
    root: Path
    model_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    preprocessing: PreprocessingConfig
    pipeline_config: PipelineConfig
    release_status: str


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictorLoadError(f"Invalid bundle {label}") from exc
    if not isinstance(value, dict):
        raise PredictorLoadError(f"Bundle {label} must be a JSON object")
    return value


def _safe_file(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name:
        raise PredictorLoadError("Bundle contains an invalid file reference")
    path = (root / name).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise PredictorLoadError("Bundle file is missing or escapes the bundle")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _scoring_fingerprint(thresholds: dict[str, Any]) -> str:
    fields = (
        "checkpoint_sha256",
        "pipeline_sha256",
        "violation",
        *POSTPROCESSING,
    )
    try:
        payload = {key: thresholds[key] for key in fields}
    except KeyError as exc:
        raise PredictorLoadError("ML bundle scoring fingerprint fields are incomplete") from exc
    return _fingerprint(payload)


def _checksum_inventory(root: Path) -> dict[str, str]:
    try:
        lines = (root / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise PredictorLoadError("Bundle checksum inventory is missing") from exc
    entries: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise PredictorLoadError("Bundle checksum inventory is invalid") from exc
        if name in entries or len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise PredictorLoadError("Bundle checksum inventory is invalid")
        path = _safe_file(root, name)
        if _sha256(path) != digest.lower():
            raise PredictorLoadError(f"Bundle checksum mismatch: {name}")
        entries[name] = digest.lower()
    return entries


def load_bundle_contract(
    root: str | Path,
    *,
    allow_development_model: bool = False,
) -> ModelBundleContract:
    bundle_root = Path(root).resolve()
    if not bundle_root.is_dir():
        raise PredictorLoadError("Model bundle directory does not exist")
    manifest_path = bundle_root / "model_manifest.json"
    manifest = _json(manifest_path, "manifest")
    required_manifest = {
        "contract_version",
        "model_name",
        "model_version",
        "artifact",
        "sha256",
        "runtime",
        "input",
        "output",
        "preprocessing_file",
        "thresholds_file",
        "labels_file",
        "metrics_file",
    }
    missing = sorted(required_manifest - manifest.keys())
    if missing:
        raise PredictorLoadError(f"Bundle manifest missing fields: {', '.join(missing)}")
    if manifest["contract_version"] != 1 or manifest["runtime"] != "onnxruntime":
        raise PredictorLoadError("Unsupported bundle manifest contract")

    entries = _checksum_inventory(bundle_root)
    required_files = {
        "model_manifest.json",
        "ml-to-inference.md",
        "compatibility/model_dynamic.onnx",
        "compatibility/inference/model_manifest.json",
        str(manifest["artifact"]),
        str(manifest["preprocessing_file"]),
        str(manifest["thresholds_file"]),
        str(manifest["labels_file"]),
        str(manifest["metrics_file"]),
    }
    if not required_files <= entries.keys():
        raise PredictorLoadError("Checksum inventory does not cover the complete bundle")
    model_path = _safe_file(bundle_root, manifest["artifact"])
    if entries[str(manifest["artifact"])] != str(manifest["sha256"]).lower():
        raise PredictorLoadError("Manifest model hash disagrees with checksum inventory")

    preprocessing_path = _safe_file(bundle_root, manifest["preprocessing_file"])
    thresholds_path = _safe_file(bundle_root, manifest["thresholds_file"])
    labels_path = _safe_file(bundle_root, manifest["labels_file"])
    metrics_path = _safe_file(bundle_root, manifest["metrics_file"])
    raw_preprocessing = _json(preprocessing_path, "preprocessing")
    thresholds = _json(thresholds_path, "thresholds")
    labels = _json(labels_path, "labels")
    metrics = _json(metrics_path, "metrics")
    preprocessing = PreprocessingConfig.from_file(preprocessing_path)

    expected_preprocessing = {
        "temporal_span_fraction": 1.0,
        "resize": "opencv_inter_linear_stretch",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "sampling": "uniform_endpoint_false_nearest_pts",
        "include_tail_window": True,
        "timestamp_origin": "stream_start_time",
        "pipeline_version": "pts-tail-smooth-v1",
    }
    if any(raw_preprocessing.get(key) != value for key, value in expected_preprocessing.items()):
        raise PredictorLoadError("Unsupported ML bundle preprocessing contract")

    input_contract = manifest["input"]
    output_contract = manifest["output"]
    expected_shape = [
        1,
        3,
        preprocessing.num_frames,
        preprocessing.image_size,
        preprocessing.image_size,
    ]
    if not isinstance(input_contract, dict) or input_contract != {
        "name": preprocessing.input_name,
        "layout": preprocessing.input_layout,
        "shape": expected_shape,
        "dtype": preprocessing.dtype,
        "color": preprocessing.color,
    }:
        raise PredictorLoadError("Manifest and preprocessing input contracts disagree")
    if not isinstance(output_contract, dict) or output_contract != {
        "name": preprocessing.output_name,
        "semantics": "possible_violation_score",
        "activation": "logit",
        "positive_label": "possible_violation",
    }:
        raise PredictorLoadError("Unsupported ML bundle output contract")
    if labels.get("0") != "no_violation" or labels.get("1") != "possible_violation":
        raise PredictorLoadError("Unsupported ML bundle labels")

    threshold = thresholds.get("violation")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise PredictorLoadError("Invalid ML bundle threshold")
    if any(thresholds.get(key) != value for key, value in POSTPROCESSING.items()):
        raise PredictorLoadError("Unsupported ML bundle postprocessing contract")
    pipeline_sha = thresholds.get("pipeline_sha256")
    checkpoint_sha = thresholds.get("checkpoint_sha256")
    scoring_sha = thresholds.get("scoring_sha256")
    digests = (pipeline_sha, checkpoint_sha, scoring_sha)
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for digest in digests
    ):
        raise PredictorLoadError("ML bundle pipeline fingerprint is invalid")
    if scoring_sha != _scoring_fingerprint(thresholds):
        raise PredictorLoadError("ML bundle scoring fingerprint is invalid")
    if metrics.get("checkpoint_sha256") != checkpoint_sha:
        raise PredictorLoadError("ML bundle metrics refer to another checkpoint")
    if metrics.get("calibration_is_test") is not False:
        raise PredictorLoadError("ML bundle calibration metadata is invalid")
    calibration_metrics = metrics.get("calibration_metrics")
    if not isinstance(calibration_metrics, dict) or calibration_metrics.get(
        "threshold", threshold
    ) != threshold:
        raise PredictorLoadError("ML bundle calibration threshold is inconsistent")

    compatibility_path = _safe_file(
        bundle_root,
        "compatibility/inference/model_manifest.json",
    )
    compatibility = _json(compatibility_path, "compatibility manifest")
    compatibility_model = "compatibility/model_dynamic.onnx"
    if (
        compatibility.get("artifact") != compatibility_model
        or compatibility.get("sha256") != entries[compatibility_model]
        or compatibility.get("input_shape") != [None, *expected_shape[1:]]
        or compatibility.get("model_version") != manifest["model_version"]
        or compatibility.get("output_semantics") != "logit"
        or compatibility.get("preprocessing_sha256")
        != entries[str(manifest["preprocessing_file"])]
        or compatibility.get("threshold_sha256")
        != entries[str(manifest["thresholds_file"])]
    ):
        raise PredictorLoadError("ML bundle compatibility metadata is inconsistent")

    release_status = str(metrics.get("release_status", "missing"))
    if release_status not in {"promoted", "production_approved"} and not allow_development_model:
        raise PredictorLoadError("Development ML bundle requires explicit opt-in")
    config = PipelineConfig(
        window_sec=preprocessing.window_sec,
        stride_sec=preprocessing.stride_sec,
        num_frames=preprocessing.num_frames,
        threshold=float(threshold),
        aggregation=str(thresholds["aggregation"]),
        sampling="pts_nearest",
        short_sequence_rule=str(thresholds["short_sequence_rule"]),
        event_merge=str(thresholds["event_merge"]),
        include_tail_window=True,
        max_events=int(thresholds["max_events"]),
        batch_size=1,
    )
    return ModelBundleContract(
        root=bundle_root,
        model_path=model_path,
        manifest_path=manifest_path,
        manifest=manifest,
        preprocessing=preprocessing,
        pipeline_config=config,
        release_status=release_status,
    )
