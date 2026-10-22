from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jsonschema
import numpy as np
import onnxruntime as ort
import torch

from .evaluation import file_hash, pipeline_hash
from .models import build_model
from .preprocessing import VideoSpec, preprocessing_payload

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "configs" / "ml" / "model_manifest.v1.schema.json"


def cpu_session(path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("Missing file or path escapes the bundle")
    return path


def validate_bundle(root: Path) -> dict:
    manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    jsonschema.validate(manifest, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    required = ["model_manifest.json", "ml-to-inference.md", manifest["artifact"], manifest["preprocessing_file"], manifest["thresholds_file"], manifest["labels_file"], manifest["metrics_file"]]
    entries = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if name in entries or len(digest) != 64:
            raise ValueError("Invalid checksum inventory")
        entries[name] = digest
        if file_hash(safe_file(root, name)) != digest:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    if not set(required) <= entries.keys():
        raise ValueError("Checksum inventory does not cover the complete bundle")
    if entries[manifest["artifact"]] != manifest["sha256"]:
        raise ValueError("Manifest model hash mismatch")
    pre = json.loads(safe_file(root, manifest["preprocessing_file"]).read_text(encoding="utf-8"))
    thresholds = json.loads(safe_file(root, manifest["thresholds_file"]).read_text(encoding="utf-8"))
    labels = json.loads(safe_file(root, manifest["labels_file"]).read_text(encoding="utf-8"))
    if pre["input_layout"] != "NCTHW" or pre["dtype"] != "float32" or pre["color"] != "RGB" or pre["normalization"] != "imagenet":
        raise ValueError("Unsupported preprocessing")
    spec = VideoSpec(pre["num_frames"], pre["image_size"], pre["window_sec"], pre["recommended_stride_sec"], pre["include_tail_window"])
    if any(pre.get(key) != value for key, value in preprocessing_payload(spec).items()):
        raise ValueError("Preprocessing differs from the implemented pipeline")
    expected = [1, 3, pre["num_frames"], pre["image_size"], pre["image_size"]]
    if manifest["input"]["shape"] != expected or manifest["input"]["layout"] != pre["input_layout"] or manifest["input"]["dtype"] != pre["dtype"] or manifest["input"]["color"] != pre["color"]:
        raise ValueError("Manifest and preprocessing disagree")
    if manifest["output"]["activation"] != "logit" or labels.get("1") != manifest["output"]["positive_label"]:
        raise ValueError("Invalid output semantics or labels")
    if not np.isfinite(thresholds["violation"]) or not 0 <= thresholds["violation"] <= 1:
        raise ValueError("Invalid threshold")
    if thresholds["pipeline_sha256"] != pipeline_hash(spec):
        raise ValueError("Bundle was calibrated for another scoring pipeline")
    session = cpu_session(safe_file(root, manifest["artifact"]))
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != manifest["input"]["name"] or inputs[0].name != pre["input_name"] or inputs[0].shape != expected or inputs[0].type != "tensor(float)":
        raise ValueError("ONNX input does not match manifest")
    if len(outputs) != 1 or outputs[0].name != manifest["output"]["name"] or outputs[0].name != pre["output_name"] or outputs[0].shape != [1] or outputs[0].type != "tensor(float)":
        raise ValueError("ONNX output does not match manifest")
    return manifest


def export_bundle(checkpoint_path: Path, calibration: dict, output: Path, parity_inputs: np.ndarray, model_version: str, evaluation_metrics: dict | None = None, code_commit: str = "uncommitted") -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing model bundle")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if calibration["checkpoint_sha256"] != file_hash(checkpoint_path):
        raise ValueError("Calibration does not belong to this checkpoint")
    spec = VideoSpec(**checkpoint["spec"])
    if calibration["pipeline_sha256"] != pipeline_hash(spec):
        raise ValueError("Calibration pipeline changed before export")
    for train_key, cal_key in (("train_source_ids", "calibration_source_ids"), ("train_sha256", "calibration_sha256"), ("train_scenario_groups", "calibration_scenario_groups")):
        if set(checkpoint["provenance"][train_key]) & set(calibration[cal_key]):
            raise ValueError("Threshold calibration overlaps training")
    if checkpoint["normalization"] != "imagenet" or checkpoint["output_semantics"] != "logit":
        raise ValueError("Unsupported checkpoint semantics")
    expected_shape = (3, spec.num_frames, spec.image_size, spec.image_size)
    if parity_inputs.dtype != np.float32 or len(parity_inputs) < 2 or tuple(parity_inputs.shape[1:]) != expected_shape or not np.isfinite(parity_inputs).all():
        raise ValueError("Parity requires at least two valid preprocessed inputs")
    torch.set_num_threads(4)
    model = build_model(checkpoint["model"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ml-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        compat = staging / "compatibility" / "inference"
        compat.mkdir(parents=True)
        fixed_path = staging / "model.onnx"
        dynamic_path = staging / "compatibility" / "model_dynamic.onnx"
        sample = torch.from_numpy(parity_inputs[:1])
        for target, axes in ((fixed_path, None), (dynamic_path, {"video": {0: "batch"}, "logits": {0: "batch"}})):
            torch.onnx.export(model, sample, str(target), input_names=["video"], output_names=["logits"], dynamic_axes=axes, opset_version=17, do_constant_folding=True)
        with torch.inference_mode():
            reference = model(torch.from_numpy(parity_inputs)).numpy()
        fixed_session = cpu_session(fixed_path)
        dynamic_session = cpu_session(dynamic_path)
        fixed = np.concatenate([fixed_session.run(["logits"], {"video": item[None]})[0] for item in parity_inputs])
        dynamic = dynamic_session.run(["logits"], {"video": parity_inputs})[0]
        if any(not np.isfinite(values).all() or values.shape != (len(parity_inputs),) for values in (reference, fixed, dynamic)):
            raise ValueError("Invalid outputs during parity")
        error = max(float(np.abs(reference - fixed).max()), float(np.abs(reference - dynamic).max()))
        threshold = float(calibration["threshold"])
        agrees = all(np.array_equal(1 / (1 + np.exp(-reference)) >= threshold, 1 / (1 + np.exp(-values)) >= threshold) for values in (fixed, dynamic))
        if error > 1e-4 or not agrees:
            raise ValueError("Export parity or threshold decisions failed")
        parity = {"max_absolute_logit_error": error, "tolerance": 1e-4, "decision_agreement": agrees, "input_count": len(parity_inputs), "batch_sizes": [1, len(parity_inputs)], "provider": "CPUExecutionProvider"}
        write_json(staging / "preprocessing.json", preprocessing_payload(spec))
        write_json(staging / "thresholds.json", {
            "violation": threshold, "selected_on": "disjoint_calibration_sources",
            "min_recall": calibration["min_recall"], "metric": calibration["selection"],
            "aggregation": "three_window_weighted_smoothing_then_peak",
            "smoothing_weights": [0.25, 0.5, 0.25], "smoothing_edges": "replicate",
            "short_sequence_rule": "identity_if_fewer_than_three",
            "event_merge": "adjacent_centers_within_stride", "max_events": 3,
            "checkpoint_sha256": calibration["checkpoint_sha256"],
            "pipeline_sha256": calibration["pipeline_sha256"],
            "calibration_source_count": calibration["source_count"],
        })
        write_json(staging / "labels.json", {"0": "no_violation", "1": "possible_violation"})
        write_json(staging / "metrics.json", {
            "release_status": "development_only_not_promoted", "code_commit": code_commit,
            "calibration_metrics": calibration["metrics"], "calibration_is_test": False,
            "nested_oof_metrics": evaluation_metrics, "parity": parity,
            "config_sha256": checkpoint["provenance"]["config_sha256"],
            "training_dataset_sha256": checkpoint["provenance"]["dataset_sha256"],
            "checkpoint_sha256": calibration["checkpoint_sha256"],
            "limitations": ["small_development_dataset", "point_annotations", "ranking_score_not_probability"],
        })
        manifest = {
            "contract_version": 1, "model_name": checkpoint["model"], "model_version": model_version,
            "artifact": "model.onnx", "sha256": file_hash(fixed_path), "runtime": "onnxruntime",
            "input": {"name": "video", "layout": "NCTHW", "shape": [1, *expected_shape], "dtype": "float32", "color": "RGB"},
            "output": {"name": "logits", "semantics": "possible_violation_score", "activation": "logit", "positive_label": "possible_violation"},
            "preprocessing_file": "preprocessing.json", "thresholds_file": "thresholds.json", "labels_file": "labels.json", "metrics_file": "metrics.json",
        }
        write_json(staging / "model_manifest.json", manifest)
        # The existing inference adapter resolves paths relative to manifest.parents[2].
        # This path layout makes that root the bundle directory, without machine-specific paths.
        legacy_manifest = {
            "contract_version": 1, "model_name": checkpoint["model"], "model_version": model_version,
            "artifact": "compatibility/model_dynamic.onnx", "sha256": file_hash(dynamic_path), "runtime": "onnxruntime",
            "input_name": "video", "input_shape": [None, *expected_shape], "input_layout": "NCTHW",
            "dtype": "float32", "color": "RGB", "output_name": "logits", "output_shape": [None],
            "output_semantics": "logit", "positive_label": "possible_violation",
            "preprocessing_file": "preprocessing.json", "preprocessing_sha256": file_hash(staging / "preprocessing.json"),
            "threshold_file": "thresholds.json", "threshold_sha256": file_hash(staging / "thresholds.json"),
        }
        write_json(compat / "model_manifest.json", legacy_manifest)
        handoff = (
            "# ML bundle handoff\n\n"
            f"Model: {model_version}. Branch: feature/ml-model. Code commit: {code_commit}.\n\n"
            "Status: development only; numerical parity does not establish recognition quality.\n\n"
            "Canonical model.onnx uses fixed batch 1 and the team's nested manifest v1. "
            "The existing inference predictor uses compatibility/model_dynamic.onnx and "
            "compatibility/inference/model_manifest.json together. Both graphs use the same checkpoint.\n\n"
            f"Input: float32 RGB NCTHW, [N,3,{spec.num_frames},{spec.image_size},{spec.image_size}]. "
            f"Window {spec.window_sec}s, stride {spec.stride_sec}s, PTS sampling, tail enabled.\n\n"
            f"Output: one logit per window; sigmoid exactly once. Threshold: {threshold:.17g}. "
            "Apply the smoothing and event rules in thresholds.json. Scores are ranking scores.\n\n"
            f"PyTorch/ONNX maximum absolute logit error: {error:.9g}; threshold decisions agree: {agrees}.\n\n"
            "See metrics.json for separate calibration and nested OOF results. "
            "Calibration is not a test score. JPEG evidence and complete API timing require integration verification.\n\n"
            "Smoke command from the ML repository, after extracting this directory to artifacts/ml/bundle:\n\n"
            "`python scripts/ml/infer_onnx.py sample.mov --bundle artifacts/ml/bundle`\n\n"
            "Training and export recipes: docs/handoffs/ml-to-inference.md in the stated code commit.\n"
        )
        (staging / "ml-to-inference.md").write_text(handoff, encoding="utf-8")
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / "SHA256SUMS.txt").write_text("".join(file_hash(path) + "  " + path.relative_to(staging).as_posix() + "\n" for path in files), encoding="ascii")
        validate_bundle(staging)
        # A complete, validated directory becomes visible atomically. Existing bundles are immutable.
        staging.rename(output)
    return {"model_version": model_version, "sha256": manifest["sha256"], "parity": parity}
