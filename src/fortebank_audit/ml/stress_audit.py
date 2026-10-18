"""OOF stress tests at clean calibration thresholds; never tune on outer videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .bundle import write_json
from .evaluation import assert_unseen, file_hash, pipeline_hash, score_sources
from .experiment_data import fingerprint, load_sources, source_identity, split_sources
from .preprocessing import VideoSpec
from .quality import detailed_metrics, evaluated_rows

CONDITIONS = ("clean", "dark", "bright", "translate", "jpeg")


def transform_frames(frames: np.ndarray, condition: str) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError("Unknown stress condition")
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3:
        raise ValueError("Stress input must be uint8 RGB THWC")
    if condition == "clean":
        return frames.copy()
    if condition in ("dark", "bright"):
        factor = 0.65 if condition == "dark" else 1.35
        return np.clip(frames.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    height, width = frames.shape[1:3]
    if condition == "translate":
        # One transform for the whole clip, not independent frame jitter.
        matrix = np.float32([[1, 0, width * 0.05], [0, 1, height * 0.05]])
        return np.stack([cv2.warpAffine(frame, matrix, (width, height),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for frame in frames])
    output = []
    for frame in frames:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 40])
        if not ok:
            raise ValueError("Stress JPEG encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Stress JPEG decoding failed")
        output.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    return np.stack(output)


def validate_fold(checkpoint_path, calibration, train, cal, test):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    provenance = checkpoint["provenance"]
    train_identity = fingerprint([source_identity(row) for row in sorted(train, key=lambda row: row["source_id"])])
    if provenance["dataset_sha256"] != train_identity:
        raise ValueError("Checkpoint training labels or annotations differ from frozen data")
    for field, key, cal_key in (
        ("source_id", "train_source_ids", "calibration_source_ids"),
        ("sha256", "train_sha256", "calibration_sha256"),
        ("scenario_group", "train_scenario_groups", "calibration_scenario_groups"),
    ):
        if set(provenance[key]) != {row[field] for row in train}:
            raise ValueError("Checkpoint does not match frozen training partition")
        if set(calibration[cal_key]) != {row[field] for row in cal}:
            raise ValueError("Calibration does not match frozen calibration partition")
        if {row[field] for row in test} & set(calibration[cal_key]):
            raise ValueError("Outer data overlap calibration")
    assert_unseen(test + cal, provenance)
    if calibration["checkpoint_sha256"] != file_hash(checkpoint_path):
        raise ValueError("Calibration checkpoint mismatch")
    if calibration["pipeline_sha256"] != pipeline_hash(VideoSpec(**checkpoint["spec"])):
        raise ValueError("Calibration pipeline mismatch")
    if not np.isfinite(calibration["threshold"]) or not 0 <= calibration["threshold"] <= 1:
        raise ValueError("Invalid calibration threshold")


def grouped_intervals(rows: list[dict], repetitions: int = 2000, seed: int = 42) -> dict:
    """Resample complete scenario groups, preserving dependence within pairs."""
    if not rows or repetitions < 1:
        raise ValueError("Bootstrap needs rows and positive repetitions")
    groups = {}
    for row in rows:
        counts = groups.setdefault(row["scenario_group"], np.zeros(4))
        counts[2 * row["label"] + row["prediction"]] += 1
    counts = np.stack([groups[key] for key in sorted(groups)])
    rng = np.random.default_rng(seed)
    draws = counts[rng.integers(0, len(counts), (repetitions, len(counts)))].sum(axis=1)
    # Skip samples with no support for one of the true classes.
    draws = draws[(draws[:, :2].sum(1) > 0) & (draws[:, 2:].sum(1) > 0)]
    if not len(draws):
        raise ValueError("Bootstrap requires both classes")
    tn, fp, fn, tp = draws.T
    pos = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    neg = 2 * tn / np.maximum(2 * tn + fp + fn, 1)
    values = {"positive_f1": pos, "macro_f1": (pos + neg) / 2,
              "recall": tp / (tp + fn), "precision": tp / np.maximum(tp + fp, 1)}
    return {"method": "scenario_group_percentile_bootstrap_of_fixed_oof_predictions",
            "limitation": "conditional_intervals_exclude_retraining_and_model_selection_uncertainty",
            "seed": seed, "repetitions": repetitions, "valid_repetitions": len(draws),
            "group_count": len(groups),
            "intervals_95": {name: np.quantile(value, [.025, .975]).tolist() for name, value in values.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="outer_N/training/best.pt and calibration.json")
    parser.add_argument("--calibrations", type=Path, help="Separate outer_N/calibration.json directory for rescored checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/ml/cache"))
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    sources = load_sources(args.manifest, args.data_root)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identity = fingerprint([source_identity(row) for row in sorted(sources, key=lambda row: row["source_id"])])
    if identity != plan["dataset_sha256"] or set(plan["folds"].values()) != set(range(5)):
        raise ValueError("Data or folds differ from the five-fold frozen protocol")
    # Validate the complete protocol before spending inference time.
    jobs = []
    for outer in range(5):
        train, cal, test = split_sources(sources, plan["folds"], outer, (outer + 1) % 5)
        folder = args.runs / f"outer_{outer}"
        checkpoint = folder / "training/best.pt"
        calibration_folder = (args.calibrations / f"outer_{outer}") if args.calibrations else folder
        calibration = json.loads((calibration_folder / "calibration.json").read_text(encoding="utf-8"))
        validate_fold(checkpoint, calibration, train, cal, test)
        jobs.append((outer, checkpoint, calibration, test))
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"dataset_sha256": identity, "threshold_policy": "frozen_clean_calibration_per_fold",
               "scoring_backend": {"device": "cuda" if torch.cuda.is_available() else "cpu",
                   "torch_version": str(torch.__version__), "batch_size": args.batch_size,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
               "checkpoint_sha256": [file_hash(checkpoint) for _, checkpoint, _, _ in jobs],
               "calibration_thresholds": [calibration["threshold"] for _, _, calibration, _ in jobs],
               "protocol": "five_fold_scenario_disjoint_reused_development_data",
               "conditions": {}, "limitations": ["synthetic_transforms_are_not_new_camera_validation",
                   "binary_labels_only_no_event_type_accuracy", "point_annotations_not_complete_intervals"],
               "transforms": {"dark": "RGB x0.65", "bright": "RGB x1.35 clipped",
                              "translate": "5 percent right/down with reflected border", "jpeg": "quality 40 after resize"}}
    clean = None
    for condition in CONDITIONS:
        rows = []
        for outer, checkpoint, calibration, test in jobs:
            raw, _, spec = score_sources(checkpoint, test, args.cache, args.batch_size,
                frame_transform=lambda frames: transform_frames(frames, condition))
            evaluated = evaluated_rows(raw, calibration["threshold"], spec.stride_sec)
            rows.extend(evaluated)
            write_json(args.output / condition / f"outer_{outer}.json", {"videos": evaluated})
        if len(rows) != len(sources) or len({row["source_id"] for row in rows}) != len(sources):
            raise ValueError("OOF must cover every source exactly once")
        result = detailed_metrics(rows, sources)
        result.pop("private_scenario_groups")
        result["uncertainty"] = grouped_intervals(rows)
        if clean is None:
            clean = {row["source_id"]: row for row in rows}
        result["decision_flips_from_clean"] = sum(row["prediction"] != clean[row["source_id"]]["prediction"] for row in rows)
        result["mean_absolute_score_shift"] = float(np.mean([abs(row["score"] - clean[row["source_id"]]["score"]) for row in rows]))
        result["macro_f1_delta_from_clean"] = result["metrics"]["macro_f1"] - summary["conditions"].get("clean", result)["metrics"]["macro_f1"]
        summary["conditions"][condition] = result
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"condition": condition, "metrics": result["metrics"], "flips": result["decision_flips_from_clean"]}), flush=True)


if __name__ == "__main__":
    main()
