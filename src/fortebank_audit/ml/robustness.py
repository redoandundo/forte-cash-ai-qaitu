"""Paired full-video OOF stress evaluation; never recalibrate on corruptions."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch

from .bundle import write_json
from .evaluation import calibration_report, score_sources
from .experiment_data import fingerprint, load_sources, source_identity, split_sources
from .quality import detailed_metrics, evaluated_rows


CONDITIONS = ("clean", "dark_070", "bright_130", "jpeg_35", "rotate_7")


def perturb(frames: np.ndarray, condition: str) -> np.ndarray:
    """Deterministic, temporally consistent RGB perturbation without mutation."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected uint8 THWC RGB frames")
    if condition not in CONDITIONS:
        raise ValueError("Unknown stress condition")
    if condition == "clean":
        return frames.copy()
    if condition in ("dark_070", "bright_130"):
        factor = 0.7 if condition == "dark_070" else 1.3
        return np.clip(frames.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    output = []
    for frame in frames:
        if condition == "jpeg_35":
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 35])
            if not ok:
                raise ValueError("Stress JPEG encoding failed")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Stress JPEG decoding failed")
            result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        else:
            height, width = frame.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 7, 1)
            result = cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        output.append(result)
    return np.stack(output)


def paired_changes(clean, changed):
    original = {row["source_id"]: row for row in clean}
    if len(original) != len(clean) or len({r["source_id"] for r in changed}) != len(changed) or set(original) != {r["source_id"] for r in changed}:
        raise ValueError("Stress comparison needs identical unique sources")
    if any(row["threshold"] != original[row["source_id"]]["threshold"] for row in changed):
        raise ValueError("Threshold must remain fixed across stress conditions")
    shifts = [abs(row["score"] - original[row["source_id"]]["score"]) for row in changed]
    return {
        "verdict_flips": sum(row["prediction"] != original[row["source_id"]]["prediction"] for row in changed),
        "correct_to_wrong": sum(original[row["source_id"]]["prediction"] == row["label"] != row["prediction"] for row in changed),
        "wrong_to_correct": sum(original[row["source_id"]]["prediction"] != row["label"] == row["prediction"] for row in changed),
        "mean_absolute_max_score_change": float(np.mean(shifts)),
        "max_absolute_max_score_change": float(max(shifts)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/ml/cache"))
    args = parser.parse_args()
    sources = load_sources(args.manifest, args.data_root)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identity = fingerprint([source_identity(row) for row in sorted(sources, key=lambda r: r["source_id"])])
    if identity != plan["dataset_sha256"]:
        raise ValueError("Dataset differs from frozen plan")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {
        "dataset_sha256": identity, "conditions": CONDITIONS,
        "threshold_selection": "clean_calibration_fold_only",
        "limitations": ["reused_development_dataset", "synthetic_rotation_is_not_a_new_camera", "JPEG_is_not_a_video_codec_test"],
    })
    combined = {condition: [] for condition in CONDITIONS}
    fold_results = []
    for outer in range(5):
        train, calibration_sources, test = split_sources(sources, plan["folds"], outer, (outer + 1) % 5)
        checkpoint = args.checkpoints / f"outer_{outer}/training/best.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if set(saved["provenance"]["train_source_ids"]) != {row["source_id"] for row in train}:
            raise ValueError("Checkpoint does not match frozen training partition")
        calibration = calibration_report(checkpoint, calibration_sources, args.cache)
        write_json(args.output / f"outer_{outer}/calibration.json", calibration)
        fold = {}
        for condition in CONDITIONS:
            rows, _, spec = score_sources(checkpoint, test, args.cache, frame_transform=partial(perturb, condition=condition))
            evaluated = evaluated_rows(rows, calibration["threshold"], spec.stride_sec)
            combined[condition].extend(evaluated)
            metrics = detailed_metrics(evaluated, test)["metrics"]
            write_json(args.output / f"outer_{outer}/{condition}.json", {"metrics": metrics, "videos": evaluated})
            fold[condition] = metrics
            print(json.dumps({"outer": outer, "condition": condition, "macro_f1": metrics["macro_f1"], "recall": metrics["recall"]}), flush=True)
        fold_results.append(fold)
    summary = {"conditions": {}, "folds": fold_results, "source_count": len(sources), "external_test": False}
    for condition, rows in combined.items():
        if len(rows) != len(sources) or {r["source_id"] for r in rows} != {r["source_id"] for r in sources}:
            raise ValueError("Incomplete or duplicated OOF coverage")
        summary["conditions"][condition] = {
            "metrics": detailed_metrics(rows, sources)["metrics"],
            "paired_changes": paired_changes(combined["clean"], rows),
        }
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
