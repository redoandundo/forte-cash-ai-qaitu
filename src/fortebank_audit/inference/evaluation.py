from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import PROJECT_ROOT
from .contracts import PipelineConfig
from .pipeline import score_video
from .predictor import OnnxPredictor
from .preprocessing import PreprocessingConfig


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    predicted = scores >= threshold
    return {
        "tn": int(np.sum((labels == 0) & ~predicted)),
        "fp": int(np.sum((labels == 0) & predicted)),
        "fn": int(np.sum((labels == 1) & ~predicted)),
        "tp": int(np.sum((labels == 1) & predicted)),
    }


def metrics(matrix: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = matrix["tp"], matrix["fp"], matrix["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "f2": f2}


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = [(p > n) + 0.5 * (p == n) for p in positives for n in negatives]
    return float(np.mean(comparisons))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ONNX video-level scores on a manifest split")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--selection", choices=("reference", "manifest"), default="reference")
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "mobilenet_noaug_full_video.json",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "open" / "videos")
    args = parser.parse_args()
    if args.selection == "reference" and args.split != "validation":
        raise ValueError("The reference report contains the historical validation set only")

    preprocessing = PreprocessingConfig.from_file(PROJECT_ROOT / "models" / "preprocessing.json")
    artifact_threshold = float(
        json.loads((PROJECT_ROOT / "models" / "thresholds.json").read_text(encoding="utf-8"))["violation"]
    )
    config = PipelineConfig(
        window_sec=preprocessing.window_sec,
        stride_sec=preprocessing.stride_sec,
        num_frames=preprocessing.num_frames,
        threshold=artifact_threshold,
    )
    predictor = OnnxPredictor(
        PROJECT_ROOT / "models" / "best_model.onnx",
        preprocessing,
        PROJECT_ROOT / "configs" / "inference" / "model_manifest.json",
        prefer_cuda=False,
    )
    manifest = json.loads((PROJECT_ROOT / "analysis" / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest_ids = {source["source_id"] for source in manifest["sources"] if source["split"] == args.split}
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    reference_ids = {video["source_id"] for video in reference["videos"]}
    selected_ids = reference_ids if args.selection == "reference" else manifest_ids
    records = []
    for source in manifest["sources"]:
        if source["source_id"] not in selected_ids:
            continue
        path = next((args.data_root / alias for alias in source["aliases"] if (args.data_root / alias).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No local alias for source {source['source_id']}")
        metadata, predictions, processing_sec, _ = score_video(path, predictor, config)
        peak = min(predictions, key=lambda item: (-item.score, item.center_sec))
        records.append(
            {
                "source_id": source["source_id"],
                "video": path.name,
                "label": int(source["label"]),
                "score": peak.score,
                "peak_time_sec": peak.center_sec,
                "events_sec": source["events_sec"],
                "duration_sec": metadata.duration_sec,
                "processing_sec": processing_sec,
            }
        )

    if not records:
        raise RuntimeError(f"No local videos matched the selected {args.split!r} source set")

    labels = np.asarray([record["label"] for record in records], dtype=int)
    scores = np.asarray([record["score"] for record in records], dtype=float)
    candidates = sorted({0.0, 1.0, *scores.tolist()})
    ranked = []
    for threshold in candidates:
        matrix = confusion(labels, scores, threshold)
        ranked.append((metrics(matrix)["f2"], metrics(matrix)["recall"], -matrix["fp"], threshold, matrix))
    _, _, _, selected_threshold, selected_matrix = max(ranked)
    artifact_matrix = confusion(labels, scores, artifact_threshold)
    negative_minutes = sum(record["duration_sec"] for record in records if record["label"] == 0) / 60.0
    timestamp_errors = [
        min(abs(record["peak_time_sec"] - event) for event in record["events_sec"])
        for record in records
        if record["label"] == 1 and record["events_sec"]
    ]
    payload = {
        "split": args.split,
        "selection": args.selection,
        "effective_source_set": (
            "reference_report_validation" if args.selection == "reference" else f"manifest_{args.split}"
        ),
        "sources": len(records),
        "manifest_source_ids": sorted(manifest_ids),
        "reference_source_ids": sorted(reference_ids),
        "split_mismatch": manifest_ids != reference_ids,
        "provider": predictor.device,
        "artifact_threshold": artifact_threshold,
        "artifact_confusion": artifact_matrix,
        "artifact_metrics": metrics(artifact_matrix),
        "roc_auc": roc_auc(labels, scores),
        "false_positive_videos_per_negative_minute": (
            artifact_matrix["fp"] / negative_minutes if negative_minutes else None
        ),
        "timestamp_mae_sec": float(np.mean(timestamp_errors)) if timestamp_errors else None,
        "same_split_best_threshold": selected_threshold,
        "same_split_best_confusion": selected_matrix,
        "same_split_best_metrics": metrics(selected_matrix),
        "records": records,
        "warning": "The best threshold is diagnostic only because it is selected on the evaluated split.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
