from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .experiment_data import fingerprint
from .metrics import calibrate_threshold, events_from_scores, match_events, smooth_scores, summarize_rows
from .models import build_model
from .preprocessing import VideoSpec, normalize_rgb
from .video import cached_video


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline_hash(spec: VideoSpec) -> str:
    folder = Path(__file__).parent
    # Git may convert LF/CRLF on checkout; line endings do not change the pipeline.
    code = {name: hashlib.sha256((folder / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for name in ("video.py", "preprocessing.py", "metrics.py")}
    return fingerprint({"spec": spec.to_dict(), "code": code})


def assert_unseen(sources: list[dict], provenance: dict) -> None:
    fields = {"source_id": "train_source_ids", "sha256": "train_sha256", "scenario_group": "train_scenario_groups"}
    for field, saved in fields.items():
        if saved not in provenance:
            raise ValueError("Checkpoint lacks split provenance")
        if {row[field] for row in sources} & set(provenance[saved]):
            raise ValueError(f"Evaluation overlaps training: {field}")


def score_sources(checkpoint_path: Path, sources: list[dict], cache: Path, batch_size: int = 8):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert_unseen(sources, checkpoint["provenance"])
    spec = VideoSpec(**checkpoint["spec"])
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rows = []
    for source in sources:
        started = time.perf_counter()
        path = Path(source["path"])
        if file_hash(path) != source["sha256"]:
            raise ValueError("Source video checksum mismatch")
        samples = cached_video(path, spec, cache)
        values = []
        with torch.inference_mode():
            for offset in range(0, len(samples.frames), batch_size):
                tensor = np.stack([normalize_rgb(frames, spec.image_size) for frames in samples.frames[offset:offset + batch_size]])
                logits = model(torch.from_numpy(tensor).to(device)).float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite model output")
                values.extend(torch.sigmoid(logits).cpu().tolist())
        scores = smooth_scores(values)
        truth = source.get("events_sec", [])
        if source.get("events"):
            truth = [(item["start_sec"] + item["end_sec"]) / 2 for item in source["events"]]
        rows.append({
            "source_id": source["source_id"], "sha256": source["sha256"],
            "scenario_group": source["scenario_group"], "label": source["label"],
            "annotation_confidence": source.get("annotation_confidence", "high"),
            "duration_sec": samples.duration, "starts": samples.starts.tolist(),
            "ends": samples.ends.tolist(), "window_scores": scores.tolist(),
            "score": float(scores.max()), "truth_points": truth,
            "scoring_with_cache_sec": time.perf_counter() - started,
        })
    return rows, checkpoint, spec


def calibration_report(checkpoint_path: Path, sources: list[dict], cache: Path, min_recall: float = 0.75) -> dict:
    rows, checkpoint, spec = score_sources(checkpoint_path, sources, cache)
    selected = calibrate_threshold([row["label"] for row in rows], [row["score"] for row in rows], min_recall)
    return {
        "threshold": selected["threshold"], "metrics": selected,
        "min_recall": min_recall, "selection": "macro_f1_subject_to_recall",
        "checkpoint_sha256": file_hash(checkpoint_path), "pipeline_sha256": pipeline_hash(spec),
        "calibration_source_ids": [row["source_id"] for row in rows],
        "calibration_sha256": [row["sha256"] for row in rows],
        "calibration_scenario_groups": sorted({row["scenario_group"] for row in rows}),
        "source_count": len(rows), "model": checkpoint["model"], "spec": spec.to_dict(),
        "training_provenance": checkpoint["provenance"],
    }


def evaluation_report(checkpoint_path: Path, sources: list[dict], cache: Path, calibration: dict) -> dict:
    if calibration["checkpoint_sha256"] != file_hash(checkpoint_path):
        raise ValueError("Threshold belongs to a different checkpoint")
    for field, key in (("source_id", "calibration_source_ids"), ("sha256", "calibration_sha256"), ("scenario_group", "calibration_scenario_groups")):
        if {row[field] for row in sources} & set(calibration[key]):
            raise ValueError(f"Evaluation overlaps threshold calibration: {field}")
    rows, _, spec = score_sources(checkpoint_path, sources, cache)
    if calibration["pipeline_sha256"] != pipeline_hash(spec):
        raise ValueError("Calibration and evaluation pipelines differ")
    threshold = calibration["threshold"]
    for row in rows:
        events = events_from_scores(row.pop("starts"), row.pop("ends"), row.pop("window_scores"), threshold, spec.stride_sec)
        row["events"] = events
        row["prediction"] = int(bool(events))
        row["event_match"] = match_events([event["peak_time_sec"] for event in events], row["truth_points"])
    return {"threshold": threshold, "checkpoint_sha256": file_hash(checkpoint_path), "pipeline_sha256": pipeline_hash(spec), "metrics": summarize_rows(rows), "videos": rows}


def promotion_decision(baseline: dict, candidate: dict, folds: list[dict]) -> dict:
    conditions = {
        "macro_f1": candidate["macro_f1"] >= max(0.64, baseline["macro_f1"]),
        "recall": candidate["recall"] >= max(0.75, baseline["recall"]),
        "precision": candidate["precision"] >= baseline["precision"],
        "false_positive_videos_per_minute": candidate["false_positive_videos_per_minute"] <= min(0.60, baseline["false_positive_videos_per_minute"]),
        "false_positive_events_per_minute": candidate["false_positive_events_per_minute"] <= min(0.60, baseline["false_positive_events_per_minute"]),
        "timestamp_mae_sec": candidate["timestamp_mae_sec"] is not None and candidate["timestamp_mae_sec"] <= 5,
        "event_recall_within_5_sec": candidate["event_recall_within_5_sec"] is not None and candidate["event_recall_within_5_sec"] >= 0.70,
        "fold_stability": len(folds) == 5 and all(row["recall"] > 0 and row["macro_f1"] >= 0.35 for row in folds),
    }
    return {"quality_passed": all(conditions.values()), "conditions": conditions, "deployment_ready": False, "remaining": ["final_checkpoint_calibration", "ONNX_parity", "full_API_latency"]}
