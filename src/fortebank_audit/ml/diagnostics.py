"""Local-only full-video traces; outputs contain private dataset diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .bundle import cpu_session, validate_bundle, write_json
from .metrics import events_from_scores, smooth_scores
from .preprocessing import VideoSpec, normalize_rgb
from .video import sample_video


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def audit_inputs(frames: np.ndarray, tensors: np.ndarray) -> dict:
    if len(frames) != len(tensors) or len(frames) == 0 or not np.isfinite(tensors).all():
        raise ValueError("Invalid diagnostic inputs")
    if frames.ndim != 5 or frames.dtype != np.uint8 or tensors.shape != (len(frames), 3, frames.shape[1], frames.shape[2], frames.shape[3]) or tensors.dtype != np.float32:
        raise ValueError("Expected RGB uint8 windows and float32 NCTHW inputs")
    frame_hashes = [array_hash(item) for item in frames]
    tensor_hashes = [array_hash(item) for item in tensors]
    changed = [i for i in range(1, len(frames)) if frame_hashes[i] != frame_hashes[i - 1]]
    collapsed = [i for i in changed if tensor_hashes[i] == tensor_hashes[i - 1]]
    if collapsed:
        raise ValueError("Different sampled frames collapsed to identical model inputs")
    return {"frame_hashes": frame_hashes, "tensor_hashes": tensor_hashes,
            "changed_adjacent_windows": len(changed), "collapsed_changed_windows": len(collapsed),
            "unique_tensors": len(set(tensor_hashes)), "window_count": len(frames)}


def trace_video(video: Path, bundle: Path, *, check_order: bool = True) -> dict:
    manifest = validate_bundle(bundle)
    pre = json.loads((bundle / "preprocessing.json").read_text(encoding="utf-8"))
    threshold = json.loads((bundle / "thresholds.json").read_text(encoding="utf-8"))["violation"]
    spec = VideoSpec(pre["num_frames"], pre["image_size"], pre["window_sec"], pre["recommended_stride_sec"], pre["include_tail_window"])
    samples = sample_video(video, spec)
    tensors = np.stack([normalize_rgb(frames, spec.image_size) for frames in samples.frames])
    audit = audit_inputs(samples.frames, tensors)
    session = cpu_session(bundle / manifest["artifact"])

    def infer(inputs):
        result = np.asarray([session.run([pre["output_name"]], {pre["input_name"]: item[None]})[0].item() for item in inputs])
        if not np.isfinite(result).all():
            raise ValueError("Non-finite ONNX logits")
        return result

    logits = infer(tensors)
    reorder_error = float(np.max(np.abs(infer(tensors[::-1])[::-1] - logits))) if check_order else None
    if reorder_error is not None and reorder_error > 1e-6:
        raise ValueError("ONNX result depends on window traversal order")
    scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    smoothed = smooth_scores(scores)
    windows = [{"window_index": i, "start_sec": float(samples.starts[i]), "end_sec": float(samples.ends[i]),
                "center_sec": float(samples.centers[i]), "raw_logit": float(logits[i]),
                "sigmoid_score": float(scores[i]), "smoothed_score": float(smoothed[i]),
                "threshold": threshold, "is_positive": bool(smoothed[i] >= threshold),
                "input_sha256": audit["tensor_hashes"][i], "frames_sha256": audit["frame_hashes"][i],
                "input_mean": float(tensors[i].mean()), "input_std": float(tensors[i].std()),
                "sampled_pts_sec": samples.actual_timestamps[i].tolist()} for i in range(len(logits))]
    events = events_from_scores(samples.starts, samples.ends, smoothed, threshold, spec.stride_sec)
    for event in events:
        event["supporting_window_indices"] = [row["window_index"] for row in windows if row["is_positive"]
                                       and row["start_sec"] >= event["start_sec"] - 1e-6
                                       and row["end_sec"] <= event["end_sec"] + 1e-6]
        event["supporting_windows"] = len(event["supporting_window_indices"])
    return {"model_version": manifest["model_version"], "duration_sec": samples.duration,
            "spec": spec.to_dict(), "threshold": threshold, "window_count": len(windows),
            "max_score": float(smoothed.max()), "peak_time_sec": float(samples.centers[int(smoothed.argmax())]),
            "positive_windows": int(np.sum(smoothed >= threshold)), "events": events,
            "input_audit": audit, "reversed_window_order_max_logit_error": reorder_error,
            "raw_logit_range": float(np.ptp(logits)), "windows": windows,
            "evaluation_role": "runtime_diagnostic_not_independent_quality_estimate"}


def trace_mock(video: Path, reference_package: Path) -> dict:
    """Execute the actual team's MockPredictor, with the reported live settings."""
    sys.path.insert(0, str(reference_package.resolve().parent))
    predictor_module = importlib.import_module(reference_package.name + ".predictor")
    contracts = importlib.import_module(reference_package.name + ".contracts")
    spec = VideoSpec(window_sec=8, stride_sec=2)
    samples = sample_video(video, spec)
    windows = [contracts.ClipWindow(i, float(start), float(end), float(center), frames)
               for i, (start, end, center, frames) in enumerate(zip(samples.starts, samples.ends, samples.centers, samples.frames))]
    predictor = predictor_module.MockPredictor([0.1, 0.2, 0.8, 0.9, 0.3])
    scores = np.array([item.score for item in predictor.predict_batch(windows)])
    blank = [contracts.ClipWindow(w.index, w.start_sec, w.end_sec, w.center_sec, np.zeros_like(w.frames_rgb)) for w in windows]
    assert scores.tolist() == [item.score for item in predictor.predict_batch(blank)]
    smoothed = smooth_scores(scores)
    return {"spec": spec.to_dict(), "threshold": 0.5, "content_invariant": True,
            "raw_peak_time_sec": float(samples.centers[int(scores.argmax())]),
            "smoothed_peak_time_sec": float(samples.centers[int(smoothed.argmax())]),
            "windows": [{"window_index": w.index, "start_sec": w.start_sec, "end_sec": w.end_sec,
                         "center_sec": w.center_sec, "raw_logit": None,
                         "sigmoid_score": None, "mock_direct_score": float(score),
                         "smoothed_score": float(value), "threshold": 0.5, "is_positive": bool(value >= 0.5)}
                        for w, score, value in zip(windows, scores, smoothed)],
            "note": "Mock emits scores directly, not logits; smoothing shown explicitly for diagnosis."}


def shifted_video(source: Path, target: Path, seconds: float) -> float:
    """Lossless diagnostic copy preserving decoded PTS; positive offset holds the first frame.

    A held frame is an artificial static prefix, not an annotated negative action.
    Frame positions are never reconstructed from an average FPS.
    """
    if target.exists():
        raise FileExistsError(target)
    with av.open(str(source)) as reader:
        stream = reader.streams.video[0]
        rate = stream.base_rate or stream.average_rate
        if not rate:
            raise ValueError("Shift fixture requires a nominal encoding rate")
        actual = float(seconds)
        origin = float((stream.start_time or 0) * stream.time_base)
        target.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(target), "w") as writer:
            output = writer.add_stream("ffv1", rate=rate)
            output.width, output.height, output.pix_fmt = stream.width, stream.height, "bgr0"
            output.time_base = output.codec_context.time_base = Fraction(1, 1000000)
            written = 0

            def emit(rgb, timestamp):
                nonlocal written
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = round(timestamp * 1000000), Fraction(1, 1000000)
                for packet in output.encode(frame):
                    writer.mux(packet)
                written += 1

            for index, frame in enumerate(reader.decode(stream)):
                if frame.pts is None:
                    raise ValueError("Missing PTS in shift source")
                timestamp = float(frame.pts * frame.time_base) - origin
                if timestamp + actual < -1e-9:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                if index == 0 and actual > 0:
                    for prefix_time in np.arange(0, actual, 1 / float(rate)):
                        emit(rgb, float(prefix_time))
                emit(rgb, max(0., timestamp + actual))
            if not written:
                raise ValueError("Trim removed all video frames")
            for packet in output.encode():
                writer.mux(packet)
    return actual


def compare_shift(original: dict, shifted: dict, offset: float) -> dict:
    tolerance = float(original["spec"]["stride_sec"])
    error = abs(shifted["peak_time_sec"] - original["peak_time_sec"] - offset)
    pairs = []
    for row in original["windows"]:
        matches = [other for other in shifted["windows"] if abs(other["start_sec"] - row["start_sec"] - offset) < 0.002
                   and abs(other["end_sec"] - row["end_sec"] - offset) < 0.002]
        if matches:
            other = matches[0]
            pairs.append({"same_tensor": row["input_sha256"] == other["input_sha256"],
                          "logit_error": abs(row["raw_logit"] - other["raw_logit"])})
    return {"offset_sec": offset, "original_peak_sec": original["peak_time_sec"],
            "shifted_peak_sec": shifted["peak_time_sec"], "peak_shift_error_sec": error,
            "tolerance_sec": tolerance, "peak_shift_passed": error <= tolerance + .002,
            "matched_windows": len(pairs), "identical_matched_tensors": sum(p["same_tensor"] for p in pairs),
            "input_translation_passed": bool(pairs) and all(p["same_tensor"] and p["logit_error"] <= 1e-6 for p in pairs),
            "max_matched_logit_error": max((p["logit_error"] for p in pairs), default=None),
            "event_present_original": bool(original["events"]), "event_present_shifted": bool(shifted["events"])}


def save_trace(target: Path, trace: dict) -> None:
    write_json(target.with_suffix(".json"), trace)
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace["windows"][0]))
        writer.writeheader()
        writer.writerows(trace["windows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Private ignored output directory")
    parser.add_argument("--reference-package", type=Path)
    parser.add_argument("--shift-seconds", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trace = trace_video(args.video, args.bundle)
    save_trace(args.output / "onnx", trace)
    summary = {key: value for key, value in trace.items() if key not in {"windows", "input_audit"}}
    if args.reference_package:
        mock = trace_mock(args.video, args.reference_package)
        save_trace(args.output / "mock", mock)
        summary["mock"] = {key: value for key, value in mock.items() if key != "windows"}
    if args.shift_seconds:
        summary["shifts"] = []
        for name, seconds in (("prepend", abs(args.shift_seconds)), ("trim", -abs(args.shift_seconds))):
            path = args.output / (name + ".mkv")
            actual = shifted_video(args.video, path, seconds)
            shifted = trace_video(path, args.bundle)
            save_trace(args.output / name, shifted)
            summary["shifts"].append(compare_shift(trace, shifted, actual))
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if any(not row["peak_shift_passed"] or not row["input_translation_passed"] for row in summary.get("shifts", [])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
