from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .bundle import validate_bundle
from .metrics import events_from_scores, smooth_scores
from .preprocessing import VideoSpec, normalize_rgb
from .video import sample_video


def predict_video(path: Path, bundle: Path) -> dict:
    started = time.perf_counter()
    # Every failure propagates; decoding or contract errors are never negative verdicts.
    manifest = validate_bundle(bundle)
    pre = json.loads((bundle / manifest["preprocessing_file"]).read_text(encoding="utf-8"))
    thresholds = json.loads((bundle / manifest["thresholds_file"]).read_text(encoding="utf-8"))
    spec = VideoSpec(pre["num_frames"], pre["image_size"], pre["window_sec"], pre["recommended_stride_sec"], pre["include_tail_window"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(bundle / manifest["artifact"]), sess_options=options, providers=["CPUExecutionProvider"])
    samples = sample_video(path, spec)
    scores = []
    for frames in samples.frames:
        tensor = normalize_rgb(frames, spec.image_size)[None]
        logits = session.run([pre["output_name"]], {pre["input_name"]: tensor})[0]
        if np.asarray(logits).shape != (1,) or not np.isfinite(logits).all():
            raise ValueError("Invalid ONNX output")
        scores.append(float(1 / (1 + np.exp(-np.clip(float(logits[0]), -80, 80)))))
    values = smooth_scores(scores)
    events = events_from_scores(samples.starts, samples.ends, values, thresholds["violation"], spec.stride_sec)
    return {
        "model_version": manifest["model_version"], "label": "possible_violation" if events else "no_violation",
        "score": float(values.max()), "score_semantics": "ranking_score_not_probability",
        "threshold": thresholds["violation"], "events": events, "windows": len(values),
        "duration_sec": samples.duration, "runtime_sec": time.perf_counter() - started,
        "runtime_scope": "bundle_validation_decode_preprocess_CPU_ONNX_events_no_API_or_JPEG",
        "device": "CPU",
    }
