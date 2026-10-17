from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fortebank_audit.ml.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained classifier to ONNX and verify numerical parity.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "best_model.onnx")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    run_metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    model = build_model(checkpoint["model"], pretrained=False, train_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    frames, image_size = int(checkpoint["frames"]), int(checkpoint["image_size"])
    sample = torch.randn(1, 3, frames, image_size, image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.output,
        input_names=["video"],
        output_names=["logits"],
        dynamic_axes={"video": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    with torch.inference_mode():
        torch_output = model(sample).numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["logits"], {"video": sample.numpy()})[0]
    max_error = float(np.max(np.abs(torch_output - onnx_output)))
    if max_error > 1e-4:
        raise RuntimeError(f"ONNX parity failed: max error {max_error}")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    threshold = float(run_metrics.get("threshold", run_metrics.get("source_metrics", {}).get("threshold", 0.5)))
    (PROJECT_ROOT / "models" / "preprocessing.json").write_text(
        json.dumps(
            {
                "model": checkpoint["model"],
                "input_layout": "NCTHW",
                "input_name": "video",
                "output_name": "logits",
                "dtype": "float32",
                "color": "RGB",
                "num_frames": frames,
                "image_size": image_size,
                "normalization": checkpoint["normalization"],
                "window_sec": 8.0,
                "temporal_span_fraction": checkpoint.get("temporal_span", 1.0),
                "recommended_stride_sec": 2.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "models" / "thresholds.json").write_text(
        json.dumps({"violation": threshold, "selected_on": "full_validation_videos", "metric": "f2"}, indent=2),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "models" / "checksums.sha256").write_text(f"{digest}  best_model.onnx\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": digest, "max_parity_error": max_error, "size_mb": round(args.output.stat().st_size / 1024**2, 2)}, indent=2))


if __name__ == "__main__":
    main()
