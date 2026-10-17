from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.bundle import write_json
from fortebank_audit.ml.evaluation import file_hash
from fortebank_audit.ml.experiment_data import load_sources, selected_training_windows
from fortebank_audit.ml.preprocessing import VideoSpec
from fortebank_audit.ml.video import cached_video


def main():
    parser = argparse.ArgumentParser(description="Prepare PTS-based RGB window cache and a private training-window inventory.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ml/baseline.yaml")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/ml/cache")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite clip inventory")
    sources = load_sources(args.manifest, args.data_root, args.review)
    spec = VideoSpec(**yaml.safe_load(args.config.read_text())["data"])
    clips = []
    for source in sources:
        path = Path(source["path"])
        if file_hash(path) != source["sha256"]:
            raise ValueError("Source checksum mismatch")
        samples = cached_video(path, spec, args.cache)
        for index, label, weight in selected_training_windows(source, samples):
            clips.append({
                "source_id": source["source_id"], "scenario_group": source["scenario_group"],
                "window_index": index, "label": label, "source_weight": weight,
                "start_sec": float(samples.starts[index]), "end_sec": float(samples.ends[index]),
                "actual_frame_pts_sec": samples.actual_timestamps[index].tolist(),
            })
    write_json(args.output, {"spec": spec.to_dict(), "clips": clips, "format": "raw uint8 RGB windows in npz cache; original PTS preserved"})
    print(json.dumps({"sources": len(sources), "training_windows": len(clips)}))


if __name__ == "__main__":
    main()
