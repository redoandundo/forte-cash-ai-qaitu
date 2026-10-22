from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fortebank_audit.ml.runtime import predict_video


def main():
    parser = argparse.ArgumentParser(description="ML smoke test: validated bundle, PTS windows, one sigmoid, separated events.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict_video(args.video, args.bundle)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite an inference report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
