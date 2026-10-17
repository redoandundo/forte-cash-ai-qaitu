from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fortebank_audit.baseline import train
from fortebank_audit.dataset import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local temporal baseline.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "baseline.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(build_index(args.data_root), args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Model: {args.out.resolve()}")


if __name__ == "__main__":
    main()
