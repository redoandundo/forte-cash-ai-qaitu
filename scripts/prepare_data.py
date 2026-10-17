from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the ForteBank dataset safely.")
    parser.add_argument("archive", type=Path, help="Path to open.zip")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        source.extractall(destination)


def main() -> None:
    args = parse_args()
    safe_extract(args.archive, args.out)
    print(f"Extracted {args.archive.resolve()} to {args.out.resolve()}")


if __name__ == "__main__":
    main()
