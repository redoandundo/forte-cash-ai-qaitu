from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "inference" / "qwen_verifier_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    target = PROJECT_ROOT / manifest["local_dir"]
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=manifest["model_id"],
        revision=manifest["revision"],
        local_dir=target,
    )
    weights = target / manifest["weights_file"]
    actual = sha256(weights)
    if actual.lower() != manifest["weights_sha256"].lower():
        raise RuntimeError(f"Qwen weights checksum mismatch: {actual}")
    print(json.dumps({"model_dir": str(target), "sha256": actual}, indent=2))


if __name__ == "__main__":
    main()
