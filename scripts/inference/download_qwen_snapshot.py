"""Download the public pinned snapshot; never send project data externally."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/inference/qwen_verifier_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["weights_files"] != ["model.safetensors"]:
        raise ValueError("Unsupported snapshot layout")
    snapshot_download(repo_id=manifest["model_id"], revision=manifest["revision"],
                      local_dir=args.output, allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors", "LICENSE*"],
                      max_workers=2)
    weights = args.output / "model.safetensors"
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if weights.stat().st_size != manifest["weights_bytes"] or digest != manifest["weights_sha256"]:
        raise ValueError("Downloaded snapshot does not match pinned weights")
    print(json.dumps({"status": "verified", "sha256": digest, "bytes": weights.stat().st_size}))


if __name__ == "__main__":
    main()
