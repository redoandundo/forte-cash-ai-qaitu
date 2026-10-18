"""Read-only update/checksum check; never merge code or replace running weights."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fetch:
        subprocess.run(["git", "fetch", "origin"], check=True)
    refs = {name: git("rev-parse", "origin/feature/" + name)
            for name in ("video-inference", "ml-model", "teamlead-ml-inference-integration")}
    manifest = json.loads(git("show", refs["video-inference"] + ":configs/inference/qwen_verifier_manifest.json"))
    weight = args.model / "model.safetensors"
    # The current contract uses one file. Reject layout changes rather than falsely verifying them.
    if manifest.get("weights_files") != ["model.safetensors"]:
        raise ValueError("New Qwen snapshot layout requires explicit integration review")
    before = weight.stat()
    with weight.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    after = weight.stat()
    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    report = {"local_commit": git("rev-parse", "HEAD"), "remote_commits": refs,
              "model_revision": manifest.get("revision"), "expected_sha256": manifest["weights_sha256"],
              "actual_sha256": actual, "stable_during_check": stable,
              "weights_match_remote_manifest": stable and actual == manifest["weights_sha256"]
                  and after.st_size == manifest["weights_bytes"],
              "scope": "identity_check_only_not_quality_validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["weights_match_remote_manifest"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
