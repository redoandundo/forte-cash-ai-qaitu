"""Run the current API, sessions, progress and dossier on a team-owned GPU host."""
import argparse
import os
from pathlib import Path
import secrets
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="models/pretrained/qwen3-vl-2b-instruct")
    parser.add_argument("--bundle", default="artifacts/ml/bundles/baseline-audited-v4")
    parser.add_argument("--sequence", action="store_true", help="Use trained binary LoRA instead of base semantic Qwen")
    parser.add_argument("--adapter", default="models/qwen-sequence-adapter")
    parser.add_argument("--allow-experimental", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.sequence and not args.allow_experimental:
        parser.error("The trained adapter failed its quality gate; --allow-experimental is required")
    for name in ("QAITU_WORKER_URL", "QAITU_WORKER_KEY_FILE", "QAITU_SEQUENCE_ADAPTER"):
        os.environ.pop(name, None)
    directory = ROOT / "secrets"
    directory.mkdir(exist_ok=True)
    key_path = directory / "site.key"
    if not key_path.exists():
        key_path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    key = key_path.read_text("utf-8").strip()
    if len(key) < 32:
        raise RuntimeError("secrets/site.key is too short")
    os.environ.update({
        "FORTEBANK_API_KEY": key, "QAITU_INFERENCE_MODE": "onnx",
        "QAITU_MODEL_BUNDLE": str(Path(args.bundle).resolve()),
        "QAITU_ALLOW_DEVELOPMENT_MODEL": "true",
        "QAITU_EXPECTED_MODEL_SHA256": "00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d",
        "QAITU_PREFER_CUDA": "false", "QAITU_MAX_CONCURRENT_JOBS": "1",
        "QAITU_VLM_MODE": "off" if args.sequence else "qwen",
        "QAITU_VLM_MODEL_PATH": str(Path(args.model).resolve()),
        "QAITU_VLM_MANIFEST_PATH": str(ROOT / "configs/inference/qwen_verifier_manifest.json"),
        "QAITU_VLM_MAX_CONCURRENT_JOBS": "1",
    })
    if args.sequence:
        os.environ["QAITU_SEQUENCE_ADAPTER"] = str(Path(args.adapter).resolve())
        os.environ["QAITU_ALLOW_EXPERIMENTAL_ADAPTER"] = "true"
    from fortebank_audit.job_runtime import get_runtime
    runtime = get_runtime()  # Validate exact weights, runtime and CUDA before accepting traffic.
    addresses = sorted({entry[4][0] for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                        if not entry[4][0].startswith("127.")})
    links = [f"http://{address}:{args.port}" for address in addresses]
    (directory / "tester-access.txt").write_text(
        "Current worker protocol: qaitu-worker-v1\n"
        + "\n".join(links) + "\nAPI key (server only): " + key + "\n", encoding="utf-8")
    print(runtime.mode_label, flush=True)
    print("Worker addresses and private API key: secrets/tester-access.txt", flush=True)
    import uvicorn
    uvicorn.run("fortebank_audit.api:app", host="0.0.0.0", port=args.port, workers=1)


if __name__ == "__main__":
    main()
