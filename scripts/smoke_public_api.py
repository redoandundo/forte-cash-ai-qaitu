"""Check a public demo using generated video only; never uploads dataset media."""
from __future__ import annotations

import argparse
import base64
import json
import tempfile
import time
from pathlib import Path

import av
import cv2
import httpx
import numpy as np


def generate_video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 160, "yuv420p"
        for index in range(90):
            frame = np.full((160, 160, 3), 35, dtype=np.uint8)
            cv2.rectangle(frame, (index % 90, 40), (index % 90 + 40, 100), (180, 90, 60), -1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def analyze(client: httpx.Client, name: str, content: bytes) -> tuple[dict, float]:
    started = time.perf_counter()
    accepted = client.post("/api/v1/analyze", files={"file": (name, content, "video/quicktime")})
    assert accepted.status_code == 202, f"Upload status {accepted.status_code}"
    job_id = accepted.json()["job_id"]
    while time.perf_counter() - started < 300:
        response = client.get(f"/api/v1/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        assert job["status"] in {"processing", "done", "failed"}
        if job["status"] != "processing":
            elapsed = time.perf_counter() - started
            assert elapsed < 300, "Exceeded five minute limit"
            return job, elapsed
        time.sleep(0.5)
    raise TimeoutError("No result within 300 seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    key = args.key_file.read_text(encoding="utf-8").strip()
    with httpx.Client(base_url=args.base_url, timeout=30) as anonymous:
        page = anonymous.get("/")
        assert page.status_code == 200 and "Qaitu" in page.text
        health = anonymous.get("/health").json()
        assert health["predictor_class"] == "OnnxPredictor"
        assert health["model_sha256"] == "00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d"
        for url in ("/api/v1/jobs/unknown", "/internal/api/v1/rules", "/internal/api/v1/jobs/unknown/progress"):
            assert anonymous.get(url).status_code == 401
            assert anonymous.get(url, headers={"X-API-Key": "wrong"}).status_code == 401
        assert anonymous.post("/api/v1/analyze", files={"file": ("test.mov", b"synthetic", "video/quicktime")}).status_code == 401
        for url in ("/secrets/public-demo-api-key.txt", "/artifacts/ml/bundles/baseline-audited-v4/model.onnx"):
            assert anonymous.get(url).status_code == 404
    with httpx.Client(base_url=args.base_url, headers={"X-API-Key": key}, timeout=30) as client:
        assert len(client.get("/internal/api/v1/rules").json()["rules"]) == 3
        assert client.get("/api/v1/jobs/unknown").status_code == 404
        assert client.post("/api/v1/analyze", files={"file": ("bad.txt", b"synthetic", "text/plain")}).status_code == 415
        with tempfile.TemporaryDirectory(prefix="qaitu-public-smoke-") as directory:
            video = Path(directory) / "synthetic.mov"
            generate_video(video)
            good, good_sec = analyze(client, "synthetic.mov", video.read_bytes())
        assert good["status"] == "done"
        assert good["verdict"] in {"violation", "no_violation"}
        assert bool(good["violations"]) == (good["verdict"] == "violation")
        for event in good["violations"]:
            assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
            assert 1 <= len(event["evidence_frames"]) <= 3
            for frame in event["evidence_frames"]:
                jpeg = base64.b64decode(frame["image_base64"], validate=True)
                assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) is not None
        bad, bad_sec = analyze(client, "corrupt.mov", b"synthetic-corrupt-video")
        assert bad["status"] == "failed" and bad["verdict"] is None and bad["violations"] == []
    result = {"public_url": args.base_url, "status": "passed", "auth": "missing/wrong keys rejected",
              "model": health["model_version"], "synthetic_duration_sec": 9,
              "synthetic_verdict": good["verdict"], "synthetic_events": len(good["violations"]),
              "upload_to_result_sec": round(good_sec, 3), "corrupt_failed_sec": round(bad_sec, 3),
              "dataset_uploaded": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
