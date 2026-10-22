"""Local acceptance check; never downloads models or starts a public server."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fortebank_audit.job_runtime import InferenceRuntime, RuntimeSettings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Optional local video for full inference acceptance")
    args = parser.parse_args()
    try:
        settings = RuntimeSettings.from_environment()
        if settings.vlm_mode != "qwen":
            raise ValueError("This check requires QAITU_VLM_MODE=qwen")
        started = time.perf_counter()
        runtime = InferenceRuntime(settings)
        report = dict(status="startup_passed", startup_sec=round(time.perf_counter() - started, 3),
                      **runtime.model_diagnostics)
        if args.video is not None:
            if not args.video.is_file():
                raise ValueError("Video does not exist")
            started = time.perf_counter()
            result = runtime.analyze(args.video, job_id=f"acceptance-{uuid.uuid4().hex}")
            elapsed = time.perf_counter() - started
            report.update(status="passed" if elapsed <= 300 else "time_limit_failed",
                          processing_sec=round(elapsed, 3), verdict=result.verdict,
                          events=len(result.violations),
                          evidence_counts=[len(e["evidence_frames"]) for e in result.violations],
                          # Aggregate counts only; no model-generated bank-video text.
                          candidate_events=result.runtime["diagnostics"]["semantic"].get("candidate_events"),
                          qwen_called=result.runtime["diagnostics"]["semantic"].get("qwen_called"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] in {"passed", "startup_passed"} else 1
    except Exception as exc:
        # Exception text can contain private paths; keep the console report shareable.
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__,
                              next_step="Check local inference revision, snapshot manifest, CUDA and environment settings")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
