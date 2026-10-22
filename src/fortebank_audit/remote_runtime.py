"""Server-side bridge to a compatible team-owned GPU worker."""
from __future__ import annotations

import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

from .inference.contracts import InferenceError


WORKER_PROTOCOL = "qaitu-worker-v1"


class RemoteWorkerError(InferenceError):
    pass


class RemoteRuntime:
    def __init__(self, url: str, key_file: str, *, client=None):
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise RemoteWorkerError("Worker URL must be an origin without credentials")
        key = Path(key_file).read_text("utf-8").strip()
        if len(key) < 32:
            raise RemoteWorkerError("Worker key must contain at least 32 characters")
        self.client = client or httpx.Client(base_url=url.rstrip("/"), timeout=10,
                                             follow_redirects=False, trust_env=False)
        self._headers = {"X-API-Key": key}
        try:
            health = self._json("GET", "/health")
            if (health.get("worker_protocol") != WORKER_PROTOCOL
                    or health.get("status") != "ok" or not health.get("model_ready")
                    or health.get("execution_location") == "remote"
                    or health.get("inference_mode") != "onnx"):
                raise RemoteWorkerError("Incompatible GPU worker; update to the teamlead build")
            catalog = self._json("GET", "/internal/api/v1/rules")
            if len(catalog.get("rules", [])) != 3:
                raise RemoteWorkerError("GPU worker rules contract is incompatible")
            self._diagnostics = {k: v for k, v in health.items()
                                 if k not in {"status", "model_ready", "runtime_mode"}}
            self.mode_label = str(health["runtime_mode"])
        except RemoteWorkerError:
            raise
        except Exception:
            raise RemoteWorkerError("GPU worker startup failed") from None

    @property
    def model_diagnostics(self):
        return {**self._diagnostics, "execution_location": "remote"}

    def _json(self, method, path, **kwargs):
        response = self.client.request(method, path, headers=self._headers, **kwargs)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RemoteWorkerError("GPU worker returned an invalid object")
        return result

    def analyze(self, video_path, *, job_id, progress=None):
        from .api import JobResponse, ProgressResponse
        from .job_runtime import RuntimeResult
        started = time.monotonic()
        try:
            with Path(video_path).open("rb") as stream:
                accepted = self._json("POST", "/api/v1/analyze",
                                      files={"file": ("upload.mov", stream, "video/quicktime")},
                                      timeout=30)
            remote_id = accepted.get("job_id")
            if (not isinstance(remote_id, str) or not remote_id
                    or len(remote_id) > 64 or not remote_id.isalnum()):
                raise RemoteWorkerError("GPU worker returned an invalid job identifier")
            while time.monotonic() - started < 300:
                detail = ProgressResponse.model_validate(self._json(
                    "GET", f"/internal/api/v1/jobs/{remote_id}/progress"))
                if detail.job_id != remote_id:
                    raise RemoteWorkerError("GPU worker returned another job")
                if detail.status == "failed":
                    raise RemoteWorkerError("GPU worker analysis failed")
                if progress and detail.stage in {"upload_received", "video_decode", "temporal_windows",
                        "model_inference", "aggregation", "evidence", "summary"}:
                    progress(detail.stage)
                if detail.status == "done":
                    result = JobResponse.model_validate(self._json("GET", f"/api/v1/jobs/{remote_id}"))
                    if (result.job_id != remote_id or result.status != "done"
                            or result.verdict not in {"violation", "no_violation"}
                            or bool(result.violations) != (result.verdict == "violation")
                            or len(detail.analysis_details) != len(result.violations)
                            or [d.event_index for d in detail.analysis_details] != list(range(1, len(result.violations)+1))):
                        raise RemoteWorkerError("GPU worker result is incomplete")
                    if time.monotonic() - started > 300:
                        raise RemoteWorkerError("GPU worker exceeded 300 seconds")
                    return RuntimeResult(
                        verdict=result.verdict, violations=[v.model_dump() for v in result.violations],
                        analysis_details=[d.model_dump() for d in detail.analysis_details],
                        summary=detail.summary or "Анализ завершён.",
                        runtime={**detail.runtime, **self.model_diagnostics},
                    )
                time.sleep(0.5)
            raise RemoteWorkerError("GPU worker exceeded 300 seconds")
        except RemoteWorkerError:
            raise
        except Exception:
            # Do not expose credentials, worker addresses or response payloads to the browser.
            raise RemoteWorkerError("GPU worker request failed") from None


def build_remote_runtime():
    url = os.environ.get("QAITU_WORKER_URL", "").strip()
    key_file = os.environ.get("QAITU_WORKER_KEY_FILE", "").strip()
    if not url or not key_file:
        raise RemoteWorkerError("Both QAITU_WORKER_URL and QAITU_WORKER_KEY_FILE are required")
    return RemoteRuntime(url, key_file)
