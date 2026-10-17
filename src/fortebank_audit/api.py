from __future__ import annotations

import os
import secrets
import threading
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from .inference.contracts import InferenceError
from .job_runtime import RuntimeConfigurationError, get_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = Path(os.environ.get("FORTEBANK_UPLOAD_ROOT", PROJECT_ROOT / "artifacts" / "uploads"))
WEB_ROOT = PROJECT_ROOT / "web"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

STAGE_PROGRESS = {
    "upload_received": 5,
    "video_decode": 18,
    "temporal_windows": 32,
    "model_inference": 58,
    "aggregation": 72,
    "evidence": 86,
    "summary": 95,
    "done": 100,
    "failed": 100,
}


class AnalyzeAccepted(BaseModel):
    job_id: str


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame]


class JobResponse(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["no_violation", "violation"] | None = None
    violations: list[Violation] = Field(default_factory=list)
    error: str | None = None


class ProgressResponse(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    stage: str
    progress_percent: int = Field(ge=0, le=100)
    runtime_mode: str
    summary: str | None = None
    runtime: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


app = FastAPI(title="ForteBank Video Audit API", version="0.1.0")
_jobs: dict[str, dict[str, object]] = {}
_jobs_lock = threading.Lock()


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("FORTEBANK_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="API key is not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _set_job(job_key: str, **values: object) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_key, {}).update(values)


def _set_stage(job_id: str, stage: str) -> None:
    if stage not in STAGE_PROGRESS:
        raise ValueError("Unknown analysis stage")
    _set_job(job_id, stage=stage, progress_percent=STAGE_PROGRESS[stage])


def _public_error(exc: Exception) -> str:
    if isinstance(exc, RuntimeConfigurationError):
        return "Inference runtime is not configured"
    if isinstance(exc, InferenceError):
        return "Video analysis failed"
    return "Unexpected analysis error"


def _run_analysis(job_id: str, video_path: Path) -> None:
    try:
        runtime = get_runtime()
        _set_job(job_id, runtime_mode=runtime.mode_label)
        result = runtime.analyze(
            video_path,
            job_id=job_id,
            progress=lambda stage: _set_stage(job_id, stage),
        )
        _set_job(
            job_id,
            status="done",
            verdict=result.verdict,
            violations=result.violations,
            summary=result.summary,
            runtime=result.runtime,
            error=None,
        )
        _set_stage(job_id, "done")
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            verdict=None,
            violations=[],
            error=_public_error(exc),
        )
        _set_stage(job_id, "failed")
    finally:
        video_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, object]:
    try:
        runtime = get_runtime()
    except Exception:
        return {"status": "degraded", "model_ready": False, "runtime_mode": "unavailable"}
    return {"status": "ok", "model_ready": True, "runtime_mode": runtime.mode_label}


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
def create_analysis(background_tasks: BackgroundTasks, file: Annotated[UploadFile, File()]) -> AnalyzeAccepted:
    if Path(file.filename or "").suffix.lower() != ".mov":
        raise HTTPException(status_code=415, detail="Only .mov files are accepted")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    target = UPLOAD_ROOT / f"{job_id}.mov"
    size = 0
    try:
        with target.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds 100 MB")
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()

    _set_job(
        job_id,
        job_id=job_id,
        status="processing",
        verdict=None,
        violations=[],
        error=None,
        stage="upload_received",
        progress_percent=STAGE_PROGRESS["upload_received"],
        runtime_mode="Preparing runtime",
        summary=None,
        runtime={},
    )
    background_tasks.add_task(_run_analysis, job_id, target)
    return AnalyzeAccepted(job_id=job_id)


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobResponse,
    dependencies=[Depends(require_api_key)],
)
def get_job(job_id: str) -> JobResponse:
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse.model_validate(job)


@app.get(
    "/internal/api/v1/jobs/{job_id}/progress",
    response_model=ProgressResponse,
    dependencies=[Depends(require_api_key)],
)
def get_job_progress(job_id: str) -> ProgressResponse:
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return ProgressResponse.model_validate(job)


app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
