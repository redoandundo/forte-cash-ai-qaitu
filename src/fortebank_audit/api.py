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

from .analyzer import analyze


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("FORTEBANK_MODEL_PATH", PROJECT_ROOT / "artifacts" / "baseline.joblib"))
UPLOAD_ROOT = Path(os.environ.get("FORTEBANK_UPLOAD_ROOT", PROJECT_ROOT / "artifacts" / "uploads"))
WEB_ROOT = PROJECT_ROOT / "web"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class AnalyzeAccepted(BaseModel):
    job_id: str


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    confidence: float = Field(ge=0, le=1)
    evidence_frames: list[EvidenceFrame]


class JobResponse(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["no_violation", "violation"] | None = None
    violations: list[Violation] = Field(default_factory=list)
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


def _set_job(job_id: str, **values: object) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(values)


def _run_analysis(job_id: str, video_path: Path) -> None:
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        result = analyze(video_path, MODEL_PATH)
        _set_job(job_id, status="done", **result)
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc))
    finally:
        video_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


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

    _set_job(job_id, job_id=job_id, status="processing", verdict=None, violations=[], error=None)
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


app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")

