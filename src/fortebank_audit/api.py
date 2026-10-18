from __future__ import annotations

import os
import secrets
import threading
import uuid
from copy import deepcopy
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from .inference.contracts import (
    EvidenceExtractionError,
    InferenceError,
    PredictionError,
    PredictorLoadError,
    UnsupportedVideoError,
    VideoOpenError,
)
from .job_runtime import RuntimeConfigurationError, get_runtime
from .rules import match_reviewed_rule, rule_catalog, unmatched_rule
from .semantic_adapter import SemanticIntegrationError


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


class RuleReference(BaseModel):
    status: Literal["matched", "not_established"]
    label: str
    document_name: str | None = None
    section: str | None = None
    source_url: str | None = None
    applicability: str


class AnalysisDetail(BaseModel):
    event_index: int = Field(ge=1)
    thesis: str
    observations: list[str] = Field(min_length=1)
    reasoning: str
    risk: str
    rule: RuleReference
    rules: list[RuleReference] = Field(default_factory=list)


class RuleReviewRequest(BaseModel):
    model_config = {"extra": "forbid"}
    rule_id: str | None = Field(default=None, min_length=1, max_length=32)
    confirmed_facts: list[str] = Field(default_factory=list, max_length=8)


class ProgressResponse(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    stage: str
    progress_percent: int = Field(ge=0, le=100)
    runtime_mode: str
    summary: str | None = None
    runtime: dict[str, object] = Field(default_factory=dict)
    analysis_details: list[AnalysisDetail] = Field(default_factory=list)
    error: str | None = None



@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail before serving traffic when the explicitly selected runtime or model is invalid.
    get_runtime()
    yield


app = FastAPI(title="ForteBank Video Audit API", version="0.1.0", lifespan=lifespan)
_jobs: dict[str, dict[str, object]] = {}
_jobs_lock = threading.Lock()


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("FORTEBANK_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="API key is not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@app.get("/internal/api/v1/rules", dependencies=[Depends(require_api_key)])
def get_rules() -> dict[str, object]:
    return rule_catalog()


@app.put("/internal/api/v1/jobs/{job_id}/events/{event_index}/rule-review",
         response_model=AnalysisDetail, dependencies=[Depends(require_api_key)])
def review_event_rule(job_id: str, event_index: int, review: RuleReviewRequest) -> AnalysisDetail:
    if review.rule_id is None and review.confirmed_facts:
        raise HTTPException(status_code=422, detail="Reset cannot contain confirmed observations")
    try:
        rule = match_reviewed_rule(review.rule_id, review.confirmed_facts) if review.rule_id else unmatched_rule()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail="Review requires completed analysis")
        details = job.get("analysis_details", [])
        detail = next((item for item in details if item["event_index"] == event_index), None)
        if detail is None or not 1 <= event_index <= len(job["violations"]):
            raise HTTPException(status_code=404, detail="Event not found")
        originals = job.setdefault("rule_review_original_descriptions", {})
        violation = job["violations"][event_index - 1]
        original = originals.setdefault(event_index, violation["description"])
        original_rules = job.setdefault("rule_review_original_rules", {}).setdefault(
            event_index, deepcopy({"rule": detail["rule"], "rules": detail.get("rules", [])})
        )
        rules_update = original_rules if review.rule_id is None else {"rule": rule, "rules": [rule]}
        updated_detail = {**detail, **rules_update}
        response = AnalysisDetail.model_validate(updated_detail)
        detail.update(deepcopy(rules_update))
        violation["description"] = original if review.rule_id is None else (
            f"{original} Ручная проверка: {rule['label']}; {rule['document_name']}, "
            f"{rule['section']}. {rule['applicability']} Источник: {rule['source_url']}"
        )
    return response


def _set_job(job_key: str, **values: object) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_key, {}).update(values)


def _set_stage(job_id: str, stage: str) -> None:
    if stage not in STAGE_PROGRESS:
        raise ValueError("Unknown analysis stage")
    _set_job(job_id, stage=stage, progress_percent=STAGE_PROGRESS[stage])


def _public_error(exc: Exception) -> str:
    if isinstance(exc, SemanticIntegrationError):
        return "Локальная проверка Qwen не завершена: проверьте модель и совместимость inference"
    if isinstance(exc, RuntimeConfigurationError):
        return "Локальный ONNX runtime не настроен"
    if isinstance(exc, UnsupportedVideoError):
        return "Видеокодек не поддерживается локальным декодером"
    if isinstance(exc, VideoOpenError):
        return "Видео повреждено или не может быть декодировано"
    if isinstance(exc, PredictorLoadError):
        return "Не удалось загрузить или проверить ONNX-модель"
    if isinstance(exc, PredictionError):
        return "Локальная ONNX-модель завершилась технической ошибкой"
    if isinstance(exc, EvidenceExtractionError):
        return "Не удалось сформировать обязательные JPEG-кадры"
    if isinstance(exc, InferenceError):
        return "Локальный анализ видео завершился технической ошибкой"
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
            analysis_details=result.analysis_details,
            error=None,
        )
        _set_stage(job_id, "done")
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            verdict=None,
            violations=[],
            analysis_details=[],
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
    return {
        "status": "ok",
        "model_ready": True,
        "runtime_mode": runtime.mode_label,
        **runtime.model_diagnostics,
    }


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
        analysis_details=[],
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
