from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .inference.bundle import load_bundle_contract
from .inference.contracts import InferenceError, PipelineConfig
from .inference.pipeline import ProgressCallback, analyze_video
from .inference.predictor import MockPredictor, OnnxPredictor, Predictor
from .semantic_adapter import SemanticAdapter, map_semantic_event


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeConfigurationError(f"{name} must be a boolean value")


class RuntimeConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeSettings:
    mode: Literal["mock", "onnx"]
    bundle_root: Path | None
    allow_development_model: bool
    evidence_root: Path
    expected_model_sha256: str | None = None
    prefer_cuda: bool = True
    max_concurrent_jobs: int = 1
    vlm_mode: Literal["off", "qwen"] = "off"

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        mode_value = os.environ.get("QAITU_INFERENCE_MODE")
        if mode_value is None or not mode_value.strip():
            raise RuntimeConfigurationError(
                "QAITU_INFERENCE_MODE must be explicitly set to mock or onnx"
            )
        raw_mode = mode_value.strip().lower()
        if raw_mode not in {"mock", "onnx"}:
            raise RuntimeConfigurationError("QAITU_INFERENCE_MODE must be mock or onnx")
        vlm_mode = os.environ.get("QAITU_VLM_MODE", "off").strip().lower()
        if vlm_mode in {"", "disabled"}:
            vlm_mode = "off"
        if vlm_mode not in {"off", "qwen"}:
            raise RuntimeConfigurationError("QAITU_VLM_MODE must be off or qwen")
        if vlm_mode == "qwen" and raw_mode != "onnx":
            raise RuntimeConfigurationError("Qwen requires an ONNX detector, not mock mode")
        bundle_value = os.environ.get("QAITU_MODEL_BUNDLE", "").strip()
        bundle_root = Path(bundle_value).resolve() if bundle_value else None
        if raw_mode == "onnx" and bundle_root is None:
            raise RuntimeConfigurationError("ONNX mode requires QAITU_MODEL_BUNDLE")
        allow_development = _environment_bool(
            "QAITU_ALLOW_DEVELOPMENT_MODEL", default=False
        )
        expected_model_sha256 = (
            os.environ.get("QAITU_EXPECTED_MODEL_SHA256", "").strip().lower() or None
        )
        if expected_model_sha256 is not None and (
            len(expected_model_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_model_sha256)
        ):
            raise RuntimeConfigurationError(
                "QAITU_EXPECTED_MODEL_SHA256 must be a lowercase SHA-256 digest"
            )
        prefer_cuda = _environment_bool("QAITU_PREFER_CUDA", default=True)
        try:
            max_concurrent_jobs = int(os.environ.get("QAITU_MAX_CONCURRENT_JOBS", "1"))
        except ValueError as exc:
            raise RuntimeConfigurationError(
                "QAITU_MAX_CONCURRENT_JOBS must be an integer"
            ) from exc
        if not 1 <= max_concurrent_jobs <= 8:
            raise RuntimeConfigurationError(
                "QAITU_MAX_CONCURRENT_JOBS must be between 1 and 8"
            )
        evidence_value = os.environ.get("QAITU_EVIDENCE_ROOT", "").strip()
        evidence_root = (
            Path(evidence_value).resolve()
            if evidence_value
            else (PROJECT_ROOT / "artifacts" / "evidence").resolve()
        )
        return cls(
            mode=raw_mode,
            bundle_root=bundle_root,
            allow_development_model=allow_development,
            evidence_root=evidence_root,
            expected_model_sha256=expected_model_sha256,
            prefer_cuda=prefer_cuda,
            max_concurrent_jobs=max_concurrent_jobs,
            vlm_mode=vlm_mode,
        )


@dataclass(frozen=True)
class RuntimeResult:
    verdict: Literal["no_violation", "violation"]
    violations: list[dict[str, object]]
    analysis_details: list[dict[str, object]]
    summary: str
    runtime: dict[str, object]


class InferenceRuntime:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        predictor: Predictor | None = None,
        config: PipelineConfig | None = None,
        semantic_adapter: SemanticAdapter | None = None,
        sequence_classifier: object | None = None,
    ) -> None:
        self.settings = settings
        sequence_path = os.environ.get("QAITU_SEQUENCE_ADAPTER", "").strip()
        if (sequence_path or sequence_classifier) and (settings.vlm_mode != "off" or settings.mode != "onnx"):
            raise RuntimeConfigurationError("Sequence adapter requires ONNX and cannot share a GPU with semantic Qwen")
        if semantic_adapter is not None and settings.vlm_mode != "qwen":
            raise RuntimeConfigurationError("Semantic adapter requires qwen mode")
        self._analysis_slots = threading.BoundedSemaphore(settings.max_concurrent_jobs)
        self.model_sha256: str | None = None
        self.release_status = "injected"
        if predictor is not None or config is not None:
            if predictor is None or config is None:
                raise RuntimeConfigurationError(
                    "Predictor and pipeline config must be injected together"
                )
            self.predictor = predictor
            self.config = config
            if settings.mode == "mock":
                self.release_status = "test_only"
        elif settings.mode == "mock":
            self.predictor = MockPredictor([0.1, 0.2, 0.8, 0.9, 0.3])
            self.config = PipelineConfig(
                window_sec=8.0,
                stride_sec=2.0,
                num_frames=4,
                threshold=0.5,
                batch_size=8,
            )
            self.release_status = "test_only"
        else:
            if settings.bundle_root is None:
                raise RuntimeConfigurationError("ONNX mode requires a bundle")
            bundle = load_bundle_contract(
                settings.bundle_root,
                allow_development_model=settings.allow_development_model,
            )
            self.model_sha256 = str(bundle.manifest["sha256"]).lower()
            if (
                settings.expected_model_sha256 is not None
                and settings.expected_model_sha256 != self.model_sha256
            ):
                raise RuntimeConfigurationError(
                    "Loaded model checksum does not match QAITU_EXPECTED_MODEL_SHA256"
                )
            self.config = bundle.pipeline_config
            self.release_status = bundle.release_status
            self.predictor = OnnxPredictor(
                bundle.model_path,
                bundle.preprocessing,
                bundle.manifest_path,
                allow_development_model=settings.allow_development_model,
                prefer_cuda=settings.prefer_cuda,
            )
        self.semantic = (
            semantic_adapter or SemanticAdapter.from_environment()
            if settings.vlm_mode == "qwen" else None
        )
        self.sequence = sequence_classifier
        if sequence_path and self.sequence is None:
            from .inference.qwen_sequence import SequenceClassifier
            self.sequence = SequenceClassifier(
                os.environ.get("QAITU_VLM_MODEL_PATH", "models/pretrained/qwen3-vl-2b-instruct"),
                sequence_path,
                allow_experimental=_environment_bool("QAITU_ALLOW_EXPERIMENTAL_ADAPTER", default=False),
            )
        LOGGER.info(
            "inference_runtime_ready mode=%s predictor_class=%s model_version=%s "
            "model_sha256=%s providers=%s release_status=%s",
            self.settings.mode,
            type(self.predictor).__name__,
            self.predictor.model_version,
            self.model_sha256 or "none",
            ",".join(self.predictor.runtime_providers),
            self.release_status,
        )

    @property
    def mode_label(self) -> str:
        if self.settings.mode == "mock":
            return "Demo predictor"
        suffix = f" + Qwen · {self.semantic.model_version}" if self.semantic else ""
        if self.sequence is not None:
            suffix += " + Qwen LoRA · experimental whole-video review"
        return f"ONNX · {self.predictor.model_version} · {self.predictor.device}{suffix}"

    @property
    def model_diagnostics(self) -> dict[str, object]:
        return {
            "inference_mode": self.settings.mode,
            "predictor_class": type(self.predictor).__name__,
            "model_version": self.predictor.model_version,
            "model_sha256": self.model_sha256,
            "runtime_providers": list(self.predictor.runtime_providers),
            "release_status": self.release_status,
            "semantic_mode": self.settings.vlm_mode,
            "semantic_model_version": self.semantic.model_version if self.semantic else None,
            "semantic_model_sha256": self.semantic.model_checksum if self.semantic else None,
            "sequence_mode": "experimental_advisory" if self.sequence is not None else "off",
            "sequence_adapter_sha256": self.sequence.manifest["adapter_sha256"] if self.sequence is not None else None,
            "sequence_quality_gate_passed": bool(self.sequence.manifest.get("quality_gate_passed")) if self.sequence is not None else None,
        }

    def analyze(
        self,
        video_path: str | Path,
        *,
        job_id: str,
        progress: ProgressCallback | None = None,
    ) -> RuntimeResult:
        started = time.perf_counter()
        sequence_result = None
        try:
            with self._analysis_slots:
                self.settings.evidence_root.mkdir(parents=True, exist_ok=True)
                if self.semantic is None:
                    result = analyze_video(
                        video_path, self.predictor, self.config,
                        evidence_root=self.settings.evidence_root, run_id=job_id,
                        progress=progress,
                    )
                else:
                    if progress is not None:
                        progress("model_inference")
                    result = self.semantic.analyze(
                        video_path, self.predictor, self.config,
                        evidence_root=self.settings.evidence_root, run_id=job_id,
                    )
                if self.sequence is not None:
                    if progress:
                        progress("model_inference")
                    try:
                        sequence_result = self.sequence.predict(video_path)
                    except Exception as exc:
                        raise InferenceError("Trained Qwen adapter analysis failed") from exc
                    result["runtime"]["total_processing_sec"] = time.perf_counter() - started
            if progress is not None:
                progress("summary")
            violations, analysis_details = self._map_violations(result)
            advisory = ""
            if sequence_result is not None:
                if sequence_result.get("decision") not in {"possible_violation", "normal_activity"}:
                    raise InferenceError("Trained Qwen adapter returned an invalid decision")
                label = ("отметил признаки возможного нарушения" if sequence_result["decision"] == "possible_violation"
                         else "оценил запись как обычную активность")
                advisory = (f" Экспериментальный обученный Qwen по всему ролику {label}. "
                            "Его проверка качества не пройдена; этот ответ не подтверждает конкретный эпизод или правило.")
                for violation, detail in zip(violations, analysis_details, strict=True):
                    violation["description"] += advisory
                    detail["reasoning"] += advisory
            verdict: Literal["no_violation", "violation"] = (
                "violation" if violations else "no_violation"
            )
            return RuntimeResult(
                verdict=verdict,
                violations=violations,
                analysis_details=analysis_details,
                summary=_build_summary(verdict, violations) + advisory,
                runtime={
                    **dict(result.get("runtime", {})),
                    "mode": self.settings.mode,
                    "mode_label": self.mode_label,
                    **self.model_diagnostics,
                    "diagnostics": self._internal_diagnostics(result),
                    "sequence_review": sequence_result,
                },
            )
        finally:
            self._remove_job_evidence(job_id)

    @staticmethod
    def _internal_diagnostics(result: dict[str, object]) -> dict[str, object]:
        raw_scores = result.get("window_scores")
        window_scores = list(raw_scores) if isinstance(raw_scores, list) else []
        raw_events = result.get("events")
        events: list[dict[str, object]] = []
        if isinstance(raw_events, list):
            for event in raw_events:
                if not isinstance(event, dict):
                    continue
                events.append(
                    {
                        "start_sec": event.get("start_sec"),
                        "end_sec": event.get("end_sec"),
                        "peak_time_sec": event.get("peak_time_sec"),
                        "peak_score": event.get("score"),
                        "supporting_windows": event.get("supporting_windows"),
                        "supporting_window_indices": event.get(
                            "supporting_window_indices", []
                        ),
                    }
                )
        return {"window_scores": window_scores, "events": events,
                "semantic": result.get("diagnostics", {})}

    def _map_violations(
        self, result: dict[str, object]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        raw_events = result.get("events")
        if not isinstance(raw_events, list):
            raise InferenceError("Inference result does not contain an event list")
        violations: list[dict[str, object]] = []
        analysis_details: list[dict[str, object]] = []
        for event_index, event in enumerate(raw_events, start=1):
            if not isinstance(event, dict):
                raise InferenceError("Inference event is invalid")
            start_time = _finite_number(event.get("start_sec"), "event start")
            end_time = _finite_number(event.get("end_sec"), "event end")
            peak_time = _finite_number(event.get("peak_time_sec"), "event timecode")
            _finite_number(event.get("score"), "event score")
            supporting_windows = event.get("supporting_windows")
            if not isinstance(supporting_windows, int) or supporting_windows < 1:
                raise InferenceError("Inference event support is invalid")
            evidence_items = event.get("evidence")
            if not isinstance(evidence_items, list) or not 1 <= len(evidence_items) <= 3:
                raise InferenceError("Inference event must contain 1-3 evidence frames")
            frames = [self._map_evidence(item) for item in evidence_items]
            if self.semantic is not None:
                title, description, detail = map_semantic_event(
                    event.get("semantic_verification"), event_index
                )
                violations.append(dict(type=title, timecode_sec=round(peak_time, 3),
                                       description=description, evidence_frames=frames))
                analysis_details.append(detail)
                continue
            violations.append(
                {
                    "type": "Визуально нетипичное действие",
                    "timecode_sec": round(peak_time, 3),
                    "description": (
                        "Автоматический локальный анализ обнаружил в этом "
                        "интервале устойчивый модельный сигнал выше порога. "
                        "Это указывает на признаки возможного нарушения, но "
                        "не является юридическим заключением."
                    ),
                    "evidence_frames": frames,
                }
            )
            analysis_details.append(
                {
                    "event_index": event_index,
                    "thesis": (
                        "Вероятно, в отмеченном интервале присутствует "
                        "визуально нетипичное действие."
                    ),
                    "observations": [
                        (
                            f"Модельный сигнал выше порога сохранялся в "
                            f"{supporting_windows} временных окнах с "
                            f"{start_time:.1f} до {end_time:.1f} сек."
                        ),
                        (
                            f"Пиковый момент отмечен на {peak_time:.1f} сек.; "
                            f"для эпизода сформировано JPEG-кадров: {len(frames)}."
                        ),
                    ],
                    "reasoning": (
                        "Сигнал поддерживается последовательностью временных "
                        "окон, а не одиночным кадром. Совпадение пикового "
                        "таймкода с сохранёнными кадрами подкрепляет вывод о "
                        "локализованном подозрительном эпизоде."
                    ),
                    "risk": (
                        "Нетипичная активность в кассовой зоне может повышать "
                        "риск потери контроля над операцией, нарушения "
                        "прослеживаемости действий или затруднения последующей "
                        "сверки. Наличие риска не доказывает фактический ущерб."
                    ),
                    "rule": {
                        "status": "not_established",
                        "label": "Нормативное основание не установлено",
                        "document_name": None,
                        "section": None,
                        "source_url": None,
                        "applicability": (
                            "Конкретное правило будет показано только после "
                            "совпадения с проверенным локальным каталогом."
                        ),
                    },
                }
            )
        return violations, analysis_details

    def _map_evidence(self, item: object) -> dict[str, object]:
        if not isinstance(item, dict):
            raise InferenceError("Evidence item is invalid")
        timestamp = _finite_number(item.get("timecode_sec"), "evidence timestamp")
        relative_value = item.get("jpeg_path")
        if not isinstance(relative_value, str) or not relative_value:
            raise InferenceError("Evidence path is missing")
        evidence_root = self.settings.evidence_root.resolve()
        candidate = (evidence_root / relative_value).resolve()
        if not candidate.is_relative_to(evidence_root) or not candidate.is_file():
            raise InferenceError("Evidence file is missing or outside the evidence root")
        payload = candidate.read_bytes()
        if len(payload) < 4 or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise InferenceError("Evidence file is not a valid JPEG")
        return {
            "timestamp_sec": round(timestamp, 3),
            "image_base64": base64.b64encode(payload).decode("ascii"),
        }

    def _remove_job_evidence(self, job_id: str) -> None:
        evidence_root = self.settings.evidence_root.resolve()
        display = re.sub(r"[^A-Za-z0-9_-]", "_", job_id)[:60] or "run"
        digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:12]
        candidate = (evidence_root / f"{display}_{digest}").resolve()
        if candidate.is_relative_to(evidence_root) and candidate.is_dir():
            shutil.rmtree(candidate)


def _finite_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise InferenceError(f"{field} is missing")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise InferenceError(f"{field} is invalid")
    return number


def _build_summary(
    verdict: Literal["no_violation", "violation"],
    violations: list[dict[str, object]],
) -> str:
    if verdict == "no_violation":
        return (
            "Автоматический анализ завершён: признаки возможного нарушения в "
            "этой записи не обнаружены. Вывод относится только к видимым "
            "действиям в загруженном видео."
        )
    timecodes = [float(item["timecode_sec"]) for item in violations]
    earliest = min(timecodes)
    return (
        f"Автоматический анализ обнаружил эпизодов с признаками возможного "
        f"нарушения: {len(violations)}. "
        f"Первый отмеченный момент — {earliest:.1f} сек. "
        "Для каждого эпизода сформированы таймкод, логическое обоснование и "
        "JPEG-доказательства."
    )


@lru_cache(maxsize=1)
def get_runtime() -> InferenceRuntime:
    if os.environ.get("QAITU_WORKER_URL") or os.environ.get("QAITU_WORKER_KEY_FILE"):
        from .remote_runtime import build_remote_runtime
        return build_remote_runtime()
    return InferenceRuntime(RuntimeSettings.from_environment())
