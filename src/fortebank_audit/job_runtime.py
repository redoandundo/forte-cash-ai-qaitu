from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .inference.benchmark import build_predictor, load_pipeline_config
from .inference.contracts import InferenceError, PipelineConfig
from .inference.pipeline import ProgressCallback, analyze_video
from .inference.predictor import MockPredictor, Predictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RuntimeConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeSettings:
    mode: Literal["mock", "onnx"]
    bundle_root: Path | None
    allow_development_model: bool
    evidence_root: Path

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        raw_mode = os.environ.get("QAITU_INFERENCE_MODE", "mock").strip().lower()
        if raw_mode not in {"mock", "onnx"}:
            raise RuntimeConfigurationError("QAITU_INFERENCE_MODE must be mock or onnx")
        bundle_value = os.environ.get("QAITU_MODEL_BUNDLE", "").strip()
        bundle_root = Path(bundle_value).resolve() if bundle_value else None
        if raw_mode == "onnx" and bundle_root is None:
            raise RuntimeConfigurationError("ONNX mode requires QAITU_MODEL_BUNDLE")
        allow_development = os.environ.get(
            "QAITU_ALLOW_DEVELOPMENT_MODEL", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
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
    ) -> None:
        self.settings = settings
        if predictor is not None or config is not None:
            if predictor is None or config is None:
                raise RuntimeConfigurationError(
                    "Predictor and pipeline config must be injected together"
                )
            self.predictor = predictor
            self.config = config
            return

        if settings.mode == "mock":
            self.predictor = MockPredictor([0.1, 0.2, 0.8, 0.9, 0.3])
            self.config = PipelineConfig(
                window_sec=8.0,
                stride_sec=2.0,
                num_frames=4,
                threshold=0.5,
                batch_size=8,
            )
            return

        if settings.bundle_root is None:
            raise RuntimeConfigurationError("ONNX mode requires a bundle")
        _, self.config, _ = load_pipeline_config(
            settings.bundle_root,
            allow_development_model=settings.allow_development_model,
        )
        self.predictor = build_predictor(
            "onnx",
            settings.bundle_root,
            allow_development_model=settings.allow_development_model,
        )

    @property
    def mode_label(self) -> str:
        if self.settings.mode == "mock":
            return "Demo predictor"
        return self.predictor.model_version

    def analyze(
        self,
        video_path: str | Path,
        *,
        job_id: str,
        progress: ProgressCallback | None = None,
    ) -> RuntimeResult:
        self.settings.evidence_root.mkdir(parents=True, exist_ok=True)
        result = analyze_video(
            video_path,
            self.predictor,
            self.config,
            evidence_root=self.settings.evidence_root,
            run_id=job_id,
            progress=progress,
        )
        if progress is not None:
            progress("summary")
        try:
            violations, analysis_details = self._map_violations(result)
        finally:
            self._remove_job_evidence(job_id)
        verdict: Literal["no_violation", "violation"] = (
            "violation" if violations else "no_violation"
        )
        return RuntimeResult(
            verdict=verdict,
            violations=violations,
            analysis_details=analysis_details,
            summary=_build_summary(verdict, violations),
            runtime={
                **dict(result.get("runtime", {})),
                "mode": self.settings.mode,
                "mode_label": self.mode_label,
            },
        )

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
    return InferenceRuntime(RuntimeSettings.from_environment())
