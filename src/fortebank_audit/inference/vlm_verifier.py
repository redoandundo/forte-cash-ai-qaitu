from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


ALLOWED_STATUSES = {"visible_support", "no_visible_support", "insufficient_evidence"}


class VerificationOutputError(RuntimeError):
    """Raised when a verifier returns output that violates the JSON contract."""


@dataclass(frozen=True)
class BackendOutput:
    text: str
    load_sec: float
    inference_sec: float
    peak_vram_gib: float | None


class VisionLanguageBackend(Protocol):
    @property
    def model_version(self) -> str: ...

    def generate(
        self,
        images: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> BackendOutput: ...


@dataclass(frozen=True)
class VerificationResult:
    status: str
    confidence: float
    summary: str
    visible_actions: tuple[str, ...]
    matched_signal_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    model_version: str
    load_sec: float
    inference_sec: float
    peak_vram_gib: float | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("visible_actions", "matched_signal_ids", "limitations"):
            payload[key] = list(payload[key])
        return payload


def load_signal_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("signals_version"), str):
        raise ValueError("Signal policy requires signals_version")
    signals = payload.get("signals")
    if not isinstance(signals, list) or not signals:
        raise ValueError("Signal policy requires a non-empty signals list")
    identifiers: set[str] = set()
    for item in signals:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Every signal requires a string id")
        if not isinstance(item.get("description"), str):
            raise ValueError("Every signal requires a string description")
        if item["id"] in identifiers:
            raise ValueError(f"Duplicate signal id: {item['id']}")
        identifiers.add(item["id"])
    return payload


def build_prompt(
    *,
    score: float,
    peak_time_sec: float,
    signal_policy: dict[str, Any],
) -> str:
    signals = "\n".join(
        f"- {item['id']}: {item['description']}" for item in signal_policy["signals"]
    )
    return f"""You are a local evidence verifier for cashier video review.
The images are ordered before, during, and after a candidate moment.
Candidate time: {peak_time_sec:.3f} seconds. First-layer score: {score:.6f}.

Evaluate only directly visible facts. Do not infer identity, intent, guilt, fraud, or a legal
violation. If the images are ambiguous, use insufficient_evidence. A first-layer score is only
a routing signal and must not be treated as visual proof.

Observable signal catalogue ({signal_policy['signals_version']}):
{signals}

Return one JSON object and no markdown with exactly these fields:
{{
  "status": "visible_support|no_visible_support|insufficient_evidence",
  "confidence": 0.0,
  "summary": "short Russian description of visible evidence",
  "visible_actions": ["directly visible action"],
  "matched_signal_ids": ["id from the catalogue"],
  "limitations": ["blur, occlusion, missing temporal context, or other limitation"]
}}
"""


def parse_verification_output(
    raw_text: str,
    *,
    allowed_signal_ids: set[str],
    model_version: str,
    backend_output: BackendOutput,
) -> VerificationResult:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end <= start:
        raise VerificationOutputError("Verifier did not return a JSON object")
    try:
        payload = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VerificationOutputError("Verifier returned invalid JSON") from exc
    status = payload.get("status")
    if status not in ALLOWED_STATUSES:
        raise VerificationOutputError(f"Unsupported verification status: {status}")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        raise VerificationOutputError("Verifier confidence must be finite")
    if not 0 <= float(confidence) <= 1:
        raise VerificationOutputError("Verifier confidence must be in [0, 1]")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise VerificationOutputError("Verifier summary must be non-empty")

    list_fields: dict[str, tuple[str, ...]] = {}
    for field in ("visible_actions", "matched_signal_ids", "limitations"):
        value = payload.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise VerificationOutputError(f"Verifier field {field} must be a string list")
        list_fields[field] = tuple(item.strip() for item in value if item.strip())
    unknown = set(list_fields["matched_signal_ids"]) - allowed_signal_ids
    if unknown:
        raise VerificationOutputError(f"Verifier returned unknown signal ids: {sorted(unknown)}")
    return VerificationResult(
        status=status,
        confidence=float(confidence),
        summary=summary.strip(),
        visible_actions=list_fields["visible_actions"],
        matched_signal_ids=list_fields["matched_signal_ids"],
        limitations=list_fields["limitations"],
        model_version=model_version,
        load_sec=backend_output.load_sec,
        inference_sec=backend_output.inference_sec,
        peak_vram_gib=backend_output.peak_vram_gib,
    )


class QwenLocalBackend:
    """Lazy, local-only Qwen3-VL backend sized for an 8 GB GPU."""

    def __init__(self, model_path: str | Path, *, max_image_side: int = 448) -> None:
        self.model_path = Path(model_path)
        self.max_image_side = max_image_side
        self._model: Any = None
        self._processor: Any = None
        self._load_sec = 0.0

    @property
    def model_version(self) -> str:
        return "qwen3-vl-2b-instruct-local"

    def _load(self) -> None:
        if self._model is not None:
            return
        if not (self.model_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Qwen weights not found under {self.model_path}")
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        started = time.perf_counter()
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self._load_sec = time.perf_counter() - started

    def generate(
        self,
        images: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> BackendOutput:
        self._load()
        import torch
        from PIL import Image

        loaded = []
        for path in images:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((self.max_image_side, self.max_image_side))
                loaded.append(image.copy())
        content = [{"type": "image", "image": image} for image in loaded]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        inference_sec = time.perf_counter() - started
        input_tokens = inputs["input_ids"].shape[-1]
        text = self._processor.decode(
            generated[0, input_tokens:],
            skip_special_tokens=True,
        )
        peak_vram = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        )
        return BackendOutput(text, self._load_sec, inference_sec, peak_vram)


class EventVerifier:
    def __init__(
        self,
        backend: VisionLanguageBackend,
        signal_policy: dict[str, Any],
        *,
        max_new_tokens: int = 192,
    ) -> None:
        self.backend = backend
        self.signal_policy = signal_policy
        self.max_new_tokens = max_new_tokens
        self.allowed_signal_ids = {item["id"] for item in signal_policy["signals"]}

    def verify(
        self,
        evidence_paths: Sequence[str | Path],
        *,
        score: float,
        peak_time_sec: float,
    ) -> VerificationResult:
        paths = tuple(Path(path) for path in evidence_paths)
        if not 1 <= len(paths) <= 9:
            raise ValueError("Verifier requires between one and nine evidence images")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Evidence image not found: {missing[0]}")
        prompt = build_prompt(
            score=score,
            peak_time_sec=peak_time_sec,
            signal_policy=self.signal_policy,
        )
        output = self.backend.generate(paths, prompt, max_new_tokens=self.max_new_tokens)
        return parse_verification_output(
            output.text,
            allowed_signal_ids=self.allowed_signal_ids,
            model_version=self.backend.model_version,
            backend_output=output,
        )
