from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


LOGGER = logging.getLogger(__name__)
DECISIONS = {"normal_activity", "possible_violation", "insufficient_evidence"}
CATEGORIES = {
    "ordinary_entry_or_presence", "cash_or_document_handling", "hand_or_object_transfer",
    "unattended_workspace", "prolonged_inactivity_or_distraction", "unclear_activity",
}
CONFIDENCES = {"low", "medium", "high"}
FRAME_ROLES = {"before", "start", "peak", "end", "after"}
TOP_LEVEL_FIELDS = {
    "decision", "category", "confidence", "thesis", "observations", "reasoning", "risk",
    "matched_signal_ids", "rule_candidate", "limitations",
}
RULE_FIELDS = {"rule_id", "document_name", "section", "applicability_reason", "certainty"}


class VerificationOutputError(RuntimeError):
    """The VLM answered, but its answer violates the semantic contract."""


class VerificationRuntimeError(RuntimeError):
    """The mandatory local VLM could not complete verification."""


@dataclass(frozen=True)
class BackendOutput:
    text: str
    load_sec: float
    inference_sec: float
    peak_vram_gib: float | None


@dataclass(frozen=True)
class TemporalFrame:
    role: str
    timestamp_sec: float
    jpeg_path: Path


@dataclass(frozen=True)
class EventContext:
    start_sec: float
    end_sec: float
    peak_time_sec: float
    ranking_score: float
    supporting_windows: int
    activity_duration_sec: float
    frames: tuple[TemporalFrame, ...]


@dataclass(frozen=True)
class Observation:
    frame_role: str
    timestamp_sec: float
    visible_fact: str


@dataclass(frozen=True)
class RuleCandidate:
    rule_id: str
    document_name: str
    section: str
    applicability_reason: str
    certainty: str


@dataclass(frozen=True)
class VerificationResult:
    decision: str
    category: str
    confidence: str
    thesis: str
    observations: tuple[Observation, ...]
    reasoning: str
    risk: str
    matched_signal_ids: tuple[str, ...]
    rule_candidate: RuleCandidate | None
    limitations: tuple[str, ...]
    model_version: str
    model_checksum: str
    load_sec: float
    inference_sec: float
    peak_vram_gib: float | None
    retry_count: int

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observations"] = [asdict(item) for item in self.observations]
        value["matched_signal_ids"] = list(self.matched_signal_ids)
        value["limitations"] = list(self.limitations)
        value["normative_basis"] = (
            "Нормативное основание автоматически не установлено"
            if self.rule_candidate is None else "Сопоставлено только как кандидат для проверки человеком"
        )
        return value


class VisionLanguageBackend(Protocol):
    model_version: str
    model_checksum: str

    def generate(self, images: Sequence[Path], prompt: str, *, max_new_tokens: int) -> BackendOutput: ...


def load_signal_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("signals_version"), str):
        raise ValueError("Signal policy requires signals_version")
    signals = payload.get("signals")
    if not isinstance(signals, list) or not signals:
        raise ValueError("Signal policy requires a non-empty signals list")
    identifiers: set[str] = set()
    for item in signals:
        if set(item) != {"id", "description"} or not all(isinstance(item[k], str) for k in item):
            raise ValueError("Every signal requires only string id and description")
        if item["id"] in identifiers:
            raise ValueError(f"Duplicate signal id: {item['id']}")
        identifiers.add(item["id"])
    return payload


def build_prompt(
    context: EventContext, signal_policy: dict[str, Any], *, repair_error: str | None = None,
    rule_catalog: dict[str, dict[str, Any]] | None = None,
) -> str:
    frames = "\n".join(
        f"- image {index + 1}: role={frame.role}, timestamp_sec={frame.timestamp_sec:.6f}"
        for index, frame in enumerate(context.frames)
    )
    signals = "\n".join(
        f"- {item['id']}: {item['description']}" for item in signal_policy["signals"]
    )
    repair_text = (
        f"\nREPAIR: The previous answer was invalid: {repair_error}. Return exactly the JSON object, with no markdown, "
        "comments, prefix, suffix, or additional fields. If view limitations prevent confirming a "
        "visible risk sequence, use insufficient_evidence with explicit limitations, not "
        "normal_activity. Copy every observation role/timestamp pair exactly from the frame list.\n"
        if repair_error else ""
    )
    rule_instruction = "rule_candidate must be null because no verified local rule catalogue is configured."
    if rule_catalog:
        rule_instruction = (
            "The following reviewed catalogue permits ONLY a candidate for human review, never a legal verdict. "
            "Suggest a rule only when every listed condition is supported by the supplied temporal observations. "
            "Missing or off-screen conditions mean rule_candidate=null. Never infer customer absence from a crop, "
            "transaction completion from a still frame, or money ownership from appearance. "
            "Copy exact rule_id/document_name/section; certainty must be candidate; explain applicability. "
            "A candidate object has exactly rule_id, document_name, section, applicability_reason, certainty.\n"
            + json.dumps(rule_catalog, ensure_ascii=False)
        )
    return f"""You are the local semantic verification layer for cashier video review.
Classify ONE ONNX candidate event using the ordered temporal images. The images are context for one
event, never separate events. The ONNX ranking score is not a legal-violation probability.

Event metadata: start_sec={context.start_sec:.6f}, end_sec={context.end_sec:.6f},
peak_time_sec={context.peak_time_sec:.6f}, ranking_score={context.ranking_score:.6f},
supporting_windows={context.supporting_windows}, activity_duration_sec={context.activity_duration_sec:.6f}.
Frames in chronological order:
{frames}
Every observation must copy one exact role/timestamp_sec pair from this list; never calculate,
round, swap, or invent a timestamp.

Decision rules:
- An ordinary entry or presence without additional suspicious action is normal_activity.
- A second person or movement alone is not a violation.
- Cash handling in a cashier area alone does not prove a violation.
- possible_violation requires a logically connected observable sequence across the supplied context.
- A sequence may be possible_violation when the images directly show an object changing hands across
  the workspace, or visible transaction items becoming unattended as the worker leaves. Require the
  transition in at least two frames; do not infer it from mere presence or a single image.
- Stable ordinary service, entry, presence, or handling with no such visible transition is normal_activity.
- Use normal_activity only with high confidence when the complete ordered context is clear. If the
  activity appears ordinary but confidence would be low or medium, return insufficient_evidence.
- High-confidence normal_activity requires distinct before and after context, no material view
  limitation, and an empty limitations list.
- Poor view or missing temporal context is insufficient_evidence.
- If view limitations prevent confirming a matched visible risk sequence, use insufficient_evidence
  with explicit limitations, not normal_activity.
- Prolonged inactivity/distraction requires multiple temporal observations and supporting windows.
- State only visible facts. Never assert identity, intent, guilt, theft, crime, or a proven violation.
- {rule_instruction}
- matched_signal_ids must contain only signals directly supported by observations. Never copy the
  whole catalogue. Do not return view_quality_limitation with high-confidence normal_activity.

Observable signal catalogue ({signal_policy['signals_version']}):
{signals}

Return one strict JSON object, no Markdown, with exactly:
{{
  "decision": "normal_activity | possible_violation | insufficient_evidence",
  "category": "ordinary_entry_or_presence | cash_or_document_handling | hand_or_object_transfer | unattended_workspace | prolonged_inactivity_or_distraction | unclear_activity",
  "confidence": "low | medium | high",
  "thesis": "Краткий тезис о том, что происходит",
  "observations": [{{"frame_role": "before | start | peak | end | after", "timestamp_sec": 0.0, "visible_fact": "Только непосредственно видимый факт"}}],
  "reasoning": "Почему последовательность наблюдаемых действий поддерживает решение",
  "risk": "Почему такая последовательность может создавать риск или почему риск не установлен",
  "matched_signal_ids": [],
  "rule_candidate": null,
  "limitations": []
}}
Use only supplied frame roles/timestamps and catalogue signal IDs.{repair_text}"""


def _string_list(payload: dict[str, Any], field: str) -> tuple[str, ...]:
    value = payload[field]
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise VerificationOutputError(f"{field} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def parse_verification_output(
    raw_text: str, *, context: EventContext, allowed_signal_ids: set[str],
    rule_catalog: dict[str, dict[str, str]] | None, model_version: str,
    model_checksum: str, backend_output: BackendOutput, retry_count: int,
) -> VerificationResult:
    if not raw_text or raw_text != raw_text.strip():
        raise VerificationOutputError("Verifier must return one unwrapped JSON object")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise VerificationOutputError("Verifier returned invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_FIELDS:
        raise VerificationOutputError("Verifier returned missing or unknown fields")
    if not isinstance(payload["decision"], str) or payload["decision"] not in DECISIONS:
        raise VerificationOutputError("Unknown decision enum")
    if not isinstance(payload["category"], str) or payload["category"] not in CATEGORIES:
        raise VerificationOutputError("Unknown category enum")
    if not isinstance(payload["confidence"], str) or payload["confidence"] not in CONFIDENCES:
        raise VerificationOutputError("Unknown confidence enum")
    for field in ("thesis", "reasoning", "risk"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise VerificationOutputError(f"{field} must be non-empty")

    supplied = {frame.role: frame.timestamp_sec for frame in context.frames}
    raw_observations = payload["observations"]
    if not isinstance(raw_observations, list) or not raw_observations:
        raise VerificationOutputError("observations must be non-empty")
    observations: list[Observation] = []
    for item in raw_observations:
        if not isinstance(item, dict) or set(item) != {"frame_role", "timestamp_sec", "visible_fact"}:
            raise VerificationOutputError("Observation has missing or unknown fields")
        role, timestamp, fact = item["frame_role"], item["timestamp_sec"], item["visible_fact"]
        if not isinstance(role, str) or role not in FRAME_ROLES:
            raise VerificationOutputError("Unknown frame role")
        if type(timestamp) not in (int, float) or not math.isfinite(float(timestamp)):
            raise VerificationOutputError("Observation timestamp must be finite")
        canonical_timestamp = supplied.get(role)
        if canonical_timestamp is None or not math.isclose(
            float(timestamp), canonical_timestamp, abs_tol=0.05
        ):
            raise VerificationOutputError(
                f"Observation timestamp does not match supplied frame role={role} "
                f"received={float(timestamp):.6f} expected={canonical_timestamp}"
            )
        if not isinstance(fact, str) or not fact.strip():
            raise VerificationOutputError("visible_fact must be non-empty")
        observations.append(Observation(role, canonical_timestamp, fact.strip()))

    signals = _string_list(payload, "matched_signal_ids")
    unknown = set(signals) - allowed_signal_ids
    if unknown:
        raise VerificationOutputError(f"Verifier returned unknown signal ids: {sorted(unknown)}")
    limitations = _string_list(payload, "limitations")

    rule: RuleCandidate | None = None
    raw_rule = payload["rule_candidate"]
    if raw_rule is not None:
        if not rule_catalog:
            raise VerificationOutputError("rule_candidate is forbidden without a verified catalogue")
        if not isinstance(raw_rule, dict) or set(raw_rule) != RULE_FIELDS:
            raise VerificationOutputError("Invalid rule_candidate fields")
        rule_id = raw_rule.get("rule_id")
        if not isinstance(rule_id, str) or rule_id not in rule_catalog or raw_rule.get("certainty") != "candidate":
            raise VerificationOutputError("Unknown rule candidate")
        if not all(isinstance(raw_rule[field], str) and raw_rule[field].strip() for field in RULE_FIELDS):
            raise VerificationOutputError("Rule candidate strings must be non-empty")
        canonical = rule_catalog[rule_id]
        if raw_rule["document_name"] != canonical["document_name"] or raw_rule["section"] != canonical["section"]:
            raise VerificationOutputError("Rule metadata does not match catalogue")
        rule = RuleCandidate(**raw_rule)

    if payload["category"] == "prolonged_inactivity_or_distraction":
        distinct_times = {round(item.timestamp_sec, 6) for item in observations}
        if len(distinct_times) < 2 or context.supporting_windows < 2:
            raise VerificationOutputError("inactivity requires multiple temporal observations/windows")
    if payload["decision"] == "possible_violation" and len({o.timestamp_sec for o in observations}) < 2:
        raise VerificationOutputError("possible_violation requires a temporal sequence")
    if payload["decision"] == "normal_activity" and rule is not None:
        raise VerificationOutputError("normal activity cannot have a rule candidate")
    if payload["decision"] == "insufficient_evidence" and not limitations:
        raise VerificationOutputError("insufficient_evidence requires limitations")

    return VerificationResult(
        decision=payload["decision"], category=payload["category"], confidence=payload["confidence"],
        thesis=payload["thesis"].strip(), observations=tuple(observations),
        reasoning=payload["reasoning"].strip(), risk=payload["risk"].strip(),
        matched_signal_ids=signals, rule_candidate=rule, limitations=limitations,
        model_version=model_version, model_checksum=model_checksum,
        load_sec=backend_output.load_sec, inference_sec=backend_output.inference_sec,
        peak_vram_gib=backend_output.peak_vram_gib, retry_count=retry_count,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QwenLocalBackend:
    """Manifest-pinned, local-only Qwen3-VL backend for one CUDA process."""

    def __init__(self, model_path: str | Path, manifest_path: str | Path, *, max_concurrent_jobs: int = 1) -> None:
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.model_version = str(self.manifest["model_id"])
        self.quantization = self.manifest.get("quantization", "none")
        if self.quantization not in {"none", "nf4"}:
            raise VerificationRuntimeError("Unsupported Qwen quantization")
        if self.quantization == "nf4":
            self.model_version += "-nf4-experimental"
        self.max_image_side = int(self.manifest["max_image_side"])
        self.max_generation_sec = float(self.manifest.get("max_generation_sec", 90.0))
        if not math.isfinite(self.max_generation_sec) or self.max_generation_sec <= 0:
            raise VerificationRuntimeError("max_generation_sec must be finite and positive")
        if max_concurrent_jobs <= 0:
            raise ValueError("max_concurrent_jobs must be positive")
        expected_files = self.manifest.get("weights_files")
        if expected_files != ["model.safetensors"]:
            raise VerificationRuntimeError("Unsupported Qwen snapshot layout")
        actual_weights = sorted(path.name for path in self.model_path.glob("*.safetensors"))
        if actual_weights != expected_files:
            raise VerificationRuntimeError("Qwen snapshot does not match manifest layout")
        weights = self.model_path / expected_files[0]
        self.model_checksum = _sha256(weights)
        if self.model_checksum != self.manifest["weights_sha256"]:
            raise VerificationRuntimeError("Qwen weights checksum mismatch")
        self._model: Any = None
        self._processor: Any = None
        self._load_sec = 0.0
        self._load_lock = threading.Lock()
        self._jobs = threading.BoundedSemaphore(max_concurrent_jobs)
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise VerificationRuntimeError("Qwen mode requires CUDA")
        try:
            import transformers
            from packaging.version import Version
        except ImportError as exc:
            raise VerificationRuntimeError("Missing Transformers runtime") from exc
        version = Version(transformers.__version__)
        if not (Version(self.manifest["transformers_min_version"]) <= version <
                Version(self.manifest["transformers_max_exclusive"])):
            raise VerificationRuntimeError("Incompatible Transformers version")

    def preload(self) -> None:
        self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            started = time.perf_counter()
            try:
                options = {}
                if self.quantization == "nf4":
                    from transformers import BitsAndBytesConfig
                    options["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
                model = AutoModelForMultimodalLM.from_pretrained(
                    self.model_path, local_files_only=True, dtype=torch.float16,
                    device_map="cuda:0", low_cpu_mem_usage=True, **options,
                ).eval()
            except torch.cuda.OutOfMemoryError as exc:
                raise VerificationRuntimeError("CUDA out of memory while loading Qwen") from exc
            except Exception as exc:
                raise VerificationRuntimeError("Could not load local Qwen snapshot") from exc
            # Publish only a complete pair. Failed processor loading must remain retryable.
            self._processor, self._model = processor, model
            self._load_sec = time.perf_counter() - started
            LOGGER.info("VLM loaded backend=%s model=%s checksum=%s device=%s load_sec=%.3f",
                        type(self).__name__, self.model_version, self.model_checksum, "cuda:0", self._load_sec)

    def _generate_tokens(self, inputs: Any, max_new_tokens: int) -> Any:
        # Transformers checks this between decoding steps; this is not a hard GPU watchdog.
        started = time.perf_counter()
        generated = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            max_time=self.max_generation_sec,
        )
        if time.perf_counter() - started >= self.max_generation_sec:
            raise VerificationRuntimeError("Qwen generation exceeded its time budget")
        return generated

    def generate(self, images: Sequence[Path], prompt: str, *, max_new_tokens: int) -> BackendOutput:
        self._load()
        import torch
        from PIL import Image, UnidentifiedImageError

        with self._jobs:
            loaded = []
            try:
                for path in images:
                    with Image.open(path) as source:
                        source.verify()
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                        image.thumbnail((self.max_image_side, self.max_image_side))
                        loaded.append(image.copy())
            except (OSError, UnidentifiedImageError) as exc:
                raise VerificationRuntimeError("Invalid JPEG evidence") from exc
            messages = [{"role": "user", "content": [
                *({"type": "image", "image": image} for image in loaded),
                {"type": "text", "text": prompt},
            ]}]
            try:
                inputs = self._processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(self._model.device)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = self._generate_tokens(inputs, max_new_tokens)
                inference_sec = time.perf_counter() - started
                input_tokens = inputs["input_ids"].shape[-1]
                text = self._processor.decode(generated[0, input_tokens:], skip_special_tokens=True)
                peak = torch.cuda.max_memory_allocated() / 1024**3
            except VerificationRuntimeError:
                raise
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise VerificationRuntimeError("CUDA out of memory during Qwen inference") from exc
            except Exception as exc:
                raise VerificationRuntimeError("Local Qwen inference failed") from exc
            LOGGER.info("VLM inference backend=%s model=%s inference_sec=%.3f peak_vram_gib=%.3f",
                        type(self).__name__, self.model_version, inference_sec, peak)
            return BackendOutput(text.strip(), self._load_sec, inference_sec, peak)


_BACKENDS: dict[tuple[str, str, int], QwenLocalBackend] = {}
_BACKENDS_LOCK = threading.Lock()


def get_qwen_backend(model_path: str | Path, manifest_path: str | Path, *, max_concurrent_jobs: int = 1) -> QwenLocalBackend:
    key = (str(Path(model_path).resolve()), str(Path(manifest_path).resolve()), max_concurrent_jobs)
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(key)
        if backend is None:
            backend = QwenLocalBackend(model_path, manifest_path, max_concurrent_jobs=max_concurrent_jobs)
            _BACKENDS[key] = backend
    return backend


def clear_backend_cache() -> None:
    _BACKENDS.clear()


class EventVerifier:
    def __init__(self, backend: VisionLanguageBackend, signal_policy: dict[str, Any], *,
                 max_new_tokens: int = 768, rule_catalog: dict[str, dict[str, str]] | None = None) -> None:
        self.backend = backend
        self.signal_policy = signal_policy
        self.max_new_tokens = max_new_tokens
        self.rule_catalog = rule_catalog
        self.allowed_signal_ids = {item["id"] for item in signal_policy["signals"]}

    def verify(self, context: EventContext) -> VerificationResult:
        if not 1 <= len(context.frames) <= 5:
            raise ValueError("Verifier requires one to five unique temporal frames")
        if list(context.frames) != sorted(context.frames, key=lambda item: item.timestamp_sec):
            raise ValueError("Temporal frames must be chronological")
        missing = [frame.jpeg_path for frame in context.frames if not frame.jpeg_path.is_file()]
        if missing:
            raise FileNotFoundError("Temporal evidence image is missing")
        last_error: VerificationOutputError | None = None
        total_inference_sec = 0.0
        maximum_load_sec = 0.0
        maximum_vram = None
        for attempt in range(2):
            output = self.backend.generate(
                [frame.jpeg_path for frame in context.frames],
                build_prompt(
                    context,
                    self.signal_policy,
                    repair_error=str(last_error) if attempt == 1 else None,
                    rule_catalog=self.rule_catalog,
                ),
                max_new_tokens=self.max_new_tokens,
            )
            total_inference_sec += output.inference_sec
            maximum_load_sec = max(maximum_load_sec, output.load_sec)
            if output.peak_vram_gib is not None:
                maximum_vram = max(maximum_vram or 0.0, output.peak_vram_gib)
            measured = BackendOutput(output.text, maximum_load_sec, total_inference_sec, maximum_vram)
            try:
                return parse_verification_output(
                    output.text, context=context, allowed_signal_ids=self.allowed_signal_ids,
                    rule_catalog=self.rule_catalog, model_version=self.backend.model_version,
                    model_checksum=self.backend.model_checksum, backend_output=measured,
                    retry_count=attempt,
                )
            except VerificationOutputError as exc:
                last_error = exc
        raise VerificationOutputError(
            f"Verifier output invalid after one controlled retry: {last_error}"
        ) from last_error
