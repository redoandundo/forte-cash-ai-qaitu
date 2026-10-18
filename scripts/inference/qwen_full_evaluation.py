from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fortebank_audit.inference.benchmark import build_predictor, load_pipeline_config
from fortebank_audit.inference.pipeline import analyze_video
from fortebank_audit.inference.semantic_pipeline import apply_semantic_verification
from fortebank_audit.inference.vlm_verifier import (
    EventVerifier,
    get_qwen_backend,
    load_signal_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RecordingVerifier:
    """Record validated model decisions without logging images or sensitive paths."""

    def __init__(self, verifier: EventVerifier) -> None:
        self.verifier = verifier
        self.backend = verifier.backend
        self.results: list[dict[str, Any]] = []

    def verify(self, context: Any) -> Any:
        result = self.verifier.verify(context)
        self.results.append(result.as_dict())
        return result


def confusion(records: list[dict[str, Any]], prediction_field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    mapping = {(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}
    for item in records:
        prediction = item.get(prediction_field)
        if prediction is None:
            continue
        counts[mapping[(int(item["label"]), int(prediction))]] += 1
    return {key: counts[key] for key in ("tp", "tn", "fp", "fn")}


def summarize(records: list[dict[str, Any]], *, model_load_sec: float, wall_sec: float) -> dict[str, Any]:
    completed = [item for item in records if item["technical_status"] == "done"]
    failures = Counter(
        item.get("error_type", "unknown")
        for item in records
        if item["technical_status"] != "done"
    )
    unique: dict[str, dict[str, Any]] = {}
    for item in records:
        unique.setdefault(item["source_id"], item)
    return {
        "report_version": "qwen-full-local-v1",
        "files": len(records),
        "unique_sources": len(unique),
        "baseline_confusion_files": confusion(records, "baseline_prediction"),
        "after_confusion_files_completed": confusion(records, "after_prediction"),
        "baseline_confusion_unique_sources": confusion(list(unique.values()), "baseline_prediction"),
        "after_confusion_unique_sources_completed": confusion(list(unique.values()), "after_prediction"),
        "candidate_events": sum(item.get("baseline_events", 0) for item in records),
        "retained_events": sum(item.get("after_events", 0) for item in completed),
        "removed_events": sum(item.get("removed_events", 0) for item in completed),
        "removed_events_on_positive_videos": sum(
            item.get("removed_events", 0) for item in completed if item["label"] == 1
        ),
        "positive_videos_flipped_to_negative": sum(
            item["label"] == 1
            and item["baseline_prediction"] == 1
            and item.get("after_prediction") == 0
            for item in completed
        ),
        "responses_requiring_repair": sum(
            result["retry_count"]
            for item in records
            for result in item.get("qwen_results", [])
        ),
        "unrecovered_schema_failures": failures["VerificationOutputError"],
        "technical_failures": dict(sorted(failures.items())),
        "model_load_sec": model_load_sec,
        "max_video_total_sec": max((item["total_sec"] for item in records), default=0.0),
        "peak_vram_gib": max(
            (item.get("peak_vram_gib", 0.0) for item in completed), default=0.0
        ),
        "wall_sec": wall_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ONNX -> Qwen on all 42 local video files")
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()

    dataset = json.loads(
        (PROJECT_ROOT / "analysis" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    data_root = PROJECT_ROOT / "data" / "raw" / "open" / "videos"
    files = [
        (source, alias, data_root / alias)
        for source in dataset["sources"]
        for alias in source["aliases"]
    ]
    missing = [path for _, _, path in files if not path.is_file()]
    if missing or len(files) != 42:
        raise RuntimeError("The complete 42-file local dataset is required")

    _, config, _ = load_pipeline_config(None, allow_development_model=False)
    predictor = build_predictor(
        "onnx", None, allow_development_model=False, prefer_cuda=False
    )
    backend = get_qwen_backend(
        args.qwen_model,
        PROJECT_ROOT / "configs" / "inference" / "qwen_verifier_manifest.json",
    )
    backend.preload()
    verifier = EventVerifier(
        backend,
        load_signal_policy(
            PROJECT_ROOT / "configs" / "inference" / "observable_signals.json"
        ),
        max_new_tokens=int(backend.manifest["max_new_tokens"]),
    )

    records: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    for index, (source, alias, path) in enumerate(files, start=1):
        started = time.perf_counter()
        record: dict[str, Any] = {
            "source_id": source["source_id"],
            "video": alias,
            "label": int(source["label"]),
        }
        recorder = RecordingVerifier(verifier)
        try:
            first = analyze_video(
                path,
                predictor,
                config,
                evidence_root=args.evidence_root,
                run_id=f"{args.run_tag}-{index:02d}",
                include_temporal_context=True,
            )
            record.update({
                "duration_sec": first["runtime"]["duration_sec"],
                "onnx_sec": first["runtime"]["processing_sec"],
                "baseline_prediction": int(bool(first["events"])),
                "baseline_events": len(first["events"]),
            })
            filtered = apply_semantic_verification(first, recorder, args.evidence_root)
            record.update({
                "technical_status": "done",
                "after_prediction": int(bool(filtered["events"])),
                "after_events": len(filtered["events"]),
                "removed_events": len(first["events"]) - len(filtered["events"]),
                "qwen_results": recorder.results,
                "qwen_inference_sec": filtered.get("runtime", {}).get("qwen", {}).get(
                    "inference_sec", 0.0
                ),
                "peak_vram_gib": filtered.get("runtime", {}).get("qwen", {}).get(
                    "peak_vram_gib", 0.0
                ),
            })
        except Exception as exc:
            record.update({
                "technical_status": "failed",
                "baseline_prediction": record.get("baseline_prediction", 0),
                "after_prediction": None,
                "baseline_events": record.get("baseline_events", 0),
                "after_events": None,
                "removed_events": 0,
                "qwen_results": recorder.results,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        record["total_sec"] = time.perf_counter() - started
        records.append(record)
        payload = {
            "summary": summarize(
                records,
                model_load_sec=backend._load_sec,
                wall_sec=time.perf_counter() - wall_started,
            ),
            "records": records,
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{index:02d}/42] status={record['technical_status']} "
            f"events={record['baseline_events']}->{record.get('after_events')} "
            f"sec={record['total_sec']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
