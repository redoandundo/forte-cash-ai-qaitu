"""Paired local ONNX/Qwen evaluation. Detailed outputs must stay outside Git."""
import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

from PIL import Image

from fortebank_audit.ml.bundle import write_json
from fortebank_audit.ml.experiment_data import load_sources
from fortebank_audit.ml.metrics import match_events, summarize_rows
from fortebank_audit.job_runtime import InferenceRuntime, RuntimeSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "bundle", "qwen-model", "qwen-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--allow-development-model", action="store_true")
    parser.add_argument("--reviewed-rules", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new evaluation directory")
    sources = load_sources(args.manifest)
    if args.fold is not None:
        if args.plan is None:
            raise ValueError("Fold selection requires a frozen plan")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        sources = [s for s in sources if plan["folds"][s["source_id"]] == args.fold]
    os.environ.update(QAITU_VLM_MODE="qwen", QAITU_VLM_MODEL_PATH=str(args.qwen_model.resolve()),
                      QAITU_VLM_MANIFEST_PATH=str(args.qwen_manifest.resolve()),
                      QAITU_VLM_REVIEWED_RULES="1" if args.reviewed_rules else "0",
                      HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    import torch
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    settings = dict(mode="onnx", bundle_root=args.bundle, allow_development_model=args.allow_development_model,
                    evidence_root=args.output / "evidence", prefer_cuda=False)
    baseline = InferenceRuntime(RuntimeSettings(**settings))
    started = time.perf_counter()
    semantic = InferenceRuntime(RuntimeSettings(**settings, vlm_mode="qwen"))
    startup_sec = time.perf_counter() - started
    summary = {"scope": "paired_development_evaluation_not_independent_holdout", "selected_fold": args.fold,
               "sources": len(sources), "startup_sec": startup_sec, "models": semantic.model_diagnostics,
               "reviewed_rules_enabled": args.reviewed_rules, "videos": [], "failures": []}
    rows = {"onnx": [], "qwen": []}
    for index, source in enumerate(sources):
        record = {"source_id": source["source_id"], "label": source["label"], "stages": {}}
        for name, runtime in (("onnx", baseline), ("qwen", semantic)):
            started = time.perf_counter()
            try:
                result = runtime.analyze(source["path"], job_id=f"{name}-{index}")
                elapsed = time.perf_counter() - started
                for event in result.violations:
                    for frame in event["evidence_frames"]:
                        with Image.open(io.BytesIO(base64.b64decode(frame["image_base64"], validate=True))) as image:
                            if image.format != "JPEG":
                                raise ValueError("Non-JPEG evidence")
                            image.verify()
                events = result.runtime["diagnostics"]["events"]
                row = {"source_id": source["source_id"], "scenario_group": source["scenario_group"],
                       "label": source["label"], "prediction": int(result.verdict == "violation"),
                       "duration_sec": source["duration_sec"], "events": events,
                       "event_match": match_events([e["peak_time_sec"] for e in events], source.get("events_sec", []))}
                rows[name].append(row)
                record["stages"][name] = {**row, "seconds": elapsed,
                    "within_300_sec": elapsed <= 300, "evidence_valid": True,
                    "semantic": result.runtime["diagnostics"]["semantic"],
                    "analysis_details": result.analysis_details, "runtime": result.runtime}
            except Exception as exc:
                summary["failures"].append({"index": index, "stage": name, "error_type": type(exc).__name__})
                record["stages"][name] = {"failed": True, "error_type": type(exc).__name__}
        write_json(args.output / f"video_{index:02}.json", record)
        summary["videos"].append({"index": index, "stages": {
            name: {k: value for k, value in stage.items() if k in {"prediction", "seconds", "within_300_sec", "failed", "error_type"}}
            for name, stage in record["stages"].items()}})
        write_json(args.output / "progress.json", summary)
        print({"completed": index + 1, "total": len(sources), "failures": len(summary["failures"])}, flush=True)
    # Failed videos are not silently treated as normal or omitted from a quality claim.
    summary["metrics"] = {name: summarize_rows(values) if len(values) == len(sources) else None
                          for name, values in rows.items()}
    if not summary["failures"]:
        summary["positive_videos_lost"] = sum(a["label"] and a["prediction"] and not b["prediction"]
                                              for a, b in zip(rows["onnx"], rows["qwen"], strict=True))
    write_json(args.output / "summary.json", summary)
    print({"metrics": summary["metrics"], "failures": len(summary["failures"])}, flush=True)


if __name__ == "__main__":
    main()
