from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def balanced_group_folds(sources: list[dict], fold_count: int, seed: int, attempts: int = 20_000) -> dict[str, int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for source in sources:
        groups[source["scenario_group"]].append(source)
    group_rows = [(name, sum(item["label"] == 0 for item in rows), sum(item["label"] == 1 for item in rows)) for name, rows in groups.items()]
    target_negative = sum(row[1] for row in group_rows) / fold_count
    target_positive = sum(row[2] for row in group_rows) / fold_count
    target_total = len(sources) / fold_count
    rng = random.Random(seed)
    best_score = float("inf")
    best_assignment: dict[str, int] = {}
    for _ in range(attempts):
        counts = [[0, 0, 0] for _ in range(fold_count)]
        assignment = {}
        mixed = [row for row in group_rows if row[1] and row[2]]
        other = [row for row in group_rows if not (row[1] and row[2])]
        rng.shuffle(mixed)
        rng.shuffle(other)
        if len(mixed) < fold_count:
            raise ValueError("Not enough mixed-label scenario groups to seed every fold")
        initial_folds = list(range(fold_count))
        rng.shuffle(initial_folds)
        for fold, (group, negative, positive) in zip(initial_folds, mixed[:fold_count]):
            assignment[group] = fold
            counts[fold] = [negative, positive, negative + positive]
        remaining = mixed[fold_count:] + other
        rng.shuffle(remaining)
        for group, negative, positive in sorted(remaining, key=lambda row: row[1] + row[2], reverse=True):
            candidates = list(range(fold_count))
            rng.shuffle(candidates)
            def total_score(index: int) -> float:
                proposed = [row[:] for row in counts]
                proposed[index][0] += negative
                proposed[index][1] += positive
                proposed[index][2] += negative + positive
                return sum(
                    (neg - target_negative) ** 2 + (pos - target_positive) ** 2 + 0.5 * (total - target_total) ** 2
                    for neg, pos, total in proposed
                )
            fold = min(candidates, key=total_score)
            assignment[group] = fold
            counts[fold][0] += negative
            counts[fold][1] += positive
            counts[fold][2] += negative + positive
        if any(negative == 0 or positive == 0 for negative, positive, _ in counts):
            continue
        score = sum(
            (negative - target_negative) ** 2 + (positive - target_positive) ** 2 + 0.5 * (total - target_total) ** 2
            for negative, positive, total in counts
        )
        if score < best_score:
            best_score, best_assignment = score, assignment
    if not best_assignment:
        raise ValueError("Could not build balanced folds containing both labels")
    return best_assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenario-safe source and clip manifests for every existing OOF fold.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "analysis" / "dataset_manifest.json")
    parser.add_argument("--clips", type=Path, default=ROOT / "analysis" / "clip_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "oof")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-existing-folds", action="store_true")
    parser.add_argument("--hard-negative-clips", type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    clips = json.loads(args.clips.read_text(encoding="utf-8"))
    if args.hard_negative_clips:
        hard_negatives = json.loads(args.hard_negative_clips.read_text(encoding="utf-8"))["clips"]
        clips = {**clips, "clips": clips["clips"] + hard_negatives}
    sources = dataset["sources"]
    source_by_id = {source["source_id"]: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("source_id values must be unique")

    group_folds: dict[str, set[int]] = defaultdict(set)
    for source in sources:
        if not source.get("scenario_group"):
            raise ValueError(f"Missing scenario_group for {source['source_id']}")
        if source.get("fold") is not None:
            group_folds[source["scenario_group"]].add(int(source["fold"]))
    leaking = {group: sorted(folds) for group, folds in group_folds.items() if len(folds) != 1}
    if leaking:
        raise ValueError(f"Scenario groups cross folds: {leaking}")

    if args.use_existing_folds:
        folds = sorted({int(source["fold"]) for source in sources})
        if folds != list(range(len(folds))):
            raise ValueError(f"Folds must be contiguous from zero, got {folds}")
        assignment = {source["scenario_group"]: int(source["fold"]) for source in sources}
        plan = "existing"
    else:
        folds = list(range(args.folds))
        assignment = balanced_group_folds(sources, args.folds, args.seed)
        plan = "deterministic_balanced_group_search"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"folds": len(folds), "sources": len(sources), "scenario_groups": len(group_folds), "plan": plan, "seed": args.seed, "folds_detail": []}
    for fold in folds:
        fold_sources = []
        counts = Counter()
        for original in sources:
            source = dict(original)
            source["fold"] = assignment[source["scenario_group"]]
            source["split"] = "validation" if source["fold"] == fold else "train"
            fold_sources.append(source)
            counts[(source["split"], int(source["label"]))] += 1
        validation_labels = {source["label"] for source in fold_sources if source["split"] == "validation"}
        if validation_labels != {0, 1}:
            raise ValueError(f"Fold {fold} does not contain both labels")

        fold_clips = []
        for original in clips["clips"]:
            source = source_by_id.get(original["source_id"])
            if source is None:
                raise ValueError(f"Unknown source in clip manifest: {original['source_id']}")
            clip = dict(original)
            clip["split"] = "validation" if assignment[source["scenario_group"]] == fold else "train"
            fold_clips.append(clip)

        dataset_payload = {**dataset, "active_oof_fold": fold, "sources": fold_sources}
        clip_payload = {
            **clips,
            "active_oof_fold": fold,
            "summary": {
                "clips": len(fold_clips),
                "positive": sum(item["label"] == 1 for item in fold_clips),
                "negative": sum(item["label"] == 0 for item in fold_clips),
                "train": sum(item["split"] == "train" for item in fold_clips),
                "validation": sum(item["split"] == "validation" for item in fold_clips),
            },
            "clips": fold_clips,
        }
        (args.output_dir / f"fold_{fold}_dataset.json").write_text(json.dumps(dataset_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / f"fold_{fold}_clips.json").write_text(json.dumps(clip_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        calibration_fold = (fold + 1) % len(folds)
        nested_clips = []
        for original in clips["clips"]:
            source_fold = assignment[source_by_id[original["source_id"]]["scenario_group"]]
            clip = dict(original)
            clip["split"] = "excluded" if source_fold == fold else ("validation" if source_fold == calibration_fold else "train")
            nested_clips.append(clip)
        calibration_sources = []
        for original in sources:
            source = dict(original)
            source_fold = assignment[source["scenario_group"]]
            source["fold"] = source_fold
            source["split"] = "validation" if source_fold == calibration_fold else "excluded"
            calibration_sources.append(source)
        nested_clip_payload = {
            **clips,
            "outer_fold": fold,
            "calibration_fold": calibration_fold,
            "summary": {
                "clips": len(nested_clips),
                "train": sum(item["split"] == "train" for item in nested_clips),
                "validation": sum(item["split"] == "validation" for item in nested_clips),
                "excluded": sum(item["split"] == "excluded" for item in nested_clips),
            },
            "clips": nested_clips,
        }
        calibration_payload = {**dataset, "outer_fold": fold, "calibration_fold": calibration_fold, "sources": calibration_sources}
        (args.output_dir / f"outer_{fold}_train_calibration_clips.json").write_text(json.dumps(nested_clip_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / f"outer_{fold}_calibration_dataset.json").write_text(json.dumps(calibration_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["folds_detail"].append({
            "fold": fold,
            "calibration_fold": calibration_fold,
            "train_sources": counts[("train", 0)] + counts[("train", 1)],
            "validation_sources": counts[("validation", 0)] + counts[("validation", 1)],
            "validation_negative": counts[("validation", 0)],
            "validation_positive": counts[("validation", 1)],
        })

    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
