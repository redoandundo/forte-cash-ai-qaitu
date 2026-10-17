from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def confusion(labels: list[int], predictions: list[int]) -> list[list[int]]:
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    return [[tn, fp], [fn, tp]]


def metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, object]:
    predictions = [int(score >= threshold) for score in scores]
    (tn, fp), (fn, tp) = confusion(labels, predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    positive_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if negative_precision + negative_recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def threshold_candidates(scores: list[float]) -> list[float]:
    ordered = sorted(set(scores))
    return [0.0, *[(left + right) / 2 for left, right in zip(ordered, ordered[1:])], 1.0]


def best_threshold(labels: list[int], scores: list[float]) -> float:
    def key(threshold: float) -> tuple[float, float, float]:
        result = metrics(labels, scores, threshold)
        return float(result["macro_f1"]), float(result["recall"]), float(result["precision"])

    return max(threshold_candidates(scores), key=key)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stratified_bootstrap(labels: list[int], scores: list[float], threshold: float, samples: int, seed: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    by_label = {label: [index for index, value in enumerate(labels) if value == label] for label in (0, 1)}
    values = {"macro_f1": [], "precision": [], "recall": []}
    for _ in range(samples):
        indices = [rng.choice(by_label[label]) for label in (0, 1) for _ in by_label[label]]
        result = metrics([labels[index] for index in indices], [scores[index] for index in indices], threshold)
        for name in values:
            values[name].append(float(result[name]))
    return {name: [quantile(metric_values, p) for p in (0.025, 0.5, 0.975)] for name, metric_values in values.items()}


def audit(payload: dict, bootstrap_samples: int, seed: int) -> dict:
    videos = payload["videos"]
    labels = [int(video["label"]) for video in videos]
    scores = [float(video["score"]) for video in videos]
    threshold = float(payload["threshold"])
    if set(labels) != {0, 1}:
        raise ValueError("Prediction audit requires both labels")

    loo_predictions: list[int] = []
    loo_thresholds: list[float] = []
    for held_out in range(len(videos)):
        train_labels = labels[:held_out] + labels[held_out + 1 :]
        train_scores = scores[:held_out] + scores[held_out + 1 :]
        fold_threshold = best_threshold(train_labels, train_scores)
        loo_thresholds.append(fold_threshold)
        loo_predictions.append(int(scores[held_out] >= fold_threshold))
    loo_score_proxy = [float(prediction) for prediction in loo_predictions]
    loo = metrics(labels, loo_score_proxy, 0.5)

    margins = [abs(score - threshold) for score in scores]
    bootstrap = stratified_bootstrap(labels, scores, threshold, bootstrap_samples, seed)
    warnings = []
    if payload.get("threshold_selected_on") == "full_validation_videos":
        warnings.append("threshold_selected_on_evaluation_set")
    if len(videos) < 40:
        warnings.append("small_evaluation_set")
    if max(scores) - min(scores) < 0.1:
        warnings.append("narrow_score_range")
    if min(margins) < 0.01:
        warnings.append("decision_boundary_fragile")
    if bootstrap["macro_f1"][0] < 0.5:
        warnings.append("macro_f1_interval_crosses_0_5")

    return {
        "model": payload.get("model"),
        "source_count": len(videos),
        "apparent_metrics": metrics(labels, scores, threshold),
        "fixed_threshold_stratified_bootstrap_95_ci": bootstrap,
        "leave_one_out_threshold_audit": {
            **loo,
            "threshold_min": min(loo_thresholds),
            "threshold_max": max(loo_thresholds),
            "predictions": loo_predictions,
        },
        "score_diagnostics": {
            "minimum": min(scores),
            "maximum": max(scores),
            "span": max(scores) - min(scores),
            "minimum_absolute_margin": min(margins),
            "median_absolute_margin": quantile(margins, 0.5),
        },
        "warnings": warnings,
        "interpretation": "Bootstrap intervals quantify sampling uncertainty at the already selected threshold. Leave-one-out thresholding estimates threshold-selection fragility; it is not a substitute for scenario-group OOF retraining.",
    }


def markdown(result: dict) -> str:
    apparent = result["apparent_metrics"]
    loo = result["leave_one_out_threshold_audit"]
    ci = result["fixed_threshold_stratified_bootstrap_95_ci"]
    score = result["score_diagnostics"]
    return f"""# Evidence audit for model v3

Date: 2026-10-17

## What the reported score means

The current `macro F1 = {apparent['macro_f1']:.3f}` is an apparent result on {result['source_count']} validation sources. The model did not train on these sources, but the final threshold was selected on the same eight scores. It is useful for a hackathon baseline and is not an unbiased production estimate.

## Uncertainty and fragility

| Check | Result |
|---|---:|
| Apparent macro F1 | {apparent['macro_f1']:.3f} |
| Fixed-threshold macro F1 bootstrap 95% interval | {ci['macro_f1'][0]:.3f}–{ci['macro_f1'][2]:.3f} |
| Leave-one-out threshold macro F1 | {loo['macro_f1']:.3f} |
| Leave-one-out precision / recall | {loo['precision']:.3f} / {loo['recall']:.3f} |
| Leave-one-out threshold range | {loo['threshold_min']:.6f}–{loo['threshold_max']:.6f} |
| Full score span | {score['span']:.6f} |
| Smallest distance from threshold | {score['minimum_absolute_margin']:.6f} |

The bootstrap interval is wide because only four positive and four negative sources are present. The narrow score span and small margins mean small domain changes can flip decisions. The leave-one-out calculation audits threshold stability only; five-fold scenario-safe retraining remains required.

## Decision

- Keep v3 as the integration baseline.
- Do not describe `score` as probability or use v3 for autonomous sanctions.
- Do not claim a production quality ceiling from eight sources.
- Complete five-fold scenario-group OOF evaluation before promoting another model.

Warnings: `{', '.join(result['warnings'])}`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit uncertainty and threshold fragility in a full-video prediction report.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "ml" / "evidence_audit_v3.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "reports" / "ml" / "evidence_audit_v3.md")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    result = audit(json.loads(args.input.read_text(encoding="utf-8")), args.bootstrap_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
