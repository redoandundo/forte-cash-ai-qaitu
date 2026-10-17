from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fortebank_audit.ml.data import ClipDataset, ScenarioPairDataset
from fortebank_audit.ml.models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate(labels: list[int], scores: list[float], source_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, dict[str, object]] = defaultdict(lambda: {"labels": [], "scores": []})
    for label, score, source_id in zip(labels, scores, source_ids):
        grouped[source_id]["labels"].append(label)
        grouped[source_id]["scores"].append(score)
    return (
        np.array([int(max(value["labels"])) for value in grouped.values()]),
        np.array([float(max(value["scores"])) for value in grouped.values()]),
    )


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(np.unique(scores))
    candidates = np.concatenate(([0.0], (ordered[:-1] + ordered[1:]) / 2, [1.0]))
    best = (0.5, -1.0, -1.0, -1.0)
    for threshold in candidates:
        predictions = scores >= threshold
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        candidate = (float(threshold), float(macro_f1), float(recall), float(f2))
        if candidate[1:] > best[1:]:
            best = candidate
    return best[0], best[3]


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": threshold,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None,
        "confusion_matrix": matrix.tolist(),
        "mean_score_negative": float(scores[labels == 0].mean()) if np.any(labels == 0) else None,
        "mean_score_positive": float(scores[labels == 1].mean()) if np.any(labels == 1) else None,
        "source_count": int(len(labels)),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[float], list[str], float]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    sources: list[str] = []
    started = time.perf_counter()
    for videos, targets, _, source_ids, _ in loader:
        logits = model(videos.to(device, non_blocking=True))
        labels.extend(targets.int().tolist())
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        sources.extend(source_ids)
    return labels, scores, sources, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen video-classification head on leakage-free clips.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "model.yaml")
    parser.add_argument("--model", choices=["mobilenet_v3_small_temporal", "mobilenet_v3_small_temporal_stats", "x3d_xs", "r3d_18"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--manifest", type=Path, help="Override the clip manifest from the config.")
    parser.add_argument("--checkpoint-policy", choices=["best-validation", "last-epoch"], default="best-validation")
    parser.add_argument("--pair-ranking", action="store_true")
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--unfreeze-last-stage", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = int(config["data"]["frames"])
    image_size = int(config["data"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    manifest = args.manifest or PROJECT_ROOT / config["data"]["clip_manifest"]
    normalization = "imagenet" if args.model.startswith("mobilenet") else ("r3d" if args.model == "r3d_18" else "kinetics")
    temporal_span = {"x3d_xs": 0.4, "r3d_18": 0.7}.get(args.model, 1.0)
    train_data = ClipDataset(manifest, "train", frames, image_size, not args.no_augmentation, normalization, temporal_span)
    validation_data = ClipDataset(manifest, "validation", frames, image_size, False, normalization, temporal_span)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=int(config["training"]["num_workers"]), pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    pair_data = ScenarioPairDataset(manifest, "train", frames, image_size, not args.no_augmentation, normalization, temporal_span) if args.pair_ranking else None
    if args.pair_ranking and not pair_data:
        raise ValueError("Pair ranking requested but the training split has no positive/negative scenario pairs")
    pair_loader = DataLoader(pair_data, batch_size=max(1, batch_size // 2), shuffle=True, num_workers=0, pin_memory=device.type == "cuda") if pair_data else None

    model = build_model(args.model, pretrained=True, train_backbone=args.unfreeze_last_stage).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epochs = args.epochs or int(config["training"]["epochs"])
    run_dir = PROJECT_ROOT / "models" / "checkpoints" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_selection = (-1.0, -1.0)
    history = []
    patience = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_ranking = 0.0
        pair_iterator = iter(pair_loader) if pair_loader else None
        for videos, targets, sample_weights, _, _ in train_loader:
            videos = videos.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(videos)
                bce_loss = (loss_function(logits, targets) * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
                ranking_loss = torch.zeros((), device=device)
                if pair_iterator is not None:
                    try:
                        positive, negative, pair_weights, _ = next(pair_iterator)
                    except StopIteration:
                        pair_iterator = iter(pair_loader)
                        positive, negative, pair_weights, _ = next(pair_iterator)
                    positive = positive.to(device, non_blocking=True)
                    negative = negative.to(device, non_blocking=True)
                    pair_weights = pair_weights.to(device, non_blocking=True)
                    positive_logits = model(positive)
                    negative_logits = model(negative)
                    ranking_values = torch.relu(args.ranking_margin - (positive_logits - negative_logits))
                    ranking_loss = (ranking_values * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
                loss = bce_loss + args.ranking_weight * ranking_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(videos)
            total_bce += bce_loss.item() * len(videos)
            total_ranking += ranking_loss.item() * len(videos)
        clip_labels, clip_scores, sources, eval_seconds = evaluate(model, validation_loader, device)
        source_labels, source_scores = aggregate(clip_labels, clip_scores, sources)
        threshold, f2 = best_threshold(source_labels, source_scores)
        epoch_metrics = metrics(source_labels, source_scores, threshold)
        selection = (epoch_metrics["macro_f1"], epoch_metrics["auroc"] or -1.0)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_data), "bce_loss": total_bce / len(train_data), "ranking_loss": total_ranking / len(train_data), "macro_f1": selection[0], "auroc": selection[1], "f2": f2, "threshold": threshold, "eval_seconds": eval_seconds}
        history.append(row)
        print(json.dumps(row))
        if args.checkpoint_policy == "last-epoch":
            torch.save({"state_dict": model.state_dict(), "model": args.model, "frames": frames, "image_size": image_size, "window_sec": float(config["data"]["clip_seconds"]), "normalization": normalization, "temporal_span": temporal_span, "train_backbone": args.unfreeze_last_stage}, run_dir / "best.pt")
        elif selection > best_selection:
            best_selection = selection
            patience = 0
            torch.save({"state_dict": model.state_dict(), "model": args.model, "frames": frames, "image_size": image_size, "window_sec": float(config["data"]["clip_seconds"]), "normalization": normalization, "temporal_span": temporal_span, "train_backbone": args.unfreeze_last_stage}, run_dir / "best.pt")
        else:
            patience += 1
            if patience >= int(config["training"]["early_stopping_patience"]):
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    clip_labels, clip_scores, sources, eval_seconds = evaluate(model, validation_loader, device)
    source_labels, source_scores = aggregate(clip_labels, clip_scores, sources)
    threshold, f2 = best_threshold(source_labels, source_scores)
    result = {
        "run_id": args.run_id,
        "model": args.model,
        "seed": seed,
        "device": str(device),
        "augmentation": not args.no_augmentation,
        "unfreeze_last_stage": args.unfreeze_last_stage,
        "checkpoint_policy": args.checkpoint_policy,
        "manifest": str(manifest),
        "pair_ranking": args.pair_ranking,
        "ranking_weight": args.ranking_weight if args.pair_ranking else 0.0,
        "ranking_margin": args.ranking_margin if args.pair_ranking else 0.0,
        "ranking_pairs": len(pair_data) if pair_data else 0,
        "epochs_completed": len(history),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "clip_metrics": metrics(np.array(clip_labels), np.array(clip_scores), threshold),
        "source_metrics": {**metrics(source_labels, source_scores, threshold), "f2": f2},
        "validation_seconds": eval_seconds,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2) if device.type == "cuda" else None,
        "history": history,
    }
    result_path = run_dir / "metrics.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
