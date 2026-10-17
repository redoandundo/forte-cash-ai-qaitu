from __future__ import annotations

import random
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .experiment_data import SourceWindowDataset, fingerprint, source_identity
from .models import build_model
from .preprocessing import VideoSpec


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weighted_bce(logits, targets, weights, denominator):
    return (nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).sum() / denominator


def train_model(sources: list[dict], config: dict, output: Path, cache: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output.name}")
    settings, data = config["training"], config["data"]
    spec = VideoSpec(**data)
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SourceWindowDataset(sources, spec, cache, augment=settings.get("augmentation", True))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=0, generator=generator)
    model = build_model(config["model"], pretrained=True, train_backbone=False).to(device)
    # Including currently frozen parameters allows later unfreezing without resetting Adam state.
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=settings.get("amp", True) and device.type == "cuda")
    accumulation = int(settings["accumulation_steps"])
    if accumulation <= 0 or settings["epochs"] <= 0:
        raise ValueError("Invalid training schedule")
    output.mkdir(parents=True)
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, settings["epochs"] + 1):
        if settings.get("unfreeze_epoch") == epoch:
            model.set_trainable_blocks(settings["unfreeze_blocks"])
            for group in optimizer.param_groups:
                group["lr"] = settings.get("finetune_learning_rate", settings["learning_rate"])
        model.train()
        iterator = iter(loader)
        numerator, total_weight, steps = 0.0, 0.0, 0
        while group := list(islice(iterator, accumulation)):
            denominator = sum(weights.sum().item() for _, _, weights in group)
            optimizer.zero_grad(set_to_none=True)
            for frames, targets, weights in group:
                frames, targets, weights = frames.to(device), targets.to(device), weights.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    loss = weighted_bce(model(frames), targets, weights, denominator)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                scaler.scale(loss).backward()
                numerator += loss.item() * denominator
                total_weight += weights.sum().item()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
        row = {"epoch": epoch, "loss": numerator / total_weight, "optimizer_steps": steps, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        history.append(row)
        print(row, flush=True)
    provenance = {
        "train_source_ids": sorted(row["source_id"] for row in sources),
        "train_sha256": sorted(row["sha256"] for row in sources),
        "train_scenario_groups": sorted({row["scenario_group"] for row in sources}),
        "dataset_sha256": fingerprint([source_identity(row) for row in sorted(sources, key=lambda row: row["source_id"])]),
        "config_sha256": fingerprint(config),
    }
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model": config["model"], "frames": spec.num_frames, "image_size": spec.image_size,
        "window_sec": spec.window_sec, "normalization": "imagenet",
        "spec": spec.to_dict(), "output_semantics": "logit", "config": config,
        "provenance": provenance, "checkpoint_policy": "fixed_last_epoch",
    }
    torch.save(checkpoint, output / "best.pt")
    return {
        "model": config["model"], "epochs": settings["epochs"], "windows": len(dataset),
        "training_sec": time.perf_counter() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        "device": str(device), "history": history, "provenance": provenance,
    }
