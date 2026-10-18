"""Full-video supervision without inventing negative labels inside positive videos."""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .experiment_data import fingerprint, source_identity, validate_sources
from .evaluation import pipeline_hash
from .models import build_model
from .preprocessing import VideoSpec, normalize_rgb
from .training import seed_everything
from .video import cached_video


def smooth_probabilities(logits):
    probabilities = logits.sigmoid()
    if probabilities.numel() < 3:
        return probabilities
    padded = F.pad(probabilities[None, None], (1, 1), mode="replicate")
    kernel = logits.new_tensor([.25, .5, .25])[None, None]
    return F.conv1d(padded, kernel).flatten()


def video_loss(logits, label: int, anchors: list[int], point_weight: float,
               hard_negative_weight: float = 0.):
    if not math.isfinite(hard_negative_weight) or hard_negative_weight < 0:
        raise ValueError("Invalid hard negative weight")
    if logits.ndim != 1 or logits.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError("MIL expects a nonempty finite video timeline")
    peak = smooth_probabilities(logits).max().clamp(1e-6, 1 - 1e-6)
    if label == 1:
        if not anchors or any(index < 0 or index >= len(logits) for index in anchors):
            raise ValueError("Positive videos require valid point anchors")
        # Other positive-video windows remain unlabelled, not artificial negatives.
        return -peak.log() + point_weight * F.softplus(-logits[anchors]).mean()
    if label != 0 or anchors:
        raise ValueError("Negative videos cannot contain positive point anchors")
    # The full negative timeline contributes, with additional pressure on its peak.
    loss = -(1 - peak).log() + .5 * F.softplus(logits).mean()
    if hard_negative_weight:
        # Mine only inside labelled-normal training videos, never held-out errors.
        hardest = logits.topk(max(1, math.ceil(len(logits) / 4))).values
        loss = loss + hard_negative_weight * F.softplus(hardest).mean()
    return loss


def encoder_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.encoder.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_features(saved, key, spec, feature_dim):
    features, centers = saved.get("features"), saved.get("centers")
    if (saved.get("key") != key or not isinstance(features, torch.Tensor)
            or not isinstance(centers, torch.Tensor) or features.ndim != 3
            or features.shape[1:] != (spec.num_frames, feature_dim)
            or centers.shape != (len(features),) or not len(features)
            or not features.is_floating_point() or not centers.is_floating_point()
            or not torch.isfinite(features).all() or not torch.isfinite(centers).all()
            or (centers < 0).any() or (centers[1:] <= centers[:-1]).any()):
        raise ValueError("Invalid feature cache")


def encode_source(model, source, spec, cache, device, encoder_digest, batch_size, condition="clean"):
    path = Path(source["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Source checksum changed before feature extraction")
    identity = {"source": source["sha256"], "spec": spec.to_dict(),
                "encoder": encoder_digest, "pipeline": pipeline_hash(spec),
                "features": "class_and_spatial_2x2_v1"}
    if condition != "clean":
        from .stress_audit import CONDITIONS, transform_frames
        if condition not in CONDITIONS:
            raise ValueError("Unknown encoder augmentation")
        identity["augmentation"] = {"condition": condition,
            "code_sha256": hashlib.sha256(inspect.getsource(transform_frames).encode()).hexdigest()}
    key = fingerprint(identity)
    target = cache / "dino_features" / (key + ".pt")
    if target.exists():
        saved = torch.load(target, map_location="cpu", weights_only=True)
        validate_features(saved, key, spec, model.feature_dim)
        return saved
    samples = cached_video(path, spec, cache)
    model.eval()
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(samples.frames), batch_size):
            frames_batch = samples.frames[start:start + batch_size]
            if condition != "clean":
                frames_batch = [transform_frames(frames.copy(), condition) for frames in frames_batch]
            inputs = np.stack([normalize_rgb(frames, spec.image_size) for frames in frames_batch])
            values = model.encode(torch.from_numpy(inputs).to(device)).float().cpu()
            if not torch.isfinite(values).all():
                raise ValueError("Non-finite pretrained features")
            encoded.append(values)
    saved = {"key": key, "features": torch.cat(encoded), "centers": torch.from_numpy(samples.centers.copy())}
    validate_features(saved, key, spec, model.feature_dim)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(target)
    return saved


def multi_view_loss(logits, label, anchors, point_weight, consistency_weight,
                    hard_negative_weight=0.):
    if logits.ndim != 2 or not len(logits) or not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("Invalid multi-view objective")
    supervised = torch.stack([video_loss(view, label, anchors, point_weight, hard_negative_weight)
                              for view in logits]).mean()
    consistency = logits.sigmoid().var(dim=0, unbiased=False).mean()
    return supervised + consistency_weight * consistency


def train_mil(sources, config, output, cache):
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing MIL run")
    validate_sources(sources)
    if config["model"] != "dinov2_small_spatial_temporal":
        raise ValueError("Temporal MIL requires the spatial temporal encoder")
    if {row["label"] for row in sources} != {0, 1}:
        raise ValueError("MIL training requires both classes")
    settings = config["training"]
    views = settings.get("feature_views", ["clean"])
    from .stress_audit import CONDITIONS
    if (not isinstance(views, list) or not views or any(not isinstance(view, str) for view in views) or views[0] != "clean"
            or len(views) != len(set(views)) or any(view not in CONDITIONS for view in views)):
        raise ValueError("Feature views must be unique supported conditions beginning with clean")
    consistency_weight = settings.get("consistency_weight", 0.)
    hard_negative_weight = settings.get("hard_negative_weight", 0.)
    if not math.isfinite(hard_negative_weight) or hard_negative_weight < 0:
        raise ValueError("Invalid hard negative weight")
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("Invalid consistency weight")
    if settings["epochs"] < 1 or settings["encoder_batch_size"] < 1:
        raise ValueError("Invalid MIL training schedule")
    for name in ("learning_rate", "weight_decay", "feature_noise_std", "point_weight", "pair_weight", "pair_margin"):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError("Invalid MIL hyperparameter")
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    spec = VideoSpec(**config["data"])
    if spec.image_size != 224:
        raise ValueError("This spatial MIL recipe requires 224x224 inputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], pretrained=True).to(device)
    digest = encoder_hash(model)
    started = time.perf_counter()
    bags = []
    totals = Counter()
    for source in sources:
        saved = encode_source(model, source, spec, cache, device, digest, settings["encoder_batch_size"])
        variants = [saved["features"]]
        for condition in views[1:]:
            augmented = encode_source(model, source, spec, cache, device, digest, settings["encoder_batch_size"], condition)
            if not torch.equal(augmented["centers"], saved["centers"]):
                raise ValueError("Augmentation changed the annotation timeline")
            variants.append(augmented["features"])
        points = source.get("events_sec", [])
        if source.get("events"):
            points = [(item["start_sec"] + item["end_sec"]) / 2 for item in source["events"]]
        anchors = sorted({int(torch.argmin((saved["centers"] - point).abs())) for point in points})
        weight = {"high": 1., "medium": .5, "low": .25}[source.get("annotation_confidence", "high")] if source["label"] else 1.
        totals[source["label"]] += weight
        bags.append({"features": torch.stack(variants).to(device), "anchors": anchors, "weight": weight,
                     "label": source["label"], "group": source["scenario_group"]})
        print({"encoded_training_sources": len(bags), "windows": len(saved["features"])}, flush=True)
    for bag in bags:
        bag["weight"] /= 2 * totals[bag["label"]]
    output.mkdir(parents=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, group_scores = [], {}
        for bag in bags:
            features = bag["features"]
            if settings["feature_noise_std"]:
                features = features + torch.randn_like(features) * settings["feature_noise_std"]
            logits = model.forward_encoded(features.flatten(0, 1)).reshape(features.shape[:2])
            losses.append(bag["weight"] * multi_view_loss(logits, bag["label"], bag["anchors"],
                                                         settings["point_weight"], consistency_weight,
                                                         hard_negative_weight))
            peak = torch.stack([smooth_probabilities(view).max() for view in logits]).mean()
            group_scores.setdefault(bag["group"], {}).setdefault(bag["label"], []).append(peak)
        pairs = [F.softplus((torch.stack(scores[0]).max() - torch.stack(scores[1]).max()) * 4 + settings["pair_margin"])
                 for scores in group_scores.values() if set(scores) == {0, 1}]
        loss = torch.stack(losses).sum()
        if pairs:
            loss = loss + settings["pair_weight"] * torch.stack(pairs).mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite MIL objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
        optimizer.step()
        history.append({"epoch": epoch, "loss": loss.item()})
        if epoch == 1 or epoch % 10 == 0:
            print(history[-1], flush=True)
    provenance = {"train_source_ids": sorted(row["source_id"] for row in sources),
        "train_sha256": sorted(row["sha256"] for row in sources),
        "train_scenario_groups": sorted({row["scenario_group"] for row in sources}),
        "dataset_sha256": fingerprint([source_identity(row) for row in sorted(sources, key=lambda row: row["source_id"])]),
        "config_sha256": fingerprint(config), "pretrained_encoder_sha256": digest}
    checkpoint = {"state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "model": config["model"], "frames": spec.num_frames, "image_size": spec.image_size,
        "window_sec": spec.window_sec, "normalization": "imagenet", "spec": spec.to_dict(),
        "output_semantics": "logit", "config": config, "provenance": provenance, "checkpoint_policy": "fixed_last_epoch"}
    torch.save(checkpoint, output / "best.pt")
    return {"model": config["model"], "sources": len(sources), "windows": sum(bag["features"].shape[1] for bag in bags),
            "epochs": settings["epochs"], "feature_views": views, "consistency_weight": consistency_weight,
            "hard_negative_weight": hard_negative_weight,
            "training_sec": time.perf_counter() - started,
            "device": str(device), "history": history, "provenance": provenance,
            "objective": "all_negative_windows_video_peak_positive_point_and_train_pair_ranking",
            "hard_negative_fraction": .25 if hard_negative_weight else None}
