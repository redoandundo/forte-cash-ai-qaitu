import json
import shutil

import numpy as np
import pytest
import torch

from fortebank_audit.ml.bundle import export_bundle, validate_bundle, write_json
from fortebank_audit.ml.evaluation import assert_unseen, evaluation_report, file_hash, pipeline_hash, promotion_decision
from fortebank_audit.ml.models import build_model
from fortebank_audit.ml.preprocessing import VideoSpec
from fortebank_audit.ml.runtime import predict_video


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("bundle")
    spec = VideoSpec(num_frames=4, image_size=32)
    torch.manual_seed(42)
    model = build_model("mobilenet_v3_small_temporal", pretrained=False)
    checkpoint = folder / "model.pt"
    provenance = {
        "train_source_ids": ["train"], "train_sha256": ["train-sha"],
        "train_scenario_groups": ["train-group"], "dataset_sha256": "dataset", "config_sha256": "config",
    }
    torch.save({"state_dict": model.state_dict(), "model": "mobilenet_v3_small_temporal",
                "spec": spec.to_dict(), "normalization": "imagenet", "output_semantics": "logit", "provenance": provenance}, checkpoint)
    calibration = {
        "checkpoint_sha256": file_hash(checkpoint), "pipeline_sha256": pipeline_hash(spec),
        "calibration_source_ids": ["cal"], "calibration_sha256": ["cal-sha"],
        "calibration_scenario_groups": ["cal-group"], "threshold": 0.5,
        "source_count": 1, "min_recall": 0.75, "metrics": {"macro_f1": 0.5},
        "selection": "macro_f1_subject_to_recall",
    }
    inputs = np.random.default_rng(1).normal(size=(3, 3, 4, 32, 32)).astype(np.float32)
    bundle = folder / "bundle"
    result = export_bundle(checkpoint, calibration, bundle, inputs, "unit-test")
    return checkpoint, calibration, inputs, bundle, result


def rehash(bundle):
    files = sorted(path for path in bundle.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (bundle / "SHA256SUMS.txt").write_text("".join(file_hash(path) + "  " + path.relative_to(bundle).as_posix() + "\n" for path in files), encoding="ascii")


def test_export_fixed_dynamic_schema_and_parity(exported):
    _, _, _, bundle, result = exported
    manifest = validate_bundle(bundle)
    assert manifest["input"]["shape"] == [1, 3, 4, 32, 32]
    assert result["parity"]["max_absolute_logit_error"] <= 1e-4
    assert result["parity"]["decision_agreement"]
    compat_path = bundle / "compatibility/inference/model_manifest.json"
    compat = json.loads(compat_path.read_text())
    assert compat_path.resolve().parents[2] == bundle.resolve()
    assert compat["input_shape"] == [None, 3, 4, 32, 32]
    assert file_hash(bundle / compat["artifact"]) == compat["sha256"]
    assert compat["output_semantics"] == "logit"


def test_cannot_overwrite_bundle(exported):
    checkpoint, calibration, inputs, bundle, _ = exported
    digest = file_hash(bundle / "model.onnx")
    with pytest.raises(FileExistsError):
        export_bundle(checkpoint, calibration, bundle, inputs, "bad")
    assert file_hash(bundle / "model.onnx") == digest


def test_failed_export_is_atomic(exported, tmp_path, monkeypatch):
    checkpoint, calibration, inputs, bundle, _ = exported
    digest = file_hash(bundle / "model.onnx")
    def fail(*args, **kwargs):
        raise RuntimeError("simulated export failure")
    monkeypatch.setattr(torch.onnx, "export", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated"):
        export_bundle(checkpoint, calibration, output, inputs, "failed")
    assert not output.exists()
    assert file_hash(bundle / "model.onnx") == digest


def test_reject_wrong_checkpoint_and_calibration_leak(exported, tmp_path):
    checkpoint, calibration, inputs, _, _ = exported
    for changed, message in [
        ({"checkpoint_sha256": "wrong"}, "does not belong"),
        ({"calibration_scenario_groups": ["train-group"]}, "overlaps training"),
    ]:
        with pytest.raises(ValueError, match=message):
            export_bundle(checkpoint, {**calibration, **changed}, tmp_path / "export", inputs, "bad")
        assert not (tmp_path / "export").exists()


def test_corrupt_model_fails_before_prediction(exported, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(exported[3], bundle)
    with (bundle / "model.onnx").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        predict_video(tmp_path / "missing.mov", bundle)


@pytest.mark.parametrize("field,value", [("mean", [0, 0, 0]), ("image_size", 64), ("sampling", "fake")])
def test_reject_changed_preprocessing_even_with_new_checksums(exported, tmp_path, field, value):
    bundle = tmp_path / "bundle"
    shutil.copytree(exported[3], bundle)
    pre = json.loads((bundle / "preprocessing.json").read_text())
    pre[field] = value
    write_json(bundle / "preprocessing.json", pre)
    rehash(bundle)
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_evaluation_cannot_select_or_reuse_calibration_sources(exported, tmp_path):
    checkpoint, calibration, _, _, _ = exported
    rows = [{"source_id": "cal", "sha256": "cal-sha", "scenario_group": "cal-group"}]
    with pytest.raises(ValueError, match="overlaps threshold calibration"):
        evaluation_report(checkpoint, rows, tmp_path, calibration)
    with pytest.raises(ValueError, match="overlaps training"):
        assert_unseen(rows, {"train_source_ids": [], "train_sha256": [], "train_scenario_groups": ["cal-group"]})


def test_quality_gate_includes_localization_false_events_and_fold_failure():
    baseline = {"macro_f1": .6, "precision": .6, "recall": .75,
                "false_positive_videos_per_minute": .5, "false_positive_events_per_minute": .5}
    candidate = {**baseline, "macro_f1": .7, "recall": .8, "timestamp_mae_sec": 4, "event_recall_within_5_sec": .8}
    folds = [{"macro_f1": .6, "recall": .8}] * 5
    assert promotion_decision(baseline, candidate, folds)["quality_passed"]
    assert not promotion_decision(baseline, candidate, folds)["deployment_ready"]
    for change in ({"recall": .7}, {"timestamp_mae_sec": 6}, {"false_positive_events_per_minute": .7}):
        assert not promotion_decision(baseline, {**candidate, **change}, folds)["quality_passed"]
    assert not promotion_decision(baseline, candidate, [*folds[:4], {"macro_f1": .3, "recall": .8}])["quality_passed"]


def test_pipeline_fingerprint_survives_windows_checkout(tmp_path, monkeypatch):
    from fortebank_audit.ml import evaluation
    monkeypatch.setattr(evaluation, "__file__", str(tmp_path / "evaluation.py"))
    for name in ("video.py", "preprocessing.py", "metrics.py"):
        (tmp_path / name).write_bytes(b"first\nsecond\n")
    unix = pipeline_hash(VideoSpec())
    for name in ("video.py", "preprocessing.py", "metrics.py"):
        (tmp_path / name).write_bytes(b"first\r\nsecond\r\n")
    assert pipeline_hash(VideoSpec()) == unix


@pytest.mark.parametrize("change", [{"smoothing_weights": [0., 1., 0.]}, {"short_sequence_rule": "edge_pad"},
                                   {"event_merge": "overlapping_windows"}, {"max_events": 4},
                                   {"checkpoint_sha256": "different"}, {"violation": .4}])
def test_rehashed_postprocessing_or_threshold_change_is_rejected(exported, tmp_path, change):
    bundle = tmp_path / "bundle"
    shutil.copytree(exported[3], bundle)
    thresholds = json.loads((bundle / "thresholds.json").read_text())
    write_json(bundle / "thresholds.json", {**thresholds, **change})
    rehash(bundle)
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_missing_compatibility_graph_is_rejected(exported, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(exported[3], bundle)
    (bundle / "compatibility/model_dynamic.onnx").unlink()
    rehash(bundle)
    with pytest.raises(ValueError, match="complete bundle"):
        validate_bundle(bundle)


def test_runtime_provides_private_timeline_and_propagates_decode_failure(exported, tmp_path):
    import av
    path = tmp_path / "short.mov"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    result = predict_video(path, exported[3], include_window_scores=True)
    assert result["windows"] == len(result["window_scores"]) == 1
    assert result["window_scores"][0]["center_sec"] == pytest.approx(.6)
    for event in result["events"]:
        assert event["supporting_windows"] == len(event["supporting_window_indices"]) == 1
    broken = tmp_path / "broken.mov"
    broken.write_bytes(b"not a video")
    with pytest.raises(Exception):
        predict_video(broken, exported[3])
