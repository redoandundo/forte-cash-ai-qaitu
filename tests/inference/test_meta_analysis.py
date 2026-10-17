from __future__ import annotations

from pathlib import Path

from fortebank_audit.inference.experience_store import ExperienceStore
from fortebank_audit.inference.meta_analysis import MetaAnalyzer, MetaFeatures


def result(
    *,
    verdict: str = "possible_violation",
    score: float = 0.54,
    mean: float = 0.539,
    std: float = 0.002,
    positive_fraction: float = 1.0,
    coverage: float = 0.95,
) -> dict:
    return {
        "verdict": verdict,
        "score": score,
        "events": [],
        "window_summary": {
            "minimum_score": mean - std,
            "mean_score": mean,
            "maximum_score": score,
            "score_std": std,
            "positive_fraction": positive_fraction,
            "longest_positive_run": 8,
            "event_coverage_fraction": coverage,
        },
        "runtime": {"model_version": "test-model"},
    }


def test_diffuse_positive_abstains_for_human_review() -> None:
    decision = MetaAnalyzer().evaluate(result(), threshold=0.534)
    assert decision.disposition == "abstain"
    assert decision.human_review_required is True
    assert "diffuse_nonlocalized_activation" in decision.reasons


def test_clear_negative_is_retained_for_sampling() -> None:
    decision = MetaAnalyzer().evaluate(
        result(
            verdict="no_violation",
            score=0.2,
            mean=0.15,
            std=0.01,
            positive_fraction=0.0,
            coverage=0.0,
        ),
        threshold=0.534,
    )
    assert decision.disposition == "no_signal"
    assert decision.human_review_required is False


def test_reviewed_history_can_flag_false_positive_neighborhood(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    baseline_result = result()
    features = MetaFeatures.from_result(baseline_result)
    baseline_decision = MetaAnalyzer().evaluate(baseline_result, threshold=0.534)
    for index in range(3):
        run_id = f"reviewed-{index}"
        video = tmp_path / f"sample-{index}.mov"
        video.write_bytes(f"synthetic-video-{index}".encode())
        store.record_run(
            run_id=run_id,
            video_path=video,
            model_version="test-model",
            model_sha256="model-hash",
            config={"threshold": 0.534},
            result=baseline_result,
            decision=baseline_decision,
            features=features,
        )
        store.add_review(run_id, outcome="confirmed_negative", error_type="false_positive")

    decision = MetaAnalyzer(store).evaluate(baseline_result, threshold=0.534)
    assert decision.context_examples_used == 3
    assert "historical_false_positive_neighborhood" in decision.reasons
    assert store.stats() == {
        "runs": 3,
        "reviewed_runs": 3,
        "review_outcomes": {"confirmed_negative": 3},
    }
    plan = store.improvement_plan(minimum_unique_videos=4)
    assert plan["unique_reviewed_videos"] == 3
    assert plan["automatic_weight_update_allowed"] is False
    assert plan["ranked_actions"] == [
        {"action": "mine_reviewed_hard_negative_windows", "evidence": 3}
    ]


def test_unknown_review_run_is_rejected(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    try:
        store.add_review("missing", outcome="confirmed_negative")
    except KeyError as exc:
        assert "Unknown run_id" in str(exc)
    else:
        raise AssertionError("Unknown runs must not accept feedback")


def test_repeated_runs_of_one_video_count_as_one_memory(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    video = tmp_path / "same.mov"
    video.write_bytes(b"same-video")
    baseline_result = result()
    features = MetaFeatures.from_result(baseline_result)
    decision = MetaAnalyzer().evaluate(baseline_result, threshold=0.534)
    for index in range(3):
        run_id = f"duplicate-{index}"
        store.record_run(
            run_id=run_id,
            video_path=video,
            model_version="test-model",
            model_sha256="model-hash",
            config={},
            result=baseline_result,
            decision=decision,
            features=features,
        )
        store.add_review(run_id, outcome="confirmed_negative")
    assert len(
        store.nearest_reviewed(
            features, model_version="test-model", limit=3, max_distance=0.15
        )
    ) == 1
