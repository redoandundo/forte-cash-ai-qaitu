from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class MetaFeatures:
    peak_score: float
    mean_score: float
    score_std: float
    positive_fraction: float
    event_coverage_fraction: float

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> "MetaFeatures":
        summary = result["window_summary"]
        return cls(
            peak_score=float(result["score"]),
            mean_score=float(summary["mean_score"]),
            score_std=float(summary["score_std"]),
            positive_fraction=float(summary["positive_fraction"]),
            event_coverage_fraction=float(summary["event_coverage_fraction"]),
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class ReviewedContext(Protocol):
    def nearest_reviewed(
        self,
        features: MetaFeatures,
        *,
        model_version: str,
        limit: int,
        max_distance: float,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class MetaDecision:
    disposition: str
    review_priority: str
    human_review_required: bool
    reasons: tuple[str, ...]
    context_examples_used: int
    next_action: str
    layer_version: str = "context-review-v1"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


class MetaAnalyzer:
    """Review-routing layer; it never changes first-layer scores or weights."""

    def __init__(
        self,
        context: ReviewedContext | None = None,
        *,
        uncertainty_margin: float = 0.02,
        diffuse_fraction: float = 0.8,
        history_distance: float = 0.15,
        minimum_history: int = 3,
    ) -> None:
        self.context = context
        self.uncertainty_margin = uncertainty_margin
        self.diffuse_fraction = diffuse_fraction
        self.history_distance = history_distance
        self.minimum_history = minimum_history

    def evaluate(self, result: dict[str, Any], *, threshold: float) -> MetaDecision:
        features = MetaFeatures.from_result(result)
        model_version = str(result["runtime"]["model_version"])
        verdict = str(result["verdict"])
        reasons: list[str] = []

        if verdict == "possible_violation":
            disposition = "human_review"
            priority = "high"
            human_review_required = True
            reasons.append("first_layer_positive_requires_human_review")
            if abs(features.peak_score - threshold) <= self.uncertainty_margin:
                disposition = "abstain"
                reasons.append("peak_score_near_threshold")
            if (
                features.positive_fraction >= self.diffuse_fraction
                or features.event_coverage_fraction >= self.diffuse_fraction
            ):
                disposition = "abstain"
                reasons.append("diffuse_nonlocalized_activation")
        else:
            disposition = "no_signal"
            priority = "low"
            human_review_required = False
            reasons.append("first_layer_below_threshold")
            if threshold - features.peak_score <= self.uncertainty_margin:
                disposition = "abstain"
                priority = "medium"
                human_review_required = True
                reasons.append("negative_result_near_threshold")

        neighbors: list[dict[str, Any]] = []
        if self.context is not None:
            neighbors = self.context.nearest_reviewed(
                features,
                model_version=model_version,
                limit=self.minimum_history,
                max_distance=self.history_distance,
            )
        if len(neighbors) >= self.minimum_history:
            outcomes = {str(item["outcome"]) for item in neighbors}
            if outcomes == {"confirmed_negative"} and verdict == "possible_violation":
                disposition = "abstain"
                priority = "high"
                human_review_required = True
                reasons.append("historical_false_positive_neighborhood")
            elif outcomes == {"confirmed_positive"} and verdict == "possible_violation":
                reasons.append("historical_positive_neighborhood")

        if disposition == "no_signal":
            next_action = "retain_for_periodic_negative_sampling"
        elif disposition == "abstain":
            next_action = "request_human_review_before_any_model_update"
        else:
            next_action = "request_human_review_and_capture_corrected_timecode"
        return MetaDecision(
            disposition=disposition,
            review_priority=priority,
            human_review_required=human_review_required,
            reasons=tuple(reasons),
            context_examples_used=len(neighbors),
            next_action=next_action,
        )
