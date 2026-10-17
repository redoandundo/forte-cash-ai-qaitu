from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .vlm_verifier import VerificationResult


def build_report(
    *,
    run_id: str,
    first_layer: dict[str, Any],
    second_layer: dict[str, Any],
    verifications: Sequence[VerificationResult],
    signals_version: str,
) -> dict[str, Any]:
    recommendation = str(second_layer.get("next_action", "human_review"))
    return {
        "report_version": "observable-review-v2",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Decision support only. This report does not establish identity, intent, guilt, "
            "fraud, or a legal violation."
        ),
        "first_layer": first_layer,
        "second_layer": second_layer,
        "visual_verifications": [item.as_dict() for item in verifications],
        "signals_version": signals_version,
        "recommendation": recommendation,
        "qwen_experiment": {
            "enabled": bool(verifications),
            "ab_evaluation_only": True,
            "public_verdict_influence": False,
            "timecode_influence": False,
            "score_influence": False,
        },
        "human_review_required": bool(verifications) or bool(
            second_layer.get("human_review_required")
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local observable-event review",
        "",
        f"Run: `{report['run_id']}`",
        f"Recommendation: `{report['recommendation']}`",
        "",
        report["scope"],
        "",
        "## Candidate events",
        "",
    ]
    events = report["first_layer"].get("events", [])
    verifications = report["visual_verifications"]
    if not events:
        lines.append("No first-layer candidate events were produced.")
    for index, event in enumerate(events):
        lines.extend(
            [
                f"### Event {index + 1}",
                "",
                f"- Peak time: {event['peak_time_sec']:.3f} s",
                f"- First-layer score: {event['score']:.6f}",
            ]
        )
        if index < len(verifications):
            verification = verifications[index]
            lines.extend(
                [
                    f"- Visual status: `{verification['status']}`",
                    f"- Evidence quality: `{verification['evidence_quality']}`",
                    f"- Summary: {verification['summary']}",
                    "- Matched signals: "
                    + (", ".join(verification["matched_signal_ids"]) or "none"),
                    "- Limitations: "
                    + ("; ".join(verification["limitations"]) or "none reported"),
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_report(report: dict[str, Any], output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = str(report["run_id"])
    json_path = root / f"{stem}.json"
    markdown_path = root / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
