from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.domain.news_v2_05_explanations import (
    ExplanationDraft,
    explanation_draft_to_dict,
)


@dataclass(frozen=True)
class OutwardResult:
    outward_id: str
    episode_id: int
    ticker: str
    outward_status: str
    manual_review: bool
    capability_boundary_notice: str | None
    degraded_reason_codes: tuple[str, ...]
    outward_notes: tuple[str, ...]
    explanation: dict[str, Any] | None


def build_outward_result(
    *,
    episode_id: int,
    ticker: str,
    explanation: ExplanationDraft | None,
    attribution_payload: dict[str, Any] | None = None,
) -> OutwardResult:
    if explanation is None:
        degraded_reason_codes = ("no_explanation_payload",)
        outward_status = "NO_EVIDENCE"
        capability_boundary_notice = (
            "This case is outside the current explanation capability boundary."
        )
        outward_notes = ("No explanation payload was available.",)
        explanation_payload = None
        manual_review = True
    else:
        degraded_reason_codes = tuple(explanation.degraded_reason_codes)
        outward_status = _map_outward_status(explanation)
        capability_boundary_notice = _build_capability_boundary_notice(
            outward_status=outward_status,
            degraded_reason_codes=degraded_reason_codes,
            explanation_notice=explanation.capability_boundary_notice,
        )
        outward_notes = _build_outward_notes(
            outward_status=outward_status,
            explanation=explanation,
        )
        explanation_payload = explanation_draft_to_dict(explanation)
        manual_review = explanation.manual_review

    # DEC-199 H: macro-shock flag is additive — force manual_review, add a
    # degraded reason code, and append an outward note so the analyst sees
    # the caveat alongside branch-specific notes.
    if attribution_payload and attribution_payload.get("macro_shock_suspected"):
        manual_review = True
        if "macro_shock_suspected" not in degraded_reason_codes:
            degraded_reason_codes = degraded_reason_codes + ("macro_shock_suspected",)
        macro_note = (
            "Macro shock period suspected (VIX elevated vs training regime) — "
            "attribution confidence is reduced; cross-asset correlation likely "
            "exceeds what training-window betas can capture."
        )
        if macro_note not in outward_notes:
            outward_notes = outward_notes + (macro_note,)

    return OutwardResult(
        outward_id=f"outward-{uuid4().hex[:10]}",
        episode_id=episode_id,
        ticker=ticker,
        outward_status=outward_status,
        manual_review=manual_review,
        capability_boundary_notice=capability_boundary_notice,
        degraded_reason_codes=degraded_reason_codes,
        outward_notes=outward_notes,
        explanation=explanation_payload,
    )


def outward_result_to_dict(result: OutwardResult) -> dict[str, Any]:
    return {
        "outward_id": result.outward_id,
        "episode_id": result.episode_id,
        "ticker": result.ticker,
        "outward_status": result.outward_status,
        "manual_review": result.manual_review,
        "capability_boundary_notice": result.capability_boundary_notice,
        "degraded_reason_codes": list(result.degraded_reason_codes),
        "outward_notes": list(result.outward_notes),
        "explanation": result.explanation,
    }


def _map_outward_status(explanation: ExplanationDraft) -> str:
    if not explanation.top_bundle_ids:
        return "NO_EVIDENCE"
    if "no_eligible_bundle" in explanation.degraded_reason_codes:
        return "NO_EVIDENCE"
    if (
        explanation.explanation_confidence in {"high", "medium"}
        and not explanation.manual_review
        and not explanation.degraded_reason_codes
    ):
        return "SUPPORTED"
    return "LIMITED_EVIDENCE"


def _build_capability_boundary_notice(
    *,
    outward_status: str,
    degraded_reason_codes: tuple[str, ...],
    explanation_notice: str | None,
) -> str | None:
    if outward_status == "NO_EVIDENCE":
        if "no_explanation_payload" in degraded_reason_codes:
            return "This case is outside the current explanation capability boundary."
        if "no_eligible_bundle" in degraded_reason_codes:
            return "No explanation-eligible evidence bundle was available."
        return "This case is outside the current explanation capability boundary."

    if "only_weak_support" in degraded_reason_codes:
        return "Only weak supporting evidence was available for this explanation."
    if "mixed_attribution" in degraded_reason_codes:
        return (
            "Attribution signals were mixed, so the explanation should be reviewed manually."
        )
    if "limited_supporting_context" in degraded_reason_codes:
        return "Supporting evidence context was limited for this explanation."
    return explanation_notice


def _build_outward_notes(
    *,
    outward_status: str,
    explanation: ExplanationDraft,
) -> tuple[str, ...]:
    notes: list[str] = []
    if explanation.top_bundle_ids:
        notes.append("A primary evidence-backed explanation is available.")
    if explanation.supporting_bundle_count == 0:
        notes.append("Supporting context is limited.")
    if explanation.manual_review:
        notes.append("Manual review is recommended.")
    if outward_status == "LIMITED_EVIDENCE" and not explanation.manual_review:
        notes.append("The explanation is available but should be read with caution.")
    if outward_status == "SUPPORTED":
        notes.append("The explanation is within the current bounded system scope.")
    return tuple(notes)
