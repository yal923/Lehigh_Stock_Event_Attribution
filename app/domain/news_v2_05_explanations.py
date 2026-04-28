from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence
from uuid import uuid4

from app.domain.news_v2_01_retrieval import EpisodeContext
from app.domain.news_v2_04_bundles import EvidenceBundle
from app.infrastructure.llm import get_openai_client


DEFAULT_EXPLANATION_MODEL = "gpt-4.1-2025-04-14"


@dataclass(frozen=True)
class KeyEventSummary:
    role: str
    bundle_id: str
    event_type: str
    summary: str


@dataclass(frozen=True)
class ExplanationDraft:
    explanation_id: str
    episode_id: int
    ticker: str
    driver_hypothesis: str
    headline_explanation: str
    top_bundle_ids: tuple[str, ...]
    key_event_summaries: tuple[KeyEventSummary, ...]
    explanation_confidence: str
    manual_review: bool
    capability_boundary_notice: str | None
    degraded_reason_codes: tuple[str, ...]
    synthesis_method: str
    supporting_bundle_count: int
    secondary_hypotheses: tuple[str, ...] = ()
    llm_enrichment_text: str | None = None


def build_explanation_draft(
    *,
    episode_id: int,
    episode_context: EpisodeContext,
    attribution_payload: dict[str, Any],
    bundles: Sequence[EvidenceBundle],
    analysis_mode: str | None = None,
    primary_mode: str | None = None,
    model_name: str = DEFAULT_EXPLANATION_MODEL,
) -> ExplanationDraft:
    eligible_bundles = [bundle for bundle in bundles if bundle.eligible_for_explanation]
    top_bundles = eligible_bundles[:2]
    driver_hypothesis, secondary_hypotheses = _driver_hypothesis(
        attribution_payload=attribution_payload,
        analysis_mode=analysis_mode,
        primary_mode=primary_mode,
    )
    headline_explanation = _build_headline_explanation(
        driver_hypothesis=driver_hypothesis,
        top_bundles=top_bundles,
    )
    key_event_summaries = _build_key_event_summaries(top_bundles)
    top_bundle_ids = tuple(bundle.bundle_id for bundle in top_bundles)
    explanation_confidence = _explanation_confidence(
        driver_hypothesis=driver_hypothesis,
        eligible_bundles=eligible_bundles,
    )
    manual_review = _manual_review(
        driver_hypothesis=driver_hypothesis,
        eligible_bundles=eligible_bundles,
    )
    degraded_reason_codes = _degraded_reason_codes(
        driver_hypothesis=driver_hypothesis,
        eligible_bundles=eligible_bundles,
    )
    capability_boundary_notice = _capability_boundary_notice(degraded_reason_codes)
    llm_enrichment_text = _build_llm_enrichment_text(
        episode_context=episode_context,
        attribution_payload=attribution_payload,
        driver_hypothesis=driver_hypothesis,
        analysis_mode=analysis_mode,
        headline_explanation=headline_explanation,
        key_event_summaries=key_event_summaries,
        top_bundles=top_bundles,
        degraded_reason_codes=degraded_reason_codes,
        model_name=model_name,
    )
    return ExplanationDraft(
        explanation_id=f"explanation-{uuid4().hex[:10]}",
        episode_id=episode_id,
        ticker=episode_context.ticker,
        driver_hypothesis=driver_hypothesis,
        headline_explanation=headline_explanation,
        top_bundle_ids=top_bundle_ids,
        key_event_summaries=key_event_summaries,
        explanation_confidence=explanation_confidence,
        manual_review=manual_review,
        capability_boundary_notice=capability_boundary_notice,
        degraded_reason_codes=degraded_reason_codes,
        synthesis_method="rule-first-bounded+llm-enrichment",
        supporting_bundle_count=max(len(top_bundles) - 1, 0),
        secondary_hypotheses=secondary_hypotheses,
        llm_enrichment_text=llm_enrichment_text,
    )


def explanation_draft_to_dict(draft: ExplanationDraft) -> dict[str, Any]:
    data = asdict(draft)
    data["top_bundle_ids"] = list(draft.top_bundle_ids)
    data["key_event_summaries"] = [
        asdict(summary) for summary in draft.key_event_summaries
    ]
    data["degraded_reason_codes"] = list(draft.degraded_reason_codes)
    data["secondary_hypotheses"] = list(draft.secondary_hypotheses)
    return data


def _driver_hypothesis(
    *,
    attribution_payload: dict[str, Any],
    analysis_mode: str | None,
    primary_mode: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    if analysis_mode in {"market", "industry", "firm"}:
        return f"{analysis_mode}_branch", ()
    # OOS means the regression never ran — there are no valid attribution ratios.
    # Do not derive a driver from ratio arithmetic; the entire attribution system
    # is outside scope for this episode.
    if attribution_payload.get("quality_status") == "OOS":
        return "oos", ()
    # When routing already established dominance, trust the router's decision rather
    # than re-deriving the dominant mode from ratios (DEC-189). The routing condition
    # (top1 >= 0.70, gap >= 0.15) is strictly stronger than the old Step-5 gap check
    # (gap >= 0.10), so re-deriving would always yield the same result — but doing it
    # explicitly here is cleaner and removes the dead "mixed" branch from Path B.
    if primary_mode in {"firm", "industry", "market"}:
        ratios = {
            "firm": float(attribution_payload.get("firm_ratio") or 0.0),
            "industry": float(attribution_payload.get("industry_ratio") or 0.0),
            "market": float(attribution_payload.get("market_ratio") or 0.0),
        }
        ranked = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        secondary = next(label for label, _ in ranked if label != primary_mode)
        return f"{primary_mode}_dominant", (f"{secondary}_dominant",)
    return _driver_hypothesis_from_attribution(attribution_payload)


def _driver_hypothesis_from_attribution(
    attribution_payload: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    ratios = [
        ("firm_dominant", float(attribution_payload.get("firm_ratio") or 0.0)),
        ("industry_dominant", float(attribution_payload.get("industry_ratio") or 0.0)),
        ("market_dominant", float(attribution_payload.get("market_ratio") or 0.0)),
    ]
    ranked = sorted(ratios, key=lambda item: item[1], reverse=True)
    top1_label, top1_value = ranked[0]
    top2_label, top2_value = ranked[1]
    if top1_value - top2_value < 0.10:
        return "mixed", (top1_label, top2_label)
    return top1_label, (top2_label,)


def _build_headline_explanation(
    *,
    driver_hypothesis: str,
    top_bundles: Sequence[EvidenceBundle],
) -> str:
    # DEC-187: alignment check removed — retrieval scope already conditions the
    # news pool on the attributed driver, so post-hoc direction comparison is
    # redundant. Headlines are purely bundle-quality-driven.
    if driver_hypothesis == "oos":
        if not top_bundles:
            return (
                "Attribution data was unavailable for this episode. "
                "No explanation-eligible news evidence was found either."
            )
        return (
            "Attribution data was unavailable for this episode. "
            f"Available news evidence points to {_bundle_phrase(top_bundles[0])}."
        )
    if not top_bundles:
        if _is_branch_driver(driver_hypothesis):
            return (
                f"This {_driver_phrase(driver_hypothesis)} explanation branch "
                "did not retain an explanation-eligible evidence bundle."
            )
        return (
            f"This episode appears {_driver_phrase(driver_hypothesis)}, "
            "but no explanation-eligible evidence bundle was available."
        )
    primary = top_bundles[0]
    if _is_branch_driver(driver_hypothesis):
        return (
            f"Within the {_driver_phrase(driver_hypothesis)} explanation branch, "
            f"the strongest retained evidence points to {_bundle_phrase(primary)}."
        )
    if driver_hypothesis == "mixed":
        return (
            "This episode appears driven by a mixed set of factors, "
            f"with the strongest evidence pointing to {_bundle_phrase(primary)}."
        )
    return (
        f"This episode appears {_driver_phrase(driver_hypothesis)}, "
        f"with the strongest evidence pointing to {_bundle_phrase(primary)}."
    )


def _build_key_event_summaries(
    top_bundles: Sequence[EvidenceBundle],
) -> tuple[KeyEventSummary, ...]:
    summaries: list[KeyEventSummary] = []
    for index, bundle in enumerate(top_bundles):
        role = "primary" if index == 0 else "supporting"
        if role == "primary":
            summary = (
                f"{_bundle_phrase(bundle, sentence_case=True)} is the primary "
                f"evidence-backed driver, supported by {bundle.corroboration_count} "
                f"claim(s) across {bundle.source_diversity_count} publisher(s) "
                f"(bundle status: {bundle.bundle_status})."
            )
        else:
            summary = (
                f"{_bundle_phrase(bundle, sentence_case=True)} provides supporting context, "
                f"supported by {bundle.corroboration_count} claim(s) across "
                f"{bundle.source_diversity_count} publisher(s) "
                f"(bundle status: {bundle.bundle_status})."
            )
        summaries.append(
            KeyEventSummary(
                role=role,
                bundle_id=bundle.bundle_id,
                event_type=bundle.canonical_event_type,
                summary=summary,
            )
        )
    return tuple(summaries)


def _explanation_confidence(
    *,
    driver_hypothesis: str,
    eligible_bundles: Sequence[EvidenceBundle],
) -> str:
    if driver_hypothesis == "oos":
        return "low"
    if not eligible_bundles:
        return "low"
    top_status = eligible_bundles[0].bundle_status
    if top_status == "strong" and driver_hypothesis != "mixed":
        return "high"
    if top_status == "moderate" or (
        top_status == "strong" and driver_hypothesis == "mixed"
    ):
        return "medium"
    return "low"


def _manual_review(
    *,
    driver_hypothesis: str,
    eligible_bundles: Sequence[EvidenceBundle],
) -> bool:
    if driver_hypothesis == "oos":
        return True
    if not eligible_bundles:
        return True
    if eligible_bundles[0].bundle_status == "weak":
        return True
    if driver_hypothesis == "mixed":
        return True
    return False


def _degraded_reason_codes(
    *,
    driver_hypothesis: str,
    eligible_bundles: Sequence[EvidenceBundle],
) -> tuple[str, ...]:
    codes: list[str] = []
    if not eligible_bundles:
        codes.append("no_eligible_bundle")
        if driver_hypothesis == "oos":
            codes.append("attribution_data_unavailable")
        return tuple(codes)
    if eligible_bundles[0].bundle_status == "weak":
        codes.append("only_weak_support")
    if driver_hypothesis == "mixed":
        codes.append("mixed_attribution")
    elif driver_hypothesis == "oos":
        codes.append("attribution_data_unavailable")
    if len(eligible_bundles) < 2:
        codes.append("limited_supporting_context")
    return tuple(codes)


def _capability_boundary_notice(
    degraded_reason_codes: Sequence[str],
) -> str | None:
    if not degraded_reason_codes:
        return None
    return (
        "This explanation is bounded by the currently eligible evidence bundles "
        "and should be read together with the explicit degraded reason codes."
    )


def _build_llm_enrichment_text(
    *,
    episode_context: EpisodeContext,
    attribution_payload: dict[str, Any],
    driver_hypothesis: str,
    analysis_mode: str | None,
    headline_explanation: str,
    key_event_summaries: Sequence[KeyEventSummary],
    top_bundles: Sequence[EvidenceBundle],
    degraded_reason_codes: Sequence[str],
    model_name: str,
) -> str | None:
    if not top_bundles:
        return None

    client = get_openai_client()
    prompt = _build_enrichment_prompt(
        episode_context=episode_context,
        attribution_payload=attribution_payload,
        driver_hypothesis=driver_hypothesis,
        analysis_mode=analysis_mode,
        headline_explanation=headline_explanation,
        key_event_summaries=key_event_summaries,
        top_bundles=top_bundles,
        degraded_reason_codes=degraded_reason_codes,
    )
    response = client.responses.create(
        model=model_name,
        input=prompt,
        max_output_tokens=500,
    )
    text = (response.output_text or "").strip()
    return text or None


def _build_enrichment_prompt(
    *,
    episode_context: EpisodeContext,
    attribution_payload: dict[str, Any],
    driver_hypothesis: str,
    analysis_mode: str | None,
    headline_explanation: str,
    key_event_summaries: Sequence[KeyEventSummary],
    top_bundles: Sequence[EvidenceBundle],
    degraded_reason_codes: Sequence[str],
) -> str:
    bundle_lines = []
    for bundle in top_bundles[:2]:
        bundle_lines.append(
            (
                f"- bundle_id={bundle.bundle_id}; "
                f"event_type={bundle.canonical_event_type}; "
                f"label={_bundle_label(bundle)}; "
                f"status={bundle.bundle_status}; "
                f"score={bundle.bundle_score}; "
                f"claims={bundle.corroboration_count}; "
                f"publishers={bundle.source_diversity_count}"
            )
        )

    summary_lines = [
        f"- role={summary.role}; bundle_id={summary.bundle_id}; summary={summary.summary}"
        for summary in key_event_summaries
    ]

    return f"""
You are writing a short evidence-grounded enrichment note for an analyst-facing financial explanation.

Rules:
- Stay subordinate to the rule-first draft.
- Use only the provided attribution context, bundle facts, and event summaries.
- Do not introduce new facts, causes, or speculation.
- Write 2 short sentences maximum.
- Write in concise analyst-note style rather than taxonomy-heavy wording.
- Prefer natural phrases like "acquisition activity" or "regulatory approval" over repeating raw event-type labels.
- If support is limited or mixed, preserve that caution in the wording.
- Do not restate every input line; synthesize only the most relevant primary and optional supporting context.

Episode context:
- ticker: {episode_context.ticker}
- company_name: {episode_context.company_name}
- episode_window: {episode_context.episode_start}..{episode_context.episode_end}
- peak_date: {episode_context.peak_date}

Attribution payload:
{attribution_payload}

Driver hypothesis:
- {driver_hypothesis}

Requested analysis mode:
- {analysis_mode or 'attribution-led-default'}

Rule-first headline:
- {headline_explanation}

Key event summaries:
{chr(10).join(summary_lines)}

Top eligible bundles:
{chr(10).join(bundle_lines)}

Degraded reason codes:
- {list(degraded_reason_codes)}
""".strip()


def _bundle_label(bundle: EvidenceBundle) -> str:
    return bundle.canonical_event_type.replace("_", " ").title()


def _driver_phrase(driver_hypothesis: str) -> str:
    mapping = {
        "firm_dominant": "primarily company-specific",
        "industry_dominant": "primarily industry-driven",
        "market_dominant": "primarily market-driven",
        "firm_branch": "company-focused",
        "industry_branch": "industry-focused",
        "market_branch": "market-focused",
        "mixed": "driven by a mixed set of factors",
        "oos": "attribution data unavailable",
    }
    return mapping.get(driver_hypothesis, driver_hypothesis.replace("_", " "))


def _is_branch_driver(driver_hypothesis: str) -> bool:
    return driver_hypothesis in {"market_branch", "industry_branch", "firm_branch"}


def _bundle_phrase(bundle: EvidenceBundle, *, sentence_case: bool = False) -> str:
    mapping = {
        "mna_or_strategic_transaction": "M&A activity",
        "regulatory_approval": "regulatory approval",
        "regulatory_or_legal_action": "regulatory or legal action",
        "earnings_result": "earnings results",
        "guidance_update": "guidance changes",
        "capital_return_action": "capital return actions",
        "capital_raise_or_balance_sheet_action": "balance-sheet or financing actions",
        "management_or_governance_change": "management or governance changes",
        "product_or_customer_event": "product or customer developments",
        "operations_or_security_incident": "operational or security incidents",
        "analyst_or_investor_view_change": "analyst or investor view changes",
        "macro_or_sector_shock": "macro or sector developments",
        "other": "other developments",
        "unknown": "uncertain developments",
    }
    phrase = mapping.get(bundle.canonical_event_type, _bundle_label(bundle).lower())
    if sentence_case:
        return phrase[0].upper() + phrase[1:] if phrase else phrase
    return phrase
