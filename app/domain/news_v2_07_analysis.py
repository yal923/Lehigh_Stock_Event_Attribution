from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from app.domain.news_v2_01_retrieval import RankedArticle
from app.domain.news_v2_04_bundles import EvidenceBundle
from app.domain.news_v2_05_explanations import ExplanationDraft, KeyEventSummary
from app.domain.news_v2_06_outward import OutwardResult


@dataclass(frozen=True)
class AnalysisV2Payload:
    outward: dict[str, Any]
    explanation: dict[str, Any]
    attribution_summary: dict[str, Any]
    evidence_bundles: tuple[dict[str, Any], ...]
    supporting_articles: tuple[dict[str, Any], ...]


def build_analysis_v2_payload(
    *,
    outward_result: OutwardResult,
    explanation_draft: ExplanationDraft,
    attribution_payload: dict[str, Any],
    evidence_bundles: Sequence[EvidenceBundle],
    ranked_articles: Sequence[RankedArticle],
) -> AnalysisV2Payload:
    return AnalysisV2Payload(
        outward=_build_outward_block(outward_result),
        explanation=_build_explanation_block(explanation_draft),
        attribution_summary=_build_attribution_summary_block(
            explanation_draft=explanation_draft,
            attribution_payload=attribution_payload,
        ),
        evidence_bundles=tuple(
            _build_evidence_bundle_row(bundle) for bundle in evidence_bundles
        ),
        supporting_articles=tuple(
            _build_supporting_article_row(article, evidence_bundles)
            for article in _supporting_articles_only(ranked_articles, evidence_bundles)
        ),
    )


def analysis_v2_payload_to_dict(payload: AnalysisV2Payload) -> dict[str, Any]:
    return {
        "outward": payload.outward,
        "explanation": payload.explanation,
        "attribution_summary": payload.attribution_summary,
        "evidence_bundles": list(payload.evidence_bundles),
        "supporting_articles": list(payload.supporting_articles),
    }


def _build_outward_block(outward_result: OutwardResult) -> dict[str, Any]:
    return {
        "outward_status": outward_result.outward_status,
        "manual_review": outward_result.manual_review,
        "capability_boundary_notice": outward_result.capability_boundary_notice,
        "degraded_reason_codes": list(outward_result.degraded_reason_codes),
        "outward_notes": list(outward_result.outward_notes),
    }


def _build_explanation_block(explanation_draft: ExplanationDraft) -> dict[str, Any]:
    return {
        "driver_hypothesis": explanation_draft.driver_hypothesis,
        "headline_explanation": explanation_draft.headline_explanation,
        "key_event_summaries": [
            _key_event_summary_to_dict(summary)
            for summary in explanation_draft.key_event_summaries
        ],
        "explanation_confidence": explanation_draft.explanation_confidence,
        "manual_review": explanation_draft.manual_review,
        "degraded_reason_codes": list(explanation_draft.degraded_reason_codes),
        "llm_enrichment_text": explanation_draft.llm_enrichment_text,
    }


def _build_attribution_summary_block(
    *,
    explanation_draft: ExplanationDraft,
    attribution_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "driver_hypothesis": explanation_draft.driver_hypothesis,
        "firm_ratio": float(attribution_payload.get("firm_ratio") or 0.0),
        "industry_ratio": float(attribution_payload.get("industry_ratio") or 0.0),
        "market_ratio": float(attribution_payload.get("market_ratio") or 0.0),
        "quality_status": attribution_payload.get("quality_status"),
        "reason": attribution_payload.get("reason"),
        "reason_code": attribution_payload.get("reason_code"),
        "adj_r2": attribution_payload.get("adj_r2"),
        "beta_mkt_t_stat": attribution_payload.get("beta_mkt_t_stat"),
        "beta_ind_t_stat": attribution_payload.get("beta_ind_t_stat"),
        "f_stat": attribution_payload.get("f_stat"),
        "f_pvalue": attribution_payload.get("f_pvalue"),
        "fit_quality_note": attribution_payload.get("fit_quality_note"),
        "market_share": attribution_payload.get("market_share"),
        "industry_share": attribution_payload.get("industry_share"),
        "firm_share": attribution_payload.get("firm_share"),
        "alpha_share": attribution_payload.get("alpha_share"),
        "near_zero_episode_return": bool(
            attribution_payload.get("near_zero_episode_return")
        ),
        "episode_return_total": attribution_payload.get("episode_return_total"),
        "industry_proxy_ticker": attribution_payload.get("industry_proxy_ticker"),
        "industry_proxy_source": attribution_payload.get("industry_proxy_source"),
        "vix_estimation_p50": attribution_payload.get("vix_estimation_p50"),
        "vix_estimation_p95": attribution_payload.get("vix_estimation_p95"),
        "vix_episode_max": attribution_payload.get("vix_episode_max"),
        "vix_episode_avg": attribution_payload.get("vix_episode_avg"),
        "macro_shock_suspected": bool(
            attribution_payload.get("macro_shock_suspected")
        ),
    }


def _build_evidence_bundle_row(bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        "canonical_event_type": bundle.canonical_event_type,
        "bundle_status": bundle.bundle_status,
        "bundle_score": bundle.bundle_score,
        "eligible_for_explanation": bundle.eligible_for_explanation,
        "supporting_event_ids": list(bundle.supporting_event_ids),
        "supporting_claim_ids": list(bundle.supporting_claim_ids),
        "supporting_article_ids": list(bundle.supporting_article_ids),
        "score_breakdown": dict(bundle.bundle_score_breakdown),
    }


def _supporting_articles_only(
    ranked_articles: Sequence[RankedArticle],
    evidence_bundles: Sequence[EvidenceBundle],
) -> list[RankedArticle]:
    supporting_article_ids = {
        article_id
        for bundle in evidence_bundles
        for article_id in bundle.supporting_article_ids
    }
    return [
        article
        for article in ranked_articles
        if article.article_id in supporting_article_ids
    ]


def _build_supporting_article_row(
    article: RankedArticle,
    evidence_bundles: Sequence[EvidenceBundle],
) -> dict[str, Any]:
    supporting_bundle_ids = [
        bundle.bundle_id
        for bundle in evidence_bundles
        if article.article_id in bundle.supporting_article_ids
    ]
    return {
        "article_id": article.article_id,
        "provider": article.provider,
        "publisher": article.publisher,
        "scope_tag": article.scope_tag,
        "headline": article.headline,
        "summary_or_snippet": article.summary_or_snippet,
        "url": article.url,
        "published_at": article.published_at,
        "rank_score": article.rank_score,
        "rank_position": article.rank_position,
        "supporting_bundle_ids": supporting_bundle_ids,
    }


def _key_event_summary_to_dict(summary: KeyEventSummary) -> dict[str, Any]:
    return asdict(summary)
