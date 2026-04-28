from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.cache import load_episode_cache, save_episode_cache
from app.domain.attribution import run_episode_window_attribution
from app.domain.attribution_router import (
    build_router_input_from_attribution_payload,
    route_attribution,
)
from app.domain.branch_analysis import (
    BranchAnalysisArtifacts,
    build_multi_branch_analysis_plan,
)
from app.domain.branch_rerank import (
    DEFAULT_CROSS_ENCODER_MODEL,
    BranchRerankInput,
    BranchRerankResult,
    run_branch_rerank,
    score_branch_articles_with_default_cross_encoder,
)
from app.domain.news_v2_01_retrieval import (
    build_episode_context_from_episode,
    build_provider_context,
    fetch_candidate_articles,
    run_retrieval_reranking,
    score_candidate_articles_with_default_cross_encoder,
)
from app.domain.news_v2_04_bundles import build_evidence_bundles
from app.domain.news_v2_02_claims import extract_claims_from_ranked_articles
from app.domain.news_v2_03_events import normalize_claims_to_events
from app.domain.news_v2_05_explanations import build_explanation_draft
from app.domain.news_v2_06_outward import build_outward_result
from app.domain.news_v2_07_analysis import (
    analysis_v2_payload_to_dict,
    build_analysis_v2_payload,
)
from app.domain.shared_raw_pool import build_shared_raw_pool
from app.infrastructure.resources import get_cross_encoder_model
from app.models import Episode


def build_analysis_v2_artifacts(
    episode: Episode,
    *,
    top_k: int = 10,
    attribution_hint: str | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    cache: dict[str, Any] = {} if force_recompute else load_episode_cache(episode.id)
    attribution_payload = get_or_compute_attribution_bundle(
        episode,
        cache=cache,
        force_recompute=force_recompute,
    )

    router_result = route_attribution(
        build_router_input_from_attribution_payload(attribution_payload)
    )
    multi_branch_plan = build_multi_branch_analysis_plan(router_result)
    # Retrieval scope is determined directly from routing decision (not re-derived from ratios).
    # dominant → fetch only the primary branch direction.
    # mixed → fetch all three scopes (market / industry / company).
    # degraded → fetch all three scopes as a best-effort fallback.
    _retrieval_mode = (
        router_result.primary_mode
        if router_result.routing_mode == "dominant" and router_result.primary_mode
        else router_result.routing_mode
    )
    episode_context = build_episode_context_from_episode(
        episode,
        attribution_hint=attribution_hint,
        attribution_payload=attribution_payload,
        retrieval_mode=_retrieval_mode,
    )
    provider_context = build_provider_context()
    branch_artifacts: BranchAnalysisArtifacts | None = None
    branch_analyses: list[dict[str, Any]] = []

    if router_result.routing_mode == "dominant" and router_result.primary_mode:
        shared_raw_pool = _build_shared_raw_pool_for_episode(
            episode=episode,
            episode_context=episode_context,
            provider_context=provider_context,
        )
        branch_artifacts = _build_branch_artifacts(
            episode=episode,
            episode_context=episode_context,
            attribution_payload=attribution_payload,
            router_result=router_result,
            shared_raw_pool=shared_raw_pool,
            branch_mode=router_result.primary_mode,
            branch_explanation_mode=None,
            primary_mode=router_result.primary_mode,
            top_k=top_k,
        )
        ranked_articles = list(branch_artifacts.rerank_result.ranked_articles)
        claims = list(branch_artifacts.claims)
        events = list(branch_artifacts.events)
        bundles = list(branch_artifacts.bundles)
        explanation = branch_artifacts.explanation
    elif router_result.routing_mode == "mixed":
        shared_raw_pool = _build_shared_raw_pool_for_episode(
            episode=episode,
            episode_context=episode_context,
            provider_context=provider_context,
        )
        branch_analyses = _build_mixed_branch_analyses(
            episode=episode,
            episode_context=episode_context,
            attribution_payload=attribution_payload,
            router_result=router_result,
            shared_raw_pool=shared_raw_pool,
            top_k=top_k,
        )
        ranked_articles = []
        claims = []
        events = []
        bundles = []
        explanation = None
        outward = None
        analysis_payload = None
        analysis_dict = _build_mixed_analysis_payload(
            attribution_payload=attribution_payload,
            branch_analyses=branch_analyses,
            multi_branch_plan=multi_branch_plan,
        )
        save_episode_cache(episode.id, **cache)
        return {
            "cache": cache,
            "attribution_payload": attribution_payload,
            "ranked_articles": ranked_articles,
            "claims": claims,
            "events": events,
            "bundles": bundles,
            "explanation": explanation,
            "outward": outward,
            "analysis_payload": analysis_payload,
            "analysis_dict": analysis_dict,
            "router_result": router_result,
            "multi_branch_plan": multi_branch_plan,
            "branch_artifacts": branch_artifacts,
            "branch_analyses": branch_analyses,
        }
    else:
        ranked_articles = run_retrieval_reranking(
            episode_context=episode_context,
            provider_context=provider_context,
            scorer=score_candidate_articles_with_default_cross_encoder,
            rank_method="cross-encoder/ms-marco-MiniLM-L6-v2",
        )[:top_k]

        claims = extract_claims_from_ranked_articles(ranked_articles)
        events = normalize_claims_to_events(claims)
        bundles = build_evidence_bundles(
            events=events,
            claims=claims,
            ranked_articles=ranked_articles,
            episode_context=episode_context,
        )
        explanation = build_explanation_draft(
            episode_id=episode.id,
            episode_context=episode_context,
            attribution_payload=attribution_payload,
            bundles=bundles,
        )

    outward = build_outward_result(
        episode_id=episode.id,
        ticker=episode_context.ticker,
        explanation=explanation,
        attribution_payload=attribution_payload,
    )
    analysis_payload = build_analysis_v2_payload(
        outward_result=outward,
        explanation_draft=explanation,
        attribution_payload=attribution_payload,
        evidence_bundles=bundles,
        ranked_articles=ranked_articles,
    )

    analysis_dict = analysis_v2_payload_to_dict(analysis_payload)
    analysis_dict["routing_mode"] = router_result.routing_mode
    save_episode_cache(episode.id, **cache)
    return {
        "cache": cache,
        "attribution_payload": attribution_payload,
        "ranked_articles": ranked_articles,
        "claims": claims,
        "events": events,
        "bundles": bundles,
        "explanation": explanation,
        "outward": outward,
        "analysis_payload": analysis_payload,
        "analysis_dict": analysis_dict,
        "router_result": router_result,
        "multi_branch_plan": multi_branch_plan,
        "branch_artifacts": branch_artifacts,
        "branch_analyses": branch_analyses,
    }


def get_or_compute_attribution_bundle(
    episode: Episode,
    *,
    cache: dict[str, Any] | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    working_cache: dict[str, Any]
    if cache is None:
        working_cache = {} if force_recompute else load_episode_cache(episode.id)
    else:
        working_cache = cache

    if not force_recompute and "attribution_bundle" in working_cache:
        return working_cache["attribution_bundle"]

    attr = run_episode_window_attribution(episode)
    attribution_bundle = _build_attribution_bundle(attr)
    working_cache["attribution_bundle"] = attribution_bundle
    return attribution_bundle


def run_episode_pipeline(
    episode: Episode,
    force_recompute: bool = False,
) -> dict[str, Any]:
    cache: dict[str, Any] = {} if force_recompute else load_episode_cache(episode.id)
    if not force_recompute and "analysis_v2_bundle" in cache:
        return cache["analysis_v2_bundle"]

    artifacts = build_analysis_v2_artifacts(
        episode,
        force_recompute=force_recompute,
    )
    analysis = artifacts["analysis_dict"]
    decision = _build_v2_decision_payload(analysis)
    multi_branch_plan = artifacts["multi_branch_plan"]
    result = {
        "episode_id": episode.id,
        "analysis": analysis,
        "attribution": analysis.get("attribution_summary") or {},
        "decision": decision,
        "branch_plan": {
            "routing_mode": multi_branch_plan.routing_mode,
            "primary_mode": multi_branch_plan.primary_mode,
            "default_modes": list(multi_branch_plan.default_modes),
            "lazy_modes": list(multi_branch_plan.lazy_modes),
        },
    }

    cache = artifacts["cache"]
    cache["analysis_v2_bundle"] = result
    save_episode_cache(episode.id, **cache)
    return result


def run_episode_secondary_branch_analysis(
    episode: Episode,
    *,
    branch_mode: str,
    force_recompute: bool = False,
) -> dict[str, Any]:
    cache: dict[str, Any] = {} if force_recompute else load_episode_cache(episode.id)
    branch_cache = dict(cache.get("analysis_v2_secondary_branches") or {})
    cache_key = f"{branch_mode}_branch"
    if not force_recompute and cache_key in branch_cache:
        return branch_cache[cache_key]

    attribution_payload = get_or_compute_attribution_bundle(
        episode,
        cache=cache,
        force_recompute=force_recompute,
    )
    router_result = route_attribution(
        build_router_input_from_attribution_payload(attribution_payload)
    )
    multi_branch_plan = build_multi_branch_analysis_plan(router_result)
    # Secondary branch fetches articles scoped to the requested branch direction,
    # not the primary branch's narrow scope.
    episode_context = build_episode_context_from_episode(
        episode,
        attribution_payload=attribution_payload,
        retrieval_mode=branch_mode,
    )

    if router_result.routing_mode != "dominant":
        raise ValueError("Secondary branch analysis is only available for dominant episodes.")
    if branch_mode not in multi_branch_plan.lazy_modes:
        raise ValueError(f"Branch mode {branch_mode!r} is not available for lazy analysis.")

    branch_artifacts = _build_branch_artifacts(
        episode=episode,
        episode_context=episode_context,
        attribution_payload=attribution_payload,
        router_result=router_result,
        shared_raw_pool=_build_shared_raw_pool_for_episode(
            episode=episode,
            episode_context=episode_context,
            provider_context=build_provider_context(),
        ),
        branch_mode=branch_mode,
        branch_explanation_mode=branch_mode,
        top_k=10,
    )
    outward = build_outward_result(
        episode_id=episode.id,
        ticker=episode_context.ticker,
        explanation=branch_artifacts.explanation,
        attribution_payload=attribution_payload,
    )
    analysis_payload = build_analysis_v2_payload(
        outward_result=outward,
        explanation_draft=branch_artifacts.explanation,
        attribution_payload=attribution_payload,
        evidence_bundles=branch_artifacts.bundles,
        ranked_articles=branch_artifacts.rerank_result.ranked_articles,
    )
    analysis = analysis_v2_payload_to_dict(analysis_payload)
    result = {
        "episode_id": episode.id,
        "branch_mode": branch_mode,
        "analysis": analysis,
        "attribution": analysis.get("attribution_summary") or {},
        "decision": _build_v2_decision_payload(analysis),
        "multi_branch_plan": {
            "routing_mode": multi_branch_plan.routing_mode,
            "primary_mode": multi_branch_plan.primary_mode,
            "default_modes": list(multi_branch_plan.default_modes),
            "lazy_modes": list(multi_branch_plan.lazy_modes),
        },
    }
    branch_cache[cache_key] = result
    cache["analysis_v2_secondary_branches"] = branch_cache
    save_episode_cache(episode.id, **cache)
    return result


def _build_v2_decision_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("routing_mode") == "mixed":
        outward = analysis.get("aggregate_outward") or {}
        explanation = analysis.get("aggregate_explanation") or {}
    else:
        outward = analysis.get("outward") or {}
        explanation = analysis.get("explanation") or {}
    return {
        "outward_status": outward.get("outward_status"),
        "manual_review": outward.get("manual_review"),
        "degraded_reason_codes": outward.get("degraded_reason_codes") or [],
        "capability_boundary_notice": outward.get("capability_boundary_notice"),
        "outward_notes": outward.get("outward_notes") or [],
        "driver_hypothesis": explanation.get("driver_hypothesis"),
        "headline_explanation": explanation.get("headline_explanation"),
        "explanation_confidence": explanation.get("explanation_confidence"),
    }


def _build_branch_artifacts(
    *,
    episode: Episode,
    episode_context: Any,
    attribution_payload: dict[str, Any],
    router_result: Any,
    shared_raw_pool: Any,
    branch_mode: str,
    branch_explanation_mode: str | None,
    top_k: int,
    primary_mode: str | None = None,
) -> BranchAnalysisArtifacts:
    rerank_result = run_branch_rerank(
        BranchRerankInput(
            episode_id=episode.id,
            branch_mode=branch_mode,
            shared_raw_pool=shared_raw_pool,
            episode_context=episode_context,
            router_result=router_result,
        ),
        scorer=score_branch_articles_with_default_cross_encoder,
        rank_method="cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    ranked_articles = tuple(rerank_result.ranked_articles[:top_k])
    selected_rerank_result = BranchRerankResult(
        branch_mode=rerank_result.branch_mode,
        rerank_query=rerank_result.rerank_query,
        ranked_articles=ranked_articles,
    )
    claims = tuple(extract_claims_from_ranked_articles(ranked_articles))
    events = tuple(normalize_claims_to_events(claims))
    bundles = tuple(
        build_evidence_bundles(
            events=events,
            claims=claims,
            ranked_articles=ranked_articles,
            episode_context=episode_context,
        )
    )
    explanation = build_explanation_draft(
        episode_id=episode.id,
        episode_context=episode_context,
        attribution_payload=attribution_payload,
        bundles=bundles,
        analysis_mode=branch_explanation_mode,
        primary_mode=primary_mode,
    )
    return BranchAnalysisArtifacts(
        branch_mode=branch_mode,
        rerank_result=selected_rerank_result,
        claims=claims,
        events=events,
        bundles=bundles,
        explanation=explanation,
    )


def _build_shared_raw_pool_for_episode(
    *,
    episode: Episode,
    episode_context: Any,
    provider_context: Any,
):
    candidates = fetch_candidate_articles(
        episode_context=episode_context,
        provider_context=provider_context,
    )
    return build_shared_raw_pool(
        episode_id=episode.id,
        episode_context=episode_context,
        candidates=candidates,
    )


def _build_mixed_branch_analyses(
    *,
    episode: Episode,
    episode_context: Any,
    attribution_payload: dict[str, Any],
    router_result: Any,
    shared_raw_pool: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    # Preload the shared cross-encoder in the main thread before branch workers
    # start scoring articles. This avoids first-load instability inside the pool.
    get_cross_encoder_model(DEFAULT_CROSS_ENCODER_MODEL)

    def _build_one(branch_mode: str) -> dict[str, Any]:
        artifacts = _build_branch_artifacts(
            episode=episode,
            episode_context=episode_context,
            attribution_payload=attribution_payload,
            router_result=router_result,
            shared_raw_pool=shared_raw_pool,
            branch_mode=branch_mode,
            branch_explanation_mode=branch_mode,
            top_k=top_k,
        )
        outward = build_outward_result(
            episode_id=episode.id,
            ticker=episode_context.ticker,
            explanation=artifacts.explanation,
            attribution_payload=attribution_payload,
        )
        analysis_payload = build_analysis_v2_payload(
            outward_result=outward,
            explanation_draft=artifacts.explanation,
            attribution_payload=attribution_payload,
            evidence_bundles=artifacts.bundles,
            ranked_articles=artifacts.rerank_result.ranked_articles,
        )
        analysis = analysis_v2_payload_to_dict(analysis_payload)
        return {
            "branch_mode": branch_mode,
            "analysis": analysis,
            "decision": _build_v2_decision_payload(analysis),
        }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_build_one, mode) for mode in ("market", "industry", "firm")]
        return [f.result() for f in futures]


def _build_mixed_analysis_payload(
    *,
    attribution_payload: dict[str, Any],
    branch_analyses: list[dict[str, Any]],
    multi_branch_plan: Any,
) -> dict[str, Any]:
    aggregate_outward = _aggregate_mixed_outward(branch_analyses)
    aggregate_explanation = _build_mixed_aggregate_explanation(
        branch_analyses=branch_analyses,
        multi_branch_plan=multi_branch_plan,
    )
    attribution_summary = {
        "driver_hypothesis": "mixed",
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
    return {
        "routing_mode": "mixed",
        "aggregate_outward": aggregate_outward,
        "aggregate_explanation": aggregate_explanation,
        "attribution_summary": attribution_summary,
        "branch_analyses": branch_analyses,
    }


def _aggregate_mixed_outward(
    branch_analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = [item.get("analysis", {}).get("outward", {}).get("outward_status") for item in branch_analyses]
    if "NO_EVIDENCE" in statuses:
        aggregate_status = "NO_EVIDENCE"
    elif "LIMITED_EVIDENCE" in statuses:
        aggregate_status = "LIMITED_EVIDENCE"
    else:
        aggregate_status = "SUPPORTED"

    degraded_reason_codes: list[str] = []
    outward_notes: list[str] = []
    capability_notices: list[str] = []
    manual_review = False
    branch_status_parts: list[str] = []

    for item in branch_analyses:
        mode = item.get("branch_mode")
        outward = item.get("analysis", {}).get("outward", {}) or {}
        branch_status_parts.append(f"{mode}={outward.get('outward_status', 'NA')}")
        manual_review = manual_review or bool(outward.get("manual_review"))
        for code in outward.get("degraded_reason_codes") or []:
            if code not in degraded_reason_codes:
                degraded_reason_codes.append(code)
        for note in outward.get("outward_notes") or []:
            if note not in outward_notes:
                outward_notes.append(note)
        notice = outward.get("capability_boundary_notice")
        if notice and notice not in capability_notices:
            capability_notices.append(notice)

    outward_notes.insert(
        0,
        "Mixed episodes are summarized conservatively from the most restrictive branch result.",
    )
    if branch_status_parts:
        outward_notes.append(
            "Branch outward statuses: " + ", ".join(branch_status_parts) + "."
        )

    capability_boundary_notice = None
    if len(capability_notices) == 1:
        capability_boundary_notice = capability_notices[0]
    elif capability_notices:
        capability_boundary_notice = (
            "Branch-specific capability notices differ across the mixed explanation set; "
            "review the branch cards below."
        )

    return {
        "outward_status": aggregate_status,
        "manual_review": manual_review,
        "capability_boundary_notice": capability_boundary_notice,
        "degraded_reason_codes": degraded_reason_codes,
        "outward_notes": outward_notes,
    }


def _build_mixed_aggregate_explanation(
    *,
    branch_analyses: list[dict[str, Any]],
    multi_branch_plan: Any,
) -> dict[str, Any]:
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    branch_summaries: list[dict[str, Any]] = []
    min_confidence = "high"
    for item in branch_analyses:
        mode = item.get("branch_mode")
        analysis = item.get("analysis") or {}
        explanation = analysis.get("explanation") or {}
        outward = analysis.get("outward") or {}
        confidence = explanation.get("explanation_confidence") or "low"
        if confidence_rank.get(confidence, 0) < confidence_rank.get(min_confidence, 0):
            min_confidence = confidence
        branch_summaries.append(
            {
                "branch_mode": mode,
                "outward_status": outward.get("outward_status"),
                "driver_hypothesis": explanation.get("driver_hypothesis"),
                "headline_explanation": explanation.get("headline_explanation"),
                "explanation_confidence": confidence,
            }
        )

    return {
        "driver_hypothesis": "mixed",
        "headline_explanation": (
            "This episode remains mixed across market, industry, and firm explanation branches. "
            "Review the branch-specific analyses below rather than relying on a single dominant narrative."
        ),
        "explanation_confidence": min_confidence,
        "default_visible_modes": list(multi_branch_plan.default_modes),
        "branch_summaries": branch_summaries,
    }


def _build_attribution_bundle(attr: Any) -> dict[str, Any]:
    return {
        "ticker": attr.ticker,
        "market_ratio": attr.market_ratio,
        "industry_ratio": attr.industry_ratio,
        "firm_ratio": attr.firm_ratio,
        "alpha_total": attr.alpha_total,
        "market_total": attr.market_total,
        "industry_total": attr.industry_total,
        "firm_total": attr.firm_total,
        "episode_return_total": attr.episode_return_total,
        "adj_r2": attr.adj_r2,
        "recon_error_bps": attr.recon_error_bps,
        "quality_status": attr.quality_status,
        "reason": attr.reason,
        "reason_code": attr.reason_code,
        "beta_mkt_t_stat": attr.beta_mkt_t_stat,
        "beta_ind_t_stat": attr.beta_ind_t_stat,
        "f_stat": attr.f_stat,
        "f_pvalue": attr.f_pvalue,
        "fit_quality_note": attr.fit_quality_note,
        "market_share": attr.market_share,
        "industry_share": attr.industry_share,
        "firm_share": attr.firm_share,
        "alpha_share": attr.alpha_share,
        "near_zero_episode_return": attr.near_zero_episode_return,
        "industry_proxy_ticker": attr.industry_proxy_ticker,
        "industry_proxy_source": attr.industry_proxy_source,
        "vix_estimation_p50": attr.vix_estimation_p50,
        "vix_estimation_p95": attr.vix_estimation_p95,
        "vix_episode_max": attr.vix_episode_max,
        "vix_episode_avg": attr.vix_episode_avg,
        "macro_shock_suspected": attr.macro_shock_suspected,
    }
