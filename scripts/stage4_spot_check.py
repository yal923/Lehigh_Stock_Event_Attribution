"""
Stage 4 spot-check: NVDA ep10 + ORCL ep478 (Hormuz week 2026-04-13..04-17).

Two layers of verification:

  Layer 1 — retrieval/filter diagnostic (no LLM):
    - Build shared raw pool from live providers.
    - Report scope_tag distribution (market / industry / company / other).
    - Apply _filter_articles_by_branch_scope per branch; assert all surviving
      articles match the expected scope (DEC-198 A).
    - For scope_tag=="market" rows, verify no headline/summary mentions the
      episode ticker or primary company name (DEC-198 D).

  Layer 2 — full pipeline run (force_recompute=True):
    - Runs LLM stages, attribution, outward, v2 analysis.
    - Reports attribution fields including macro_shock_suspected, VIX
      diagnostics, shares, industry proxy (DEC-199 H/G/F).
    - Reports outward_notes and degraded_reason_codes for macro-shock caveat.
    - Inspects serialized supporting_articles for scope_tag alignment in the
      final UI-facing payload.

Run:
    ~/venvs/vds/bin/python scripts/stage4_spot_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from app.domain.branch_rerank import (
    BRANCH_MODE_TO_SCOPE_TAG,
    _filter_articles_by_branch_scope,
)
from app.domain.episode_pipeline import _build_shared_raw_pool_for_episode
from app.domain.news_v2_01_retrieval import (
    _market_article_mentions_subject,
    build_episode_context_from_episode,
    build_provider_context,
)
from app.models import Episode
from app.services.pipeline_service import run_episode_pipeline


SPOT_CHECK_EPISODE_IDS = [10, 478]  # NVDA, ORCL


def _layer1_retrieval_diagnostic(
    episode, episode_context, provider_context,
) -> dict:
    shared_pool = _build_shared_raw_pool_for_episode(
        episode=episode,
        episode_context=episode_context,
        provider_context=provider_context,
    )
    articles = shared_pool.articles

    scope_counts: dict[str, int] = {}
    for article in articles:
        scope_counts[article.scope_tag] = scope_counts.get(article.scope_tag, 0) + 1

    per_branch: dict[str, dict] = {}
    for branch_mode in ("market", "industry", "firm"):
        filtered = _filter_articles_by_branch_scope(
            articles=articles,
            branch_mode=branch_mode,
        )
        expected = BRANCH_MODE_TO_SCOPE_TAG[branch_mode]
        misaligned = [a for a in filtered if a.scope_tag != expected]
        per_branch[branch_mode] = {
            "expected_scope": expected,
            "post_filter_count": len(filtered),
            "misaligned_count": len(misaligned),
            "sample_headlines": [a.headline[:90] for a in filtered[:3]],
        }

    # DEC-198 D verification: market-scope articles must not mention ticker/name.
    market_articles = [a for a in articles if a.scope_tag == "market"]
    market_contaminated = []
    for a in market_articles:
        # RawPoolArticle has the same surface as CandidateArticle for this check.
        if _market_article_mentions_subject(
            candidate=a,
            ticker=episode_context.ticker,
            company_name=episode_context.company_name,
        ):
            market_contaminated.append(a.headline[:120])

    return {
        "shared_pool_size": len(articles),
        "provider_summary": shared_pool.provider_summary,
        "scope_counts": scope_counts,
        "per_branch_filter": per_branch,
        "market_contaminated_count": len(market_contaminated),
        "market_contaminated_samples": market_contaminated[:3],
    }


def _scope_alignment_from_supporting_articles(result: dict) -> list[dict]:
    analysis = result.get("analysis") or {}
    branch_analyses = analysis.get("branch_analyses") or []
    rows: list[dict] = []

    if branch_analyses:  # mixed routing
        for ba in branch_analyses:
            branch_mode = ba.get("branch_mode")
            inner_analysis = ba.get("analysis") or {}
            supporting = inner_analysis.get("supporting_articles") or []
            rows.append(_alignment_row(branch_mode, supporting))
    else:  # dominant / degraded routing
        branch_mode = (result.get("branch_plan") or {}).get("primary_mode") or "unknown"
        supporting = analysis.get("supporting_articles") or []
        rows.append(_alignment_row(branch_mode, supporting))

    return rows


def _alignment_row(branch_mode: str, supporting_articles: list) -> dict:
    expected_scope = BRANCH_MODE_TO_SCOPE_TAG.get(branch_mode)
    scope_counts: dict[str, int] = {}
    off_scope: list[str] = []
    for article in supporting_articles:
        tag = article.get("scope_tag") or "unknown"
        scope_counts[tag] = scope_counts.get(tag, 0) + 1
        if expected_scope and tag != expected_scope:
            off_scope.append(f"[{tag}] {article.get('headline', '')[:90]}")
    return {
        "branch_mode": branch_mode,
        "expected_scope": expected_scope,
        "supporting_count": len(supporting_articles),
        "scope_counts": scope_counts,
        "off_scope_count": len(off_scope),
        "off_scope_samples": off_scope[:3],
        "aligned": len(off_scope) == 0,
    }


def _summarize_pipeline_result(result: dict) -> dict:
    attribution = result.get("attribution") or {}
    decision = result.get("decision") or {}
    branch_plan = result.get("branch_plan") or {}

    outward_notes = decision.get("outward_notes") or []
    return {
        "routing_mode": branch_plan.get("routing_mode"),
        "primary_mode": branch_plan.get("primary_mode"),
        "quality_status": attribution.get("quality_status"),
        "adj_r2": attribution.get("adj_r2"),
        "outward_status": decision.get("outward_status"),
        "manual_review": decision.get("manual_review"),
        "macro_shock_suspected": attribution.get("macro_shock_suspected"),
        "macro_note_present": any("Macro shock" in n for n in outward_notes),
        "vix_train_p50": attribution.get("vix_estimation_p50"),
        "vix_train_p95": attribution.get("vix_estimation_p95"),
        "vix_episode_avg": attribution.get("vix_episode_avg"),
        "vix_episode_max": attribution.get("vix_episode_max"),
        "industry_proxy_ticker": attribution.get("industry_proxy_ticker"),
        "industry_proxy_source": attribution.get("industry_proxy_source"),
        "market_share": attribution.get("market_share"),
        "industry_share": attribution.get("industry_share"),
        "firm_share": attribution.get("firm_share"),
        "alpha_share": attribution.get("alpha_share"),
        "market_ratio": attribution.get("market_ratio"),
        "industry_ratio": attribution.get("industry_ratio"),
        "firm_ratio": attribution.get("firm_ratio"),
        "episode_return_total": attribution.get("episode_return_total"),
        "near_zero_episode_return": attribution.get("near_zero_episode_return"),
        "degraded_reason_codes": decision.get("degraded_reason_codes") or [],
    }


def _print_layer1(ticker: str, diag: dict) -> None:
    print(f"[Layer 1] Retrieval + filter diagnostic")
    print(f"  shared pool size  : {diag['shared_pool_size']}")
    print(f"  provider_summary  : {diag['provider_summary']}")
    print(f"  scope_counts      : {diag['scope_counts']}")
    for branch_mode, row in diag["per_branch_filter"].items():
        print(
            f"  {branch_mode:<8} expected={row['expected_scope']:<8} "
            f"post-filter={row['post_filter_count']}  "
            f"misaligned={row['misaligned_count']}"
        )
        for headline in row["sample_headlines"]:
            print(f"    sample: {headline}")
    print(
        f"  market scope contamination (DEC-198 D): "
        f"{diag['market_contaminated_count']} found"
    )
    for headline in diag["market_contaminated_samples"]:
        print(f"    CONTAMINATED: {headline}")


def _print_layer2(ticker: str, summary: dict, alignment: list) -> None:
    print(f"\n[Layer 2] Full pipeline result")
    print(f"  routing_mode     : {summary['routing_mode']}  (primary={summary['primary_mode']})")
    print(f"  quality_status   : {summary['quality_status']}  adj_r2={summary['adj_r2']}")
    print(f"  outward_status   : {summary['outward_status']}  manual_review={summary['manual_review']}")
    print(
        f"  macro_shock      : {summary['macro_shock_suspected']}  "
        f"note_present={summary['macro_note_present']}"
    )
    print(
        f"    VIX train p50={summary['vix_train_p50']}  p95={summary['vix_train_p95']}"
    )
    print(
        f"    VIX ep   avg={summary['vix_episode_avg']}  max={summary['vix_episode_max']}"
    )
    print(
        f"  industry_proxy   : {summary['industry_proxy_ticker']}  "
        f"({summary['industry_proxy_source']})"
    )
    print(
        f"  shares (signed)  : firm={summary['firm_share']}  ind={summary['industry_share']}  "
        f"mkt={summary['market_share']}  alpha={summary['alpha_share']}"
    )
    print(
        f"  ratios (|abs|)   : firm={summary['firm_ratio']}  ind={summary['industry_ratio']}  "
        f"mkt={summary['market_ratio']}"
    )
    print(
        f"  episode_return   : {summary['episode_return_total']}  "
        f"near_zero={summary['near_zero_episode_return']}"
    )
    print(f"  degraded_codes   : {summary['degraded_reason_codes']}")

    print(f"\n  Branch alignment (from serialized supporting_articles):")
    for row in alignment:
        status = "OK" if row["aligned"] else "MISALIGNED"
        print(
            f"    {row['branch_mode']:<8} expected={row['expected_scope']}  "
            f"supporting={row['supporting_count']}  counts={row['scope_counts']}  [{status}]"
        )
        for sample in row["off_scope_samples"]:
            print(f"      off-scope: {sample}")


def main() -> int:
    app = create_app()
    with app.app_context():
        all_reports: list[dict] = []
        for ep_id in SPOT_CHECK_EPISODE_IDS:
            ep = Episode.query.get(ep_id)
            if ep is None:
                print(f"[ERROR] Episode {ep_id} not found")
                continue
            ticker = ep.company.ticker if ep.company else f"ep{ep_id}"
            print(f"\n{'=' * 72}")
            print(
                f"[{ticker}] episode_id={ep_id}  "
                f"window={ep.window_start}..{ep.window_end}  peak={ep.peak_date}"
            )
            print("=" * 72)

            # Layer 1: retrieval + filter diagnostic (fast, no LLM).
            episode_context = build_episode_context_from_episode(ep)
            provider_context = build_provider_context()
            layer1 = _layer1_retrieval_diagnostic(
                ep, episode_context, provider_context
            )
            _print_layer1(ticker, layer1)

            # Layer 2: full pipeline (force_recompute=True).
            result = run_episode_pipeline(ep, force_recompute=True)
            summary = _summarize_pipeline_result(result)
            alignment = _scope_alignment_from_supporting_articles(result)
            _print_layer2(ticker, summary, alignment)

            all_reports.append(
                {
                    "ticker": ticker,
                    "episode_id": ep_id,
                    "layer1": layer1,
                    "layer2_summary": summary,
                    "layer2_alignment": alignment,
                }
            )

        out_path = Path("instance") / "verification" / "stage4_spot_check.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_reports, indent=2, default=str))
        print(f"\nReport saved: {out_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
