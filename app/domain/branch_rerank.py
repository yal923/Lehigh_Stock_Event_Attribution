from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from app.domain.attribution_router import AttributionRouterResult
from app.domain.news_v2_01_retrieval import (
    DEFAULT_CROSS_ENCODER_MODEL,
    EpisodeContext,
    RankedArticle,
)
from app.domain.shared_raw_pool import RawPoolArticle, SharedRawPool
from app.infrastructure.resources import get_cross_encoder_model


BranchScorer = Callable[[str, Sequence[RawPoolArticle]], Sequence[float]]
BRANCH_MODES = ("market", "industry", "firm")
# DEC-198 A: hard scope filter per branch. Branch modes are named after the
# attribution factor ("firm"), but the retrieval scope_tag uses "company" for
# firm-level content. Keep the mapping explicit so neither side has to know
# about the other's vocabulary.
BRANCH_MODE_TO_SCOPE_TAG = {
    "market": "market",
    "industry": "industry",
    "firm": "company",
}


@dataclass(frozen=True)
class BranchRerankInput:
    episode_id: int
    branch_mode: str
    shared_raw_pool: SharedRawPool
    episode_context: EpisodeContext
    router_result: AttributionRouterResult


@dataclass(frozen=True)
class BranchRerankResult:
    branch_mode: str
    rerank_query: str
    ranked_articles: tuple[RankedArticle, ...]


def build_branch_rerank_query(
    *,
    branch_mode: str,
    episode_context: EpisodeContext,
    router_result: AttributionRouterResult,
) -> str:
    _validate_branch_mode(branch_mode)
    mode_weights = router_result.mode_weights
    branch_instruction = _branch_instruction(branch_mode)
    visible_modes = ", ".join(router_result.default_visible_modes) or "none"

    return (
        f"episode={episode_context.company_name} ({episode_context.ticker}); "
        f"window={episode_context.episode_start}..{episode_context.episode_end}; "
        f"peak={episode_context.peak_date}; "
        f"direction={episode_context.direction}; "
        f"branch_mode={branch_mode}; "
        f"router_mode={router_result.routing_mode}; "
        f"default_visible_modes={visible_modes}; "
        f"mode_weights=market:{mode_weights.get('market', 0.0):.2f},"
        f"industry:{mode_weights.get('industry', 0.0):.2f},"
        f"firm:{mode_weights.get('firm', 0.0):.2f}; "
        f"{branch_instruction}"
    )


def build_branch_article_text(
    *,
    article: RawPoolArticle,
) -> str:
    scope_phrase = _scope_phrase(article.scope_tag)
    parts = [
        f"scope: {article.scope_tag}",
        f"scope_context: {scope_phrase}",
        f"provider: {article.provider}",
        f"publisher: {article.publisher or 'unknown'}",
        article.headline.strip(),
    ]
    snippet = article.summary_or_snippet.strip()
    if snippet:
        parts.append(snippet)
    return "\n\n".join(part for part in parts if part)


def score_branch_articles_with_cross_encoder(
    rerank_query: str,
    articles: Sequence[RawPoolArticle],
    *,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> list[float]:
    if not articles:
        return []

    model = get_cross_encoder_model(model_name)
    if model is None:
        raise RuntimeError(
            "Cross-encoder reranker is unavailable. "
            f"Expected model: {model_name}"
        )

    pairs = [
        (rerank_query, build_branch_article_text(article=article))
        for article in articles
    ]
    scores = model.predict(pairs)
    return [float(score) for score in scores]


def score_branch_articles_with_default_cross_encoder(
    rerank_query: str,
    articles: Sequence[RawPoolArticle],
) -> list[float]:
    return score_branch_articles_with_cross_encoder(
        rerank_query=rerank_query,
        articles=articles,
        model_name=DEFAULT_CROSS_ENCODER_MODEL,
    )


def run_branch_rerank(
    branch_input: BranchRerankInput,
    *,
    scorer: BranchScorer,
    rank_method: str,
) -> BranchRerankResult:
    rerank_query = build_branch_rerank_query(
        branch_mode=branch_input.branch_mode,
        episode_context=branch_input.episode_context,
        router_result=branch_input.router_result,
    )
    articles = _filter_articles_by_branch_scope(
        articles=branch_input.shared_raw_pool.articles,
        branch_mode=branch_input.branch_mode,
    )
    if not articles:
        return BranchRerankResult(
            branch_mode=branch_input.branch_mode,
            rerank_query=rerank_query,
            ranked_articles=(),
        )

    scores = scorer(rerank_query, articles)
    ranked_pairs = sorted(
        zip(articles, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    ranked_articles = tuple(
        _ranked_article_from_raw_pool_article(
            article=article,
            rerank_query=rerank_query,
            rank_score=score,
            rank_method=rank_method,
            rank_position=position,
        )
        for position, (article, score) in enumerate(ranked_pairs, start=1)
    )
    return BranchRerankResult(
        branch_mode=branch_input.branch_mode,
        rerank_query=rerank_query,
        ranked_articles=ranked_articles,
    )


def branch_rerank_result_to_dict(
    result: BranchRerankResult,
) -> dict[str, Any]:
    return {
        "branch_mode": result.branch_mode,
        "rerank_query": result.rerank_query,
        "ranked_articles": [asdict(article) for article in result.ranked_articles],
    }


def _ranked_article_from_raw_pool_article(
    *,
    article: RawPoolArticle,
    rerank_query: str,
    rank_score: float,
    rank_method: str,
    rank_position: int,
) -> RankedArticle:
    return RankedArticle(
        article_id=article.article_id,
        ticker=article.ticker,
        provider=article.provider,
        scope_tag=article.scope_tag,
        external_id=article.external_id,
        raw_payload_ref=dict(article.raw_payload_ref),
        headline=article.headline,
        summary_or_snippet=article.summary_or_snippet,
        url=article.url,
        published_at=article.published_at,
        publisher=article.publisher,
        episode_query=rerank_query,
        rank_score=float(rank_score),
        rank_method=rank_method,
        rank_position=rank_position,
    )


def _branch_instruction(branch_mode: str) -> str:
    instructions = {
        "market": (
            "prioritize market-wide or sector-wide explanations; "
            "company-specific stories may be secondary"
        ),
        "industry": (
            "prioritize industry, sector, or peer-company explanations; "
            "general market context and firm-only stories may be secondary"
        ),
        "firm": (
            "prioritize company-specific explanations such as earnings, guidance, "
            "analyst views, M&A, products, management, or firm operations"
        ),
    }
    return instructions[branch_mode]


def _scope_phrase(scope_tag: str) -> str:
    phrases = {
        "market": "This article entered the shared pool through market-oriented retrieval scope.",
        "industry": "This article entered the shared pool through industry- or peer-oriented retrieval scope.",
        "company": "This article entered the shared pool through company-specific retrieval scope.",
    }
    return phrases.get(scope_tag, "This article entered the shared pool through an unspecified retrieval scope.")


def _validate_branch_mode(branch_mode: str) -> None:
    if branch_mode not in BRANCH_MODES:
        raise ValueError(f"Unsupported branch mode: {branch_mode}")


def _filter_articles_by_branch_scope(
    *,
    articles: Sequence[RawPoolArticle],
    branch_mode: str,
) -> tuple[RawPoolArticle, ...]:
    required_scope_tag = BRANCH_MODE_TO_SCOPE_TAG[branch_mode]
    return tuple(
        article
        for article in articles
        if article.scope_tag == required_scope_tag
    )


__all__ = [
    "BRANCH_MODES",
    "BRANCH_MODE_TO_SCOPE_TAG",
    "BranchRerankInput",
    "BranchRerankResult",
    "BranchScorer",
    "branch_rerank_result_to_dict",
    "build_branch_article_text",
    "build_branch_rerank_query",
    "run_branch_rerank",
    "score_branch_articles_with_cross_encoder",
    "score_branch_articles_with_default_cross_encoder",
]
