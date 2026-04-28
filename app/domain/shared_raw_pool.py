from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.domain.news_v2_01_retrieval import CandidateArticle, EpisodeContext


@dataclass(frozen=True)
class RawPoolArticle:
    article_id: str
    ticker: str
    provider: str
    scope_tag: str
    scope_key: str | None
    external_id: str | None
    headline: str
    summary_or_snippet: str
    url: str
    published_at: Any
    publisher: str | None
    raw_payload_ref: dict[str, Any]


@dataclass(frozen=True)
class SharedRawPool:
    episode_id: int
    retrieval_mode: str
    articles: tuple[RawPoolArticle, ...]
    provider_summary: dict[str, int]
    scope_summary: dict[str, int]


def build_shared_raw_pool(
    *,
    episode_id: int,
    episode_context: EpisodeContext,
    candidates: list[CandidateArticle],
) -> SharedRawPool:
    articles = tuple(raw_pool_article_from_candidate(candidate) for candidate in candidates)
    return SharedRawPool(
        episode_id=episode_id,
        retrieval_mode=episode_context.retrieval_mode,
        articles=articles,
        provider_summary=_count_by_provider(candidates),
        scope_summary=_count_by_scope(candidates),
    )


def raw_pool_article_from_candidate(
    candidate: CandidateArticle,
) -> RawPoolArticle:
    raw_payload_ref = dict(candidate.raw_payload_ref)
    return RawPoolArticle(
        article_id=candidate.article_id,
        ticker=candidate.ticker,
        provider=candidate.provider,
        scope_tag=candidate.scope_tag,
        scope_key=_scope_key_from_payload(raw_payload_ref),
        external_id=candidate.external_id,
        headline=candidate.headline,
        summary_or_snippet=candidate.summary_or_snippet,
        url=candidate.url,
        published_at=candidate.published_at,
        publisher=candidate.publisher,
        raw_payload_ref=raw_payload_ref,
    )


def shared_raw_pool_to_dict(
    pool: SharedRawPool,
) -> dict[str, Any]:
    return asdict(pool)


def _count_by_provider(
    candidates: list[CandidateArticle],
) -> dict[str, int]:
    summary: dict[str, int] = {}
    for candidate in candidates:
        summary[candidate.provider] = summary.get(candidate.provider, 0) + 1
    return summary


def _count_by_scope(
    candidates: list[CandidateArticle],
) -> dict[str, int]:
    summary: dict[str, int] = {}
    for candidate in candidates:
        summary[candidate.scope_tag] = summary.get(candidate.scope_tag, 0) + 1
    return summary


def _scope_key_from_payload(raw_payload_ref: dict[str, Any]) -> str | None:
    value = raw_payload_ref.get("scope_key")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "RawPoolArticle",
    "SharedRawPool",
    "build_shared_raw_pool",
    "raw_pool_article_from_candidate",
    "shared_raw_pool_to_dict",
]
