from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from typing import Any, Callable, Sequence

from app.config.constants import (
    COMPANY_PROFILES,
    QUERY_TEMPLATE,
    TICKER_TO_NAME_MAP,
    clean_company_name,
    fingerprint_news,
    normalize_publisher,
)
from app.config.settings import Settings, get_settings
from app.domain.news_retrieval import (
    _build_provider_budgets,
    _consume_provider_budget,
    _deduplicate_rows,
    _fetch_finnhub_items,
    _fetch_finnhub_market_news_items,
    _fetch_google_rss_items,
    _fetch_google_rss_query_items,
    _fetch_stocknews_items,
)
from app.infrastructure.resources import get_cross_encoder_model
from app.models import Episode


ArticleScorer = Callable[[str, Sequence["CandidateArticle"]], Sequence[float]]
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
LOW_VALUE_PUBLISHER_DENYLIST = frozenset(
    {
        "mexc",
        "mexcexchange",
        "moneycheck",
        "tradingkey",
    }
)
NEAR_DUPLICATE_HEADLINE_THRESHOLD = 0.92


@dataclass(frozen=True)
class EpisodeContext:
    ticker: str
    company_name: str
    episode_start: str
    episode_end: str
    peak_date: str
    direction: str
    attribution_hint: str | None = None
    retrieval_mode: str = "degraded"
    sector_name: str | None = None
    sub_industry_name: str | None = None
    industry_terms: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProviderContext:
    enabled_providers: tuple[str, ...]
    api_capability_flags: dict[str, dict[str, bool]]
    quota_or_budget_hints: dict[str, int | None]
    stocknews_api_key: str | None = field(default=None, repr=False)
    finnhub_api_key: str | None = field(default=None, repr=False)
    stocknews_base_url: str = field(
        default="https://stocknewsapi.com/api/v1",
        repr=False,
    )
    enable_google_rss: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class CandidateArticle:
    article_id: str
    ticker: str
    provider: str
    scope_tag: str
    external_id: str | None
    raw_payload_ref: dict[str, Any]
    headline: str
    summary_or_snippet: str
    url: str
    published_at: Any
    publisher: str | None


@dataclass(frozen=True)
class RankedArticle(CandidateArticle):
    episode_query: str
    rank_score: float
    rank_method: str
    rank_position: int


def build_episode_context_from_episode(
    episode: Episode,
    attribution_hint: str | None = None,
    attribution_payload: dict[str, Any] | None = None,
    retrieval_mode: str | None = None,
) -> EpisodeContext:
    ticker = episode.company.ticker.upper()
    company_name = clean_company_name(
        TICKER_TO_NAME_MAP.get(ticker, episode.company.name or ticker)
    )
    direction_map = {
        "up": "positive",
        "down": "negative",
    }

    return EpisodeContext(
        ticker=ticker,
        company_name=company_name,
        episode_start=episode.window_start.isoformat(),
        episode_end=episode.window_end.isoformat(),
        peak_date=episode.peak_date.isoformat(),
        direction=direction_map.get(episode.direction, episode.direction),
        attribution_hint=attribution_hint,
        retrieval_mode=retrieval_mode if retrieval_mode is not None else _retrieval_mode_from_attribution(attribution_payload),
        sector_name=_string_or_none(getattr(episode.company, "gics_sector", None)),
        sub_industry_name=_string_or_none(
            getattr(episode.company, "gics_sub_industry", None)
        ),
        industry_terms=_industry_terms_for_ticker(
            ticker=ticker,
            sub_industry_name=_string_or_none(
                getattr(episode.company, "gics_sub_industry", None)
            ),
        ),
    )


def build_provider_context(settings: Settings | None = None) -> ProviderContext:
    settings = settings or get_settings()

    enabled: list[str] = []
    if settings.stocknews_api_key:
        enabled.append("StockNewsAPI")
    if settings.finnhub_api_key:
        enabled.append("Finnhub")
    if settings.enable_google_rss:
        enabled.append("GoogleRSS")

    api_capability_flags = {
        "StockNewsAPI": {
            "ticker_news": bool(settings.stocknews_api_key),
            "general_market_news": bool(settings.stocknews_api_key),
            "source_filters": bool(settings.stocknews_api_key),
            "type_sentiment_filters": bool(settings.stocknews_api_key),
            "rank_sort": bool(settings.stocknews_api_key),
        },
        "Finnhub": {
            "ticker_news": bool(settings.finnhub_api_key),
            "general_market_news": bool(settings.finnhub_api_key),
        },
        "GoogleRSS": {
            "ticker_news": bool(settings.enable_google_rss),
            "general_market_news": bool(settings.enable_google_rss),
        },
    }

    budgets = _build_provider_budgets(settings)
    budgets.setdefault("GoogleRSS", None)

    return ProviderContext(
        enabled_providers=tuple(enabled),
        api_capability_flags=api_capability_flags,
        quota_or_budget_hints=budgets,
        stocknews_api_key=settings.stocknews_api_key,
        finnhub_api_key=settings.finnhub_api_key,
        stocknews_base_url=settings.stocknews_base_url,
        enable_google_rss=settings.enable_google_rss,
    )


def build_structured_episode_query(context: EpisodeContext) -> str:
    direction_label = context.direction
    template = QUERY_TEMPLATE.get(direction_label)
    if template:
        directional_clause = template.format(context.company_name)
    else:
        directional_clause = (
            f"{context.company_name} ({context.ticker}) episode around {context.peak_date}"
        )

    attribution_clause = context.attribution_hint or "unspecified"
    return (
        f"{directional_clause}; "
        f"ticker={context.ticker}; "
        f"window={context.episode_start}..{context.episode_end}; "
        f"peak={context.peak_date}; "
        f"attribution_hint={attribution_clause}"
    )


def fetch_candidate_articles(
    episode_context: EpisodeContext,
    provider_context: ProviderContext,
) -> list[CandidateArticle]:
    budgets = dict(provider_context.quota_or_budget_hints)
    rows: list[dict[str, Any]] = []

    if (
        "StockNewsAPI" in provider_context.enabled_providers
        and provider_context.stocknews_api_key
        and _consume_provider_budget("StockNewsAPI", budgets)
    ):
        rows.extend(
            _fetch_stocknews_scoped_rows(
                episode_context=episode_context,
                provider_context=provider_context,
            )
        )

    if (
        "Finnhub" in provider_context.enabled_providers
        and provider_context.finnhub_api_key
        and _consume_provider_budget("Finnhub", budgets)
    ):
        rows.extend(
            _fetch_finnhub_scoped_rows(
                episode_context=episode_context,
                provider_context=provider_context,
            )
        )

    if (
        "GoogleRSS" in provider_context.enabled_providers
        and provider_context.enable_google_rss
    ):
        rows.extend(
            _fetch_google_scoped_rows(
                episode_context=episode_context,
            )
        )

    deduped_rows = _deduplicate_rows(rows)
    candidates = [_candidate_from_row(episode_context.ticker, row) for row in deduped_rows]
    return _apply_candidate_hygiene(
        candidates=candidates,
        episode_context=episode_context,
    )


def rerank_candidate_articles(
    candidates: Sequence[CandidateArticle],
    episode_query: str,
    scores: Sequence[float],
    rank_method: str,
) -> list[RankedArticle]:
    if len(candidates) != len(scores):
        raise ValueError("Candidate article count must match score count")

    ranked_pairs = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    ranked: list[RankedArticle] = []
    for position, (candidate, score) in enumerate(ranked_pairs, start=1):
        ranked.append(
            RankedArticle(
                article_id=candidate.article_id,
                ticker=candidate.ticker,
                provider=candidate.provider,
                scope_tag=candidate.scope_tag,
                external_id=candidate.external_id,
                raw_payload_ref=candidate.raw_payload_ref,
                headline=candidate.headline,
                summary_or_snippet=candidate.summary_or_snippet,
                url=candidate.url,
                published_at=candidate.published_at,
                publisher=candidate.publisher,
                episode_query=episode_query,
                rank_score=float(score),
                rank_method=rank_method,
                rank_position=position,
            )
        )

    return ranked


def score_candidate_articles_with_cross_encoder(
    episode_query: str,
    candidates: Sequence[CandidateArticle],
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> list[float]:
    if not candidates:
        return []

    model = get_cross_encoder_model(model_name)
    if model is None:
        raise RuntimeError(
            "Cross-encoder reranker is unavailable. "
            f"Expected model: {model_name}"
        )

    pairs = [
        (episode_query, _build_article_text(candidate))
        for candidate in candidates
    ]
    scores = model.predict(pairs)
    return [float(score) for score in scores]


def score_candidate_articles_with_default_cross_encoder(
    episode_query: str,
    candidates: Sequence[CandidateArticle],
) -> list[float]:
    return score_candidate_articles_with_cross_encoder(
        episode_query=episode_query,
        candidates=candidates,
        model_name=DEFAULT_CROSS_ENCODER_MODEL,
    )


def run_retrieval_reranking(
    episode_context: EpisodeContext,
    provider_context: ProviderContext,
    scorer: ArticleScorer,
    rank_method: str,
) -> list[RankedArticle]:
    candidates = fetch_candidate_articles(
        episode_context=episode_context,
        provider_context=provider_context,
    )
    if not candidates:
        return []

    episode_query = build_structured_episode_query(episode_context)
    scores = scorer(episode_query, candidates)
    return rerank_candidate_articles(
        candidates=candidates,
        episode_query=episode_query,
        scores=scores,
        rank_method=rank_method,
    )


def _candidate_from_row(ticker: str, row: dict[str, Any]) -> CandidateArticle:
    headline = (row.get("headline") or "").strip()
    summary_or_snippet = (row.get("summary") or "").strip()
    url = (row.get("url") or "").strip()
    article_id = row.get("fingerprint") or fingerprint_news(url, headline)

    raw_payload_ref = {
        "provider": row.get("api_source"),
        "external_id": row.get("external_id"),
        "url": url,
        "published_at": row.get("published_at"),
        "scope_tag": row.get("scope_tag"),
        "scope_key": row.get("scope_key"),
    }

    return CandidateArticle(
        article_id=article_id,
        ticker=ticker,
        provider=(row.get("api_source") or "").strip(),
        scope_tag=(row.get("scope_tag") or "company").strip(),
        external_id=_string_or_none(row.get("external_id")),
        raw_payload_ref=raw_payload_ref,
        headline=headline,
        summary_or_snippet=summary_or_snippet,
        url=url,
        published_at=row.get("published_at"),
        publisher=_string_or_none(row.get("publisher")),
    )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _apply_candidate_hygiene(
    *,
    candidates: Sequence[CandidateArticle],
    episode_context: EpisodeContext,
) -> list[CandidateArticle]:
    filtered: list[CandidateArticle] = []
    seen_headlines: list[str] = []

    for candidate in candidates:
        if _is_low_value_publisher(candidate.publisher):
            continue

        # DEC-198 D: market-scope articles must not mention the episode ticker
        # or the company's primary name. Such articles are ticker-colored market
        # news, not genuinely macro content, and contaminate the market branch.
        if candidate.scope_tag == "market" and _market_article_mentions_subject(
            candidate=candidate,
            ticker=episode_context.ticker,
            company_name=episode_context.company_name,
        ):
            continue

        normalized_headline = _normalized_headline_for_dedup(
            candidate.headline,
            ticker=episode_context.ticker,
            company_name=episode_context.company_name,
        )
        if normalized_headline and any(
            SequenceMatcher(None, normalized_headline, seen).ratio() >= NEAR_DUPLICATE_HEADLINE_THRESHOLD
            for seen in seen_headlines
        ):
            continue

        filtered.append(candidate)
        if normalized_headline:
            seen_headlines.append(normalized_headline)

    return filtered


def _market_article_mentions_subject(
    *,
    candidate: CandidateArticle,
    ticker: str,
    company_name: str,
) -> bool:
    text_parts = [candidate.headline or "", candidate.summary_or_snippet or ""]
    text = " ".join(part for part in text_parts if part).lower()
    if not text.strip():
        return False

    ticker_lower = ticker.strip().lower()
    if ticker_lower and re.search(rf"\b{re.escape(ticker_lower)}\b", text):
        return True

    name_lower = (company_name or "").strip().lower()
    if name_lower and name_lower in text:
        return True

    return False


def _is_low_value_publisher(publisher: str | None) -> bool:
    if not publisher:
        return False
    return normalize_publisher(publisher) in LOW_VALUE_PUBLISHER_DENYLIST


def _normalized_headline_for_dedup(
    headline: str,
    *,
    ticker: str,
    company_name: str,
) -> str:
    text = headline.lower()
    text = text.replace(ticker.lower(), " ")
    for token in company_name.lower().split():
        text = text.replace(token, " ")
    for token in ("stock", "shares", "share", "company", "co", "inc"):
        text = text.replace(token, " ")
    parts = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text).split()
    return " ".join(parts)


def _build_article_text(candidate: CandidateArticle) -> str:
    parts = [candidate.headline.strip()]
    snippet = candidate.summary_or_snippet.strip()
    if snippet:
        parts.append(snippet)
    return "\n\n".join(part for part in parts if part)


def _retrieval_mode_from_attribution(
    attribution_payload: dict[str, Any] | None,
) -> str:
    if not attribution_payload:
        return "degraded"

    if (attribution_payload.get("quality_status") or "").upper() != "PASS":
        return "degraded"

    ratios = [
        ("firm", float(attribution_payload.get("firm_ratio") or 0.0)),
        ("industry", float(attribution_payload.get("industry_ratio") or 0.0)),
        ("market", float(attribution_payload.get("market_ratio") or 0.0)),
    ]
    ranked = sorted(ratios, key=lambda item: item[1], reverse=True)
    top1_label, top1_value = ranked[0]
    top2_label, top2_value = ranked[1]
    if top1_value - top2_value < 0.10:
        return "mixed"
    return top1_label


def _industry_terms_for_ticker(
    *,
    ticker: str,
    sub_industry_name: str | None,
) -> tuple[str, ...]:
    terms: list[str] = []
    if sub_industry_name:
        terms.append(sub_industry_name)

    profile = COMPANY_PROFILES.get(ticker.upper()) or {}
    for term in profile.get("industry", []):
        normalized = _string_or_none(term)
        if normalized:
            terms.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return tuple(deduped)


def _primary_industry_term(context: EpisodeContext) -> str | None:
    if context.sub_industry_name:
        return context.sub_industry_name
    if context.industry_terms:
        return context.industry_terms[0]
    return None


def _build_google_market_query(context: EpisodeContext) -> str:
    return (
        "stock market equities economy federal reserve "
        f"after:{context.episode_start} before:{context.episode_end}"
    )


def _build_google_industry_queries(context: EpisodeContext) -> tuple[str, ...]:
    queries: list[str] = []
    industry_term = _primary_industry_term(context)
    if industry_term:
        queries.append(
            f'"{industry_term}" stocks after:{context.episode_start} '
            f"before:{context.episode_end}"
        )
    if context.sector_name:
        queries.append(
            f'"{context.sector_name}" sector stocks after:{context.episode_start} '
            f"before:{context.episode_end}"
        )
    return tuple(queries)


def _fetch_stocknews_scoped_rows(
    *,
    episode_context: EpisodeContext,
    provider_context: ProviderContext,
) -> list[dict[str, Any]]:
    # DEC-201: on the $19.99 Basic tier, StockNewsAPI's only historical-usable
    # endpoint is the individual-ticker one (`/api/v1?tickers=X`). Both
    # `section=general` and `section=alltickers` require the Premium tier's
    # Historical Data feature (they return latest-only on Basic, or 403 when
    # combined with `date=`). Market-scope + industry-scope coverage therefore
    # comes from GoogleRSS date-windowed queries, not StockNewsAPI.
    return list(
        _fetch_stocknews_items(
            ticker=episode_context.ticker,
            date_from=episode_context.episode_start,
            date_to=episode_context.episode_end,
            stocknews_key=provider_context.stocknews_api_key,
            stocknews_base_url=provider_context.stocknews_base_url,
        )
    )


def _fetch_finnhub_scoped_rows(
    *,
    episode_context: EpisodeContext,
    provider_context: ProviderContext,
) -> list[dict[str, Any]]:
    rows = [
        *_fetch_finnhub_items(
            ticker=episode_context.ticker,
            date_from=episode_context.episode_start,
            date_to=episode_context.episode_end,
            finnhub_key=provider_context.finnhub_api_key,
        )
    ]

    mode = episode_context.retrieval_mode
    if mode in {"market", "mixed", "degraded"}:
        rows.extend(
            _fetch_finnhub_market_news_items(
                category="general",
                date_from=episode_context.episode_start,
                date_to=episode_context.episode_end,
                finnhub_key=provider_context.finnhub_api_key,
            )
        )

    # DEC-198 C: industry scope no longer uses Finnhub peer-company news.
    # Peer articles are firm news from other tickers, not industry coverage.
    # Real industry-level content comes from StockNewsAPI sector/industry
    # endpoints and GoogleRSS sector queries.
    return rows


def _fetch_google_scoped_rows(
    *,
    episode_context: EpisodeContext,
) -> list[dict[str, Any]]:
    rows = [
        *_fetch_google_rss_items(
            ticker=episode_context.ticker,
            company_name=episode_context.company_name,
            date_from=episode_context.episode_start,
            date_to=episode_context.episode_end,
        )
    ]

    mode = episode_context.retrieval_mode
    if mode in {"market", "mixed", "degraded"}:
        rows.extend(
            _fetch_google_rss_query_items(
                ticker=episode_context.ticker,
                query=_build_google_market_query(episode_context),
                date_from=episode_context.episode_start,
                date_to=episode_context.episode_end,
                scope_tag="market",
                scope_key="general",
            )
        )

    if mode in {"industry", "mixed", "degraded"}:
        for query in _build_google_industry_queries(episode_context):
            rows.extend(
                _fetch_google_rss_query_items(
                    ticker=episode_context.ticker,
                    query=query,
                    date_from=episode_context.episode_start,
                    date_to=episode_context.episode_end,
                    scope_tag="industry",
                    scope_key=query,
                )
            )

    return rows


__all__ = [
    "ArticleScorer",
    "CandidateArticle",
    "DEFAULT_CROSS_ENCODER_MODEL",
    "EpisodeContext",
    "ProviderContext",
    "RankedArticle",
    "build_episode_context_from_episode",
    "build_provider_context",
    "build_structured_episode_query",
    "fetch_candidate_articles",
    "rerank_candidate_articles",
    "run_retrieval_reranking",
    "score_candidate_articles_with_cross_encoder",
    "score_candidate_articles_with_default_cross_encoder",
]
