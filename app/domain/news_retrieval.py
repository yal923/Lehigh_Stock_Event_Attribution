from __future__ import annotations

from datetime import date
from email.utils import parsedate_to_datetime
import math
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import pandas as pd
import requests
from sentence_transformers import util as st_util

from app.config.constants import (
    COMPANY_PROFILES,
    FIN_KEYWORDS,
    QUERY_TEMPLATE,
    TICKER_TO_NAME_MAP,
    TIME_WEIGHTS,
    clean_company_name,
    fingerprint_news,
    get_publisher_score,
)
from app.config.settings import get_settings
from app.infrastructure.resources import get_embedding_model, get_nlp


W_SEMANTIC = 0.30
W_ENTITY = 0.20
W_TIME = 0.20
W_PUBLISHER = 0.15
W_KEYWORD = 0.15

SOURCE_SCORE_WEIGHTS = {
    "StockNewsAPI": 1.00,
    "Finnhub": 0.95,
    "GoogleRSS": 0.75,
}


def _flatten_profile_keywords(profile: dict) -> dict:
    org_names_flat = set(name.lower() for name in profile.get("org_names", []))

    people_flat = set()
    people_dict = profile.get("people", {})
    for key in people_dict:
        people_flat.update(person.lower() for person in people_dict[key])
        people_flat.update(
            person.split(" ")[-1].lower()
            for person in people_dict[key]
            if len(person.split(" ")) > 1
        )

    products_flat = set()
    products_flat.update(item.lower() for item in profile.get("products", []))
    products_flat.update(item.lower() for item in profile.get("brands_units", []))

    return {
        "org_names": org_names_flat,
        "people": people_flat,
        "products_brands": products_flat,
    }


FLATTENED_PROFILES = {
    ticker: _flatten_profile_keywords(profile)
    for ticker, profile in COMPANY_PROFILES.items()
}


def _calculate_entity_score(doc, flat_profile: dict) -> float:
    score = 0.0
    org_names_set = flat_profile.get("org_names", set())
    people_set = flat_profile.get("people", set())
    products_brands_set = flat_profile.get("products_brands", set())

    if not org_names_set and not people_set and not products_brands_set:
        return 0.0

    for ent in doc.ents:
        ent_text = ent.text.lower().strip()
        ent_label = ent.label_

        if ent_label == "ORG" and ent_text in org_names_set:
            score += 1.0
            continue
        if ent_label == "PERSON" and ent_text in people_set:
            score += 0.5
            continue
        if ent_text in products_brands_set:
            score += 0.5

    return score


def _safe_parse_published_date(raw_value) -> date | None:
    if raw_value is None:
        return None

    try:
        if isinstance(raw_value, (int, float)):
            if raw_value > 10_000_000_000:
                ts = pd.to_datetime(raw_value, unit="ms", utc=True)
            else:
                ts = pd.to_datetime(raw_value, unit="s", utc=True)
            return ts.date()

        value = str(raw_value).strip()
        if not value:
            return None

        try:
            return parsedate_to_datetime(value).date()
        except Exception:
            pass

        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def _build_provider_budgets(settings) -> dict[str, int | None]:
    ratio = settings.news_budget_ratio
    if ratio < 0:
        ratio = 0.0
    if ratio > 1:
        ratio = 1.0

    def _budget_from_remaining(remaining: int | None) -> int | None:
        if remaining is None:
            return None
        return max(0, int(remaining * ratio))

    return {
        "StockNewsAPI": _budget_from_remaining(settings.stocknews_remaining_quota),
        "Finnhub": _budget_from_remaining(settings.finnhub_remaining_quota),
    }


def _consume_provider_budget(provider: str, budgets: dict[str, int | None]) -> bool:
    remaining = budgets.get(provider)
    if remaining is None:
        return True
    if remaining <= 0:
        print(f"[{provider}] Request skipped by budget guardrail")
        return False
    budgets[provider] = remaining - 1
    return True


def _normalize_finnhub_items(ticker: str, items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in items or []:
        headline = (item.get("headline") or "").strip()
        summary = (item.get("summary") or "").strip()
        if not headline and not summary:
            continue

        out.append(
            {
                "tickers": ticker,
                "api_source": "Finnhub",
                "publisher": item.get("source"),
                "headline": headline,
                "summary": summary,
                "url": (item.get("url") or "").strip(),
                "published_at": _safe_parse_published_date(item.get("datetime")),
                "external_id": item.get("id"),
            }
        )
    return out


def _normalize_finnhub_items_with_scope(
    *,
    ticker: str,
    items: list[dict],
    scope_tag: str,
    scope_key: str | None = None,
) -> list[dict]:
    rows = _normalize_finnhub_items(ticker=ticker, items=items)
    for row in rows:
        row["scope_tag"] = scope_tag
        row["scope_key"] = scope_key
    return rows


def _fetch_stocknews_items(
    ticker: str,
    date_from: str,
    date_to: str,
    stocknews_key: str,
    stocknews_base_url: str,
) -> list[dict]:
    params = {
        "tickers": ticker,
        "items": 100,
        "sortby": "rank",
        "token": stocknews_key,
    }

    url = stocknews_base_url.rstrip("/")
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"[StockNewsAPI] Request failed for {ticker}: {e}")
        return []

    raw_items = payload.get("data") if isinstance(payload, dict) else None
    if raw_items is None and isinstance(payload, dict):
        raw_items = payload.get("news")
    if raw_items is None:
        raw_items = payload if isinstance(payload, list) else []

    out = _normalize_stocknews_items(
        raw_items or [],
        scope_tag="company",
        scope_key=ticker,
        tickers=ticker,
    )

    return _filter_rows_by_date_window(out, date_from=date_from, date_to=date_to)


def _filter_rows_by_date_window(
    rows: list[dict],
    *,
    date_from: str,
    date_to: str,
) -> list[dict]:
    start = _safe_parse_published_date(date_from)
    end = _safe_parse_published_date(date_to)
    if start is None or end is None:
        return rows

    filtered = [
        row for row in rows
        if row.get("published_at") is not None and start <= row["published_at"] <= end
    ]
    return filtered


def _fetch_finnhub_items(
    ticker: str,
    date_from: str,
    date_to: str,
    finnhub_key: str,
    *,
    scope_tag: str = "company",
    scope_key: str | None = None,
) -> list[dict]:
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": date_from,
        "to": date_to,
        "token": finnhub_key,
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"[Finnhub] Request failed for {ticker}: {e}")
        return []

    return _normalize_finnhub_items_with_scope(
        ticker=ticker,
        items=payload or [],
        scope_tag=scope_tag,
        scope_key=scope_key or ticker,
    )


def _fetch_finnhub_market_news_items(
    *,
    category: str,
    date_from: str,
    date_to: str,
    finnhub_key: str,
) -> list[dict]:
    url = "https://finnhub.io/api/v1/news"
    params = {
        "category": category,
        "token": finnhub_key,
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"[Finnhub] Market news request failed for category={category}: {e}")
        return []

    rows = _normalize_finnhub_items_with_scope(
        ticker="__market__",
        items=payload or [],
        scope_tag="market",
        scope_key=category,
    )
    return _filter_rows_by_date_window(rows, date_from=date_from, date_to=date_to)


def _fetch_google_rss_items(
    ticker: str,
    company_name: str,
    date_from: str,
    date_to: str,
) -> list[dict]:
    q = f'{company_name} {ticker} after:{date_from} before:{date_to}'
    return _fetch_google_rss_query_items(
        ticker=ticker,
        query=q,
        date_from=date_from,
        date_to=date_to,
        scope_tag="company",
        scope_key=ticker,
    )


def _fetch_google_rss_query_items(
    *,
    ticker: str,
    query: str,
    date_from: str,
    date_to: str,
    scope_tag: str,
    scope_key: str | None = None,
) -> list[dict]:
    url = (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception as e:
        print(f"[GoogleRSS] Request failed for {ticker}: {e}")
        return []

    out: list[dict] = []
    for item in root.findall("./channel/item"):
        headline = (item.findtext("title") or "").strip()
        summary = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = item.findtext("source")
        if not headline and not summary:
            continue
        out.append(
            {
                "tickers": ticker,
                "api_source": "GoogleRSS",
                "publisher": source,
                "headline": headline,
                "summary": summary,
                "url": link,
                "published_at": _safe_parse_published_date(item.findtext("pubDate")),
                "external_id": link,
                "scope_tag": scope_tag,
                "scope_key": scope_key,
            }
        )

    return out


def _normalize_stocknews_items(
    items: list[dict],
    *,
    scope_tag: str,
    scope_key: str | None,
    tickers: str,
) -> list[dict]:
    out: list[dict] = []
    for item in items or []:
        headline = (item.get("title") or item.get("headline") or "").strip()
        summary = (
            item.get("text")
            or item.get("summary")
            or item.get("snippet")
            or ""
        ).strip()
        if not headline and not summary:
            continue

        out.append(
            {
                "tickers": tickers,
                "api_source": "StockNewsAPI",
                "publisher": item.get("source_name") or item.get("source"),
                "headline": headline,
                "summary": summary,
                "url": (item.get("news_url") or item.get("url") or "").strip(),
                "published_at": _safe_parse_published_date(
                    item.get("date")
                    or item.get("published_at")
                    or item.get("published")
                ),
                "external_id": item.get("id") or item.get("uuid"),
                "scope_tag": scope_tag,
                "scope_key": scope_key,
            }
        )

    return out


def _deduplicate_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []

    for row in rows:
        fp = fingerprint_news(row.get("url", ""), row.get("headline", ""))
        if fp in seen:
            continue
        seen.add(fp)
        clean_row = dict(row)
        clean_row["fingerprint"] = fp
        deduped.append(clean_row)

    return deduped


def fetch_and_score_news(
    ticker: str,
    date_from: str,
    date_to: str,
    peak_date: str,
    movement_direction: str,
    attribution_context: dict | None = None,
):
    """Fetch company news from multiple sources and compute unified features."""
    del attribution_context

    settings = get_settings()
    finnhub_key = settings.finnhub_api_key
    stocknews_key = settings.stocknews_api_key
    embedding_model = get_embedding_model()

    if embedding_model is None:
        print("Configuration Error: embedding model missing.")
        return pd.DataFrame([]), None

    v_event = None
    company_name = clean_company_name(TICKER_TO_NAME_MAP.get(ticker, ticker))
    event_template = QUERY_TEMPLATE.get(movement_direction)

    if event_template:
        try:
            event_query = event_template.format(company_name)
            print(f"    > Event Query: {event_query}")
            v_event = embedding_model.encode(event_query, convert_to_tensor=True)
        except Exception as e:
            print(f"    ! Error encoding event query: {e}")

    provider_budgets = _build_provider_budgets(settings)
    all_rows: list[dict] = []

    if stocknews_key and _consume_provider_budget("StockNewsAPI", provider_budgets):
        all_rows.extend(
            _fetch_stocknews_items(
                ticker=ticker,
                date_from=date_from,
                date_to=date_to,
                stocknews_key=stocknews_key,
                stocknews_base_url=settings.stocknews_base_url,
            )
        )

    if finnhub_key and _consume_provider_budget("Finnhub", provider_budgets):
        all_rows.extend(
            _fetch_finnhub_items(
                ticker=ticker,
                date_from=date_from,
                date_to=date_to,
                finnhub_key=finnhub_key,
            )
        )

    if settings.enable_google_rss:
        all_rows.extend(
            _fetch_google_rss_items(
                ticker=ticker,
                company_name=company_name,
                date_from=date_from,
                date_to=date_to,
            )
        )

    rows_dedup = _deduplicate_rows(all_rows)
    if not rows_dedup:
        return pd.DataFrame([]), None

    news_texts = [
        ((item.get("headline") or "") + " " + (item.get("summary") or "")).strip()
        for item in rows_dedup
    ]
    n_items = len(rows_dedup)

    embedding_matrix = None
    semantic_scores = [0.0] * n_items

    try:
        v_news_tensor = embedding_model.encode(news_texts, convert_to_tensor=True)
        embedding_matrix = v_news_tensor.cpu().numpy()
    except Exception as e:
        print(f"    ! Embedding generation failed: {e}")
        return pd.DataFrame([]), None

    if v_event is not None and n_items > 0:
        try:
            similarities = st_util.cos_sim(v_event, v_news_tensor)[0]
            semantic_scores = similarities.cpu().numpy().tolist()
        except Exception as e:
            print(f"    ! Semantic similarity failed: {e}")

    entity_scores = [0.0] * n_items
    profile = FLATTENED_PROFILES.get(ticker.upper())
    nlp = get_nlp()

    if profile and nlp is not None:
        entity_scores = [
            math.tanh(_calculate_entity_score(doc, profile))
            for doc in nlp.pipe(news_texts)
        ]
    else:
        if not profile:
            print(f"    > No entity profile for {ticker}")
        if nlp is None:
            print("    > spaCy model not available, entity scores set to 0.0")

    rows = []
    peak_dt = pd.Timestamp(peak_date)

    for i, item in enumerate(rows_dedup):
        headline = item.get("headline", "") or ""
        summary = item.get("summary", "") or ""
        url_item = item.get("url", "") or ""

        time_prox = 0.0
        published_dt_obj = item.get("published_at")

        if published_dt_obj is not None:
            delta_days = (pd.Timestamp(published_dt_obj) - peak_dt).days
            time_prox = TIME_WEIGHTS.get(delta_days, 0.0)

        semantic = semantic_scores[i] or 0.0
        entity = entity_scores[i] or 0.0
        publisher_name = item.get("publisher")
        publisher = get_publisher_score(publisher_name) or 0.0
        keyword_flag = any(k in (headline + " " + summary).lower() for k in FIN_KEYWORDS)
        source_weight = SOURCE_SCORE_WEIGHTS.get(item.get("api_source"), 0.90)

        final_score = (
            semantic * W_SEMANTIC
            + entity * W_ENTITY
            + time_prox * W_TIME
            + publisher * W_PUBLISHER
            + (1 if keyword_flag else 0) * W_KEYWORD
        ) * source_weight

        rows.append(
            {
                "tickers": ticker,
                "api_source": item.get("api_source"),
                "publisher": publisher_name,
                "headline": headline,
                "summary": summary,
                "url": url_item,
                "published_at": published_dt_obj,
                "finnhub_id": item.get("external_id") if item.get("api_source") == "Finnhub" else None,
                "fingerprint": item.get("fingerprint"),
                "keyword_related": keyword_flag,
                "semantic_similarity": semantic,
                "time_proximity": time_prox,
                "publisher_score": publisher,
                "entity_match_score": entity,
                "source_weight": source_weight,
                "final_score": final_score,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, embedding_matrix

    df = df.sort_values("final_score", ascending=False)
    if embedding_matrix is not None:
        order = df.index.to_numpy()
        embedding_matrix = embedding_matrix[order]

    df = df.reset_index(drop=True)
    return df, embedding_matrix
