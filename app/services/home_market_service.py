from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
import json
import time

import requests

from app.config.settings import get_settings
from app.models import Company


FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
QUOTE_TTL_SECONDS = 180
ASSET_TTL_SECONDS = 60 * 60 * 24 * 30

_ASSET_CACHE_PATH = (
    Path(__file__).resolve().parents[2]
    / "instance"
    / "company_assets"
    / "company_assets.json"
)

_QUOTE_CACHE: dict[str, dict[str, Any]] = {}
_QUOTE_CACHE_TS = 0.0
_QUOTE_LOCK = Lock()
_ASSET_LOCK = Lock()


def get_home_market_cards(companies: list[Company]) -> dict[str, Any]:
    """
    Return optional home-card enrichment.

    Finnhub is the only runtime source used here. If it is unavailable or not
    configured, callers still receive one card per company with empty quote/logo
    fields so the homepage can render normally.
    """
    tickers = [company.ticker for company in companies]
    settings = get_settings()
    if not settings.finnhub_api_key:
        return {
            "ok": True,
            "source": None,
            "quote_ttl_seconds": QUOTE_TTL_SECONDS,
            "asset_ttl_seconds": ASSET_TTL_SECONDS,
            "cards": [_empty_card(company) for company in companies],
        }

    assets = _load_or_refresh_assets(companies, settings.finnhub_api_key)
    quotes = _load_or_refresh_quotes(tickers, settings.finnhub_api_key)

    cards = []
    for company in companies:
        ticker = company.ticker
        cards.append({
            "ticker": ticker,
            "quote": quotes.get(ticker),
            "asset": assets.get(ticker) or {},
        })

    return {
        "ok": True,
        "source": "Finnhub",
        "quote_ttl_seconds": QUOTE_TTL_SECONDS,
        "asset_ttl_seconds": ASSET_TTL_SECONDS,
        "cards": cards,
    }


def _empty_card(company: Company) -> dict[str, Any]:
    return {
        "ticker": company.ticker,
        "quote": None,
        "asset": {},
    }


def _load_or_refresh_quotes(tickers: list[str], api_key: str) -> dict[str, dict[str, Any]]:
    global _QUOTE_CACHE_TS

    now = time.monotonic()
    with _QUOTE_LOCK:
        if _QUOTE_CACHE and now - _QUOTE_CACHE_TS < QUOTE_TTL_SECONDS:
            return dict(_QUOTE_CACHE)

        refreshed: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            quote = _fetch_quote(ticker, api_key)
            if quote is not None:
                refreshed[ticker] = quote
        _QUOTE_CACHE.clear()
        _QUOTE_CACHE.update(refreshed)
        _QUOTE_CACHE_TS = now
        return dict(_QUOTE_CACHE)


def _load_or_refresh_assets(companies: list[Company], api_key: str) -> dict[str, dict[str, Any]]:
    with _ASSET_LOCK:
        cached = _read_asset_cache()
        changed = False
        now_epoch = time.time()

        for company in companies:
            ticker = company.ticker
            cached_item = cached.get(ticker)
            if _asset_is_fresh(cached_item, now_epoch):
                continue
            profile = _fetch_profile(ticker, api_key)
            cached[ticker] = {
                "logo_url": str((profile or {}).get("logo") or ""),
                "weburl": str((profile or {}).get("weburl") or ""),
                "source": "Finnhub" if profile else None,
                "updated_at": _utc_now_iso(),
            }
            changed = True

        if changed:
            _write_asset_cache(cached)
        return {
            company.ticker: cached.get(company.ticker, {})
            for company in companies
        }


def _fetch_quote(ticker: str, api_key: str) -> dict[str, Any] | None:
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/quote",
            params={"symbol": _finnhub_symbol(ticker), "token": api_key},
            timeout=6,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    try:
        current = float(payload.get("c") or 0.0)
    except Exception:
        current = 0.0
    if current <= 0:
        return None

    change = _optional_float(payload.get("d"))
    change_pct = _optional_float(payload.get("dp"))
    timestamp = _optional_int(payload.get("t"))
    return {
        "current": current,
        "change": change,
        "change_percent": change_pct,
        "previous_close": _optional_float(payload.get("pc")),
        "open": _optional_float(payload.get("o")),
        "high": _optional_float(payload.get("h")),
        "low": _optional_float(payload.get("l")),
        "timestamp": timestamp,
        "updated_at": _iso_from_epoch(timestamp) if timestamp else _utc_now_iso(),
        "source": "Finnhub",
    }


def _fetch_profile(ticker: str, api_key: str) -> dict[str, Any] | None:
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/stock/profile2",
            params={"symbol": _finnhub_symbol(ticker), "token": api_key},
            timeout=6,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload:
        return None
    return payload


def _asset_is_fresh(item: dict[str, Any] | None, now_epoch: float) -> bool:
    if not isinstance(item, dict):
        return False
    updated_at = item.get("updated_at")
    if not updated_at:
        return False
    try:
        parsed = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
    except Exception:
        return False
    return now_epoch - parsed.timestamp() < ASSET_TTL_SECONDS


def _read_asset_cache() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(_ASSET_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def _write_asset_cache(payload: dict[str, dict[str, Any]]) -> None:
    try:
        _ASSET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ASSET_CACHE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


def _finnhub_symbol(ticker: str) -> str:
    return ticker.replace("-", ".")


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _iso_from_epoch(value: int) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
