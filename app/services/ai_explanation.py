"""
AI Explanation service (DEC-203).

Parallel path to the V2 pipeline: delegates the qualitative half of an
episode analysis (news retrieval + synthesis + causal explanation) to a
single configured LLM provider call with web search enabled, mimicking the
ChatGPT consumer experience for prompts of the form
"why did $TICKER move on $DATE_RANGE".

Output shape (the parsed payload stored in the cache row):
  {
    "headline_explanation":  short 1-sentence summary,
    "full_explanation":      markdown prose with inline [n] citations,
    "evidence_quality":      "SUPPORTED" | "LIMITED" | "NONE",
    "sources": [
      {"title", "url", "publisher", "date"},
      ...
    ],
    "provider_name":         e.g. "openai",
    "model_name":            e.g. "gpt-5.5",
    "cost_estimate_usd":     float | null,
    "computed_at":           ISO-8601 string,
    "raw_response_path":     filesystem path to the raw provider response
                             (kept for capstone paper failure-mode analysis,
                             not displayed),
  }
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re

from app.cache import _normalize_for_json
from app.models import Episode, EpisodeAiExplanationCache, db
from app.services.ai_provider import (
    OPENAI_DEFAULT_MODEL,
    run_ai_provider,
)


DEFAULT_MODEL = OPENAI_DEFAULT_MODEL

# Where to drop the raw provider response for paper analysis (DEC-203).
_RAW_RESPONSE_DIR = (
    Path(__file__).resolve().parents[2] / "instance" / "ai_explanation_raw"
)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def get_or_compute_ai_explanation(
    episode: Episode,
    *,
    attribution_payload: dict[str, Any] | None = None,
    episode_metrics: dict[str, Any] | None = None,
    force_recompute: bool = False,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Return a cached AI Explanation payload, or compute and cache one.

    Cache TTL is indefinite — only `force_recompute=True` triggers a fresh
    paid API call.
    """
    if not force_recompute:
        cached = _load_cached(episode.id)
        if cached is not None:
            return cached

    payload = _compute_ai_explanation(
        episode=episode,
        attribution_payload=attribution_payload or {},
        episode_metrics=episode_metrics or {},
        provider_name=provider_name,
        model_name=model_name,
    )
    _save_cache(
        episode_id=episode.id,
        payload=payload,
        model_name=str(payload.get("model_name") or model_name or DEFAULT_MODEL),
    )
    return payload


def get_cached_ai_explanation(episode_id: int) -> dict[str, Any] | None:
    """Read-only accessor — returns None if no cache row exists."""
    return _load_cached(episode_id)


def cache_age_seconds(episode_id: int) -> float | None:
    """Seconds since the cached payload was computed; None if no cache."""
    row = EpisodeAiExplanationCache.query.filter_by(episode_id=episode_id).first()
    if row is None or row.created_at is None:
        return None
    now = datetime.utcnow()
    created = row.created_at
    if created.tzinfo is not None:
        created = created.replace(tzinfo=None)
    return max(0.0, (now - created).total_seconds())


# --------------------------------------------------------------------------- #
# Cache I/O
# --------------------------------------------------------------------------- #

def _load_cached(episode_id: int) -> dict[str, Any] | None:
    row = EpisodeAiExplanationCache.query.filter_by(episode_id=episode_id).first()
    if row is None:
        return None
    try:
        payload = json.loads(row.payload_json)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("provider_name", "openai")
    payload.setdefault("model_name", row.model_name)
    payload.setdefault("cost_estimate_usd", row.cost_estimate_usd)
    payload.setdefault("computed_at", row.created_at.isoformat() if row.created_at else None)
    payload.setdefault("raw_response_path", row.raw_response_path)
    return payload


def _save_cache(*, episode_id: int, payload: dict[str, Any], model_name: str) -> None:
    cost_raw = payload.get("cost_estimate_usd")
    try:
        cost = float(cost_raw) if cost_raw is not None else None
    except Exception:
        cost = None
    raw_path = payload.get("raw_response_path")
    payload_clean = _normalize_for_json({k: v for k, v in payload.items()})
    payload_json = json.dumps(payload_clean, ensure_ascii=False, indent=2)

    row = EpisodeAiExplanationCache.query.filter_by(episode_id=episode_id).first()
    if row is None:
        row = EpisodeAiExplanationCache(
            episode_id=episode_id,
            model_name=model_name,
            payload_json=payload_json,
            cost_estimate_usd=cost,
            raw_response_path=raw_path,
        )
        db.session.add(row)
    else:
        row.model_name = model_name
        row.payload_json = payload_json
        row.cost_estimate_usd = cost
        row.raw_response_path = raw_path
        row.created_at = datetime.utcnow()
    db.session.commit()


# --------------------------------------------------------------------------- #
# Provider call + parsing
# --------------------------------------------------------------------------- #

@dataclass
class _AttributionContext:
    firm_share: float | None
    industry_share: float | None
    market_share: float | None
    alpha_share: float | None
    industry_proxy_ticker: str | None
    quality_status: str | None
    near_zero: bool


def _compute_ai_explanation(
    *,
    episode: Episode,
    attribution_payload: dict[str, Any],
    episode_metrics: dict[str, Any],
    provider_name: str | None,
    model_name: str | None,
) -> dict[str, Any]:
    ctx = _attribution_context(attribution_payload)
    prompt = _build_prompt(episode=episode, ctx=ctx, episode_metrics=episode_metrics)

    provider_result = run_ai_provider(
        prompt,
        provider_name=provider_name,
        model_name=model_name,
    )
    raw_text = provider_result.output_text.strip()
    raw_path = _save_raw_response(
        episode_id=episode.id,
        provider_name=provider_result.provider_name,
        response=provider_result.raw_response,
    )

    parsed = _parse_response(raw_text)

    return {
        "headline_explanation": parsed["headline"],
        "full_explanation": parsed["full_explanation"],
        "evidence_quality": parsed["evidence_quality"],
        "sources": parsed["sources"],
        "provider_name": provider_result.provider_name,
        "model_name": provider_result.model_name,
        "cost_estimate_usd": provider_result.cost_estimate_usd,
        "computed_at": datetime.utcnow().isoformat(),
        "raw_response_path": str(raw_path) if raw_path else None,
    }


def _attribution_context(payload: dict[str, Any]) -> _AttributionContext:
    return _AttributionContext(
        firm_share=payload.get("firm_share"),
        industry_share=payload.get("industry_share"),
        market_share=payload.get("market_share"),
        alpha_share=payload.get("alpha_share"),
        industry_proxy_ticker=payload.get("industry_proxy_ticker"),
        quality_status=payload.get("quality_status"),
        near_zero=bool(payload.get("near_zero_episode_return")),
    )


def _build_prompt(
    *,
    episode: Episode,
    ctx: _AttributionContext,
    episode_metrics: dict[str, Any] | None = None,
) -> str:
    company = episode.company
    ticker = company.ticker if company else "?"
    name = company.name if company else "?"
    sector = (company.gics_sector or "?") if company else "?"
    sub = (company.gics_sub_industry or "?") if company else "?"

    start = episode.window_start.isoformat() if episode.window_start else "?"
    end = episode.window_end.isoformat() if episode.window_end else "?"

    attribution_block = _format_attribution_block(ctx)
    episode_block = _format_episode_block(
        episode=episode,
        episode_metrics=episode_metrics or {},
        ticker=ticker,
        name=name,
        sector=sector,
        sub=sub,
        start=start,
        end=end,
    )

    return f"""You are an equity research analyst writing a concise, evidence-grounded
explanation of a notable price move. Use the `web_search` tool freely to find
news, filings, regulatory actions, analyst notes, and macro context relevant to
the episode below. Cite every factual claim inline using bracketed numbers
[1], [2], ... that match the sources you list at the end.

EPISODE
{episode_block}

QUANTITATIVE ATTRIBUTION (from our internal factor regression)
{attribution_block}

YOUR TASK
1. Search the web for what actually happened to {ticker} during {start} → {end}.
2. Write a 2–4 paragraph causal explanation in plain English. Use the full
   daily path and rolling signal list when deciding whether this is one coherent
   event or a composite selloff/reversal episode.
3. Cover all three
   forces (firm-specific, industry, macro/market) that the attribution above
   identifies as material; do NOT force-fit a single narrative if the
   attribution shows multiple meaningful contributors. Cite inline.
4. If web evidence is sparse, partial, or contradicts the attribution, say so
   explicitly rather than papering over it.

OUTPUT FORMAT
First, write the markdown prose explanation (no leading heading, no preamble
about your reasoning process — just the analysis). Use inline [n] citations.

Then, on a new line, append a fenced JSON block with EXACTLY this shape:

```json
{{
  "headline": "one-sentence (≤ 25 words) summary of the dominant cause",
  "evidence_quality": "SUPPORTED" | "LIMITED" | "NONE",
  "sources": [
    {{"title": "...", "url": "https://...", "publisher": "...", "date": "YYYY-MM-DD"}}
  ]
}}
```

Rules for the JSON block:
- `evidence_quality` = SUPPORTED if you found multiple specific, dated,
  publisher-credible sources directly tied to the move; LIMITED if only
  partial or weak coverage; NONE if no relevant news was found.
- `sources` indices must match the inline [n] citations in the prose, in
  order. Include every source you cite.
- Use ISO YYYY-MM-DD for dates; use empty string if unknown.
"""


def _format_episode_block(
    *,
    episode: Episode,
    episode_metrics: dict[str, Any],
    ticker: str,
    name: str,
    sector: str,
    sub: str,
    start: str,
    end: str,
) -> str:
    """
    Build the prompt's episode metadata block.

    `Episode.pct_move` is the largest absolute rolling signal return inside the
    merged episode, not the whole-window cumulative return. Keep that distinction
    explicit so reversal episodes (for example NVDA#15) are not mislabeled.
    """
    stored_direction = episode.direction or "?"
    stored_peak_pct = _fmt_pct(episode.pct_move)

    cumulative_pct = episode_metrics.get("cumulative_pct")
    peak_pct = episode_metrics.get("peak_pct")
    peak_date = episode_metrics.get("peak") or (
        episode.peak_date.isoformat() if episode.peak_date else "?"
    )
    signal_anchors = episode_metrics.get("signal_anchors") or []
    daily_pct = episode_metrics.get("daily_pct") or []

    if peak_pct is None:
        peak_pct = episode.pct_move

    lines = [
        f"- ticker:                         {ticker}",
        f"- company:                        {name}",
        f"- gics sector:                    {sector}",
        f"- gics sub-industry:              {sub}",
        f"- window:                         {start} -> {end}",
        f"- stored direction:               {stored_direction} (direction of the peak rolling signal, not necessarily the whole window)",
        f"- episode-level cumulative return: {_fmt_pct(cumulative_pct)}",
        f"- peak rolling signal move:       {_fmt_pct(peak_pct)} on {peak_date}",
        f"- stored pct_move field meaning:  {stored_peak_pct} = largest absolute rolling signal return inside the merged episode",
        f"- signal count:                   {len(signal_anchors) if signal_anchors else episode.signal_count}",
    ]

    if signal_anchors:
        lines.append("- rolling signal anchors:")
        for signal in signal_anchors:
            lines.append(
                "  "
                f"- {signal.get('baseline_date') or '?'} -> {signal.get('anchor_date') or '?'}: "
                f"{_fmt_pct(signal.get('ret_pct'))} ({signal.get('direction') or '?'})"
            )
        directions = {
            str(signal.get("direction") or "").strip().lower()
            for signal in signal_anchors
            if signal.get("direction")
        }
        if len(directions) > 1:
            lines.append(
                "- episode path warning: contains opposite-direction rolling signals; "
                "treat it as a possible composite selloff/reversal episode rather "
                "than assuming one smooth directional move."
            )

    if daily_pct:
        lines.append("- daily close-to-close path inside the episode window:")
        for row in daily_pct:
            day_pct = row.get("pct")
            day_pct_text = "n/a" if day_pct is None else _fmt_pct(day_pct)
            close = row.get("close")
            close_text = "?" if close is None else str(close)
            lines.append(
                f"  - {row.get('date') or '?'}: {day_pct_text} "
                f"(close {close_text})"
            )

    return "\n".join(lines)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return str(value)


def _format_attribution_block(ctx: _AttributionContext) -> str:
    if ctx.quality_status == "OOS":
        return "- attribution: OOS (regression did not run; treat all three forces as unknown)"
    if ctx.near_zero:
        return "- attribution: near-zero net return; signed shares unstable, treat as informational"

    def fmt(label: str, val: float | None) -> str:
        if val is None:
            return f"- {label:18s} unknown"
        return f"- {label:18s} {val * 100:+.0f}% of episode return"

    lines = [
        fmt("firm-specific:", ctx.firm_share),
        fmt("industry:", ctx.industry_share),
        fmt("market (SPY):", ctx.market_share),
    ]
    if ctx.alpha_share is not None:
        lines.append(fmt("alpha (residual):", ctx.alpha_share))
    if ctx.industry_proxy_ticker:
        lines.append(f"- industry proxy:    {ctx.industry_proxy_ticker}")
    if ctx.quality_status:
        lines.append(f"- regression fit:    {ctx.quality_status}")
    return "\n".join(lines)


_JSON_BLOCK_RE = re.compile(
    r"```json\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _parse_response(raw_text: str) -> dict[str, Any]:
    match = _JSON_BLOCK_RE.search(raw_text)
    if match is None:
        return {
            "headline": "(model did not return a structured headline)",
            "full_explanation": raw_text,
            "evidence_quality": "LIMITED",
            "sources": [],
        }

    json_str = match.group(1)
    prose = (raw_text[: match.start()] + raw_text[match.end():]).strip()

    try:
        meta = json.loads(json_str)
    except Exception:
        meta = {}

    headline = (meta.get("headline") or "").strip() or "(headline unavailable)"
    quality_raw = (meta.get("evidence_quality") or "").strip().upper()
    if quality_raw not in {"SUPPORTED", "LIMITED", "NONE"}:
        quality_raw = "LIMITED"
    sources_raw = meta.get("sources") or []
    sources: list[dict[str, str]] = []
    if isinstance(sources_raw, list):
        for item in sources_raw:
            if not isinstance(item, dict):
                continue
            sources.append({
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "publisher": str(item.get("publisher") or "").strip(),
                "date": str(item.get("date") or "").strip(),
            })

    return {
        "headline": headline,
        "full_explanation": prose,
        "evidence_quality": quality_raw,
        "sources": sources,
    }


def _save_raw_response(
    *,
    episode_id: int,
    provider_name: str,
    response: Any,
) -> Path | None:
    """
    Drop the full provider response to disk for capstone paper failure-mode
    analysis. Per DEC-203 user request — sources list alone is not enough
    to audit which queries the model ran and which URLs it browsed.
    """
    try:
        _RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    payload: Any
    if isinstance(response, dict):
        payload = response
    else:
        try:
            payload = response.model_dump()
        except Exception:
            try:
                payload = response.to_dict()
            except Exception:
                payload = {
                    "output_text": getattr(response, "output_text", None),
                    "_note": "model_dump/to_dict unavailable on this SDK version",
                }
    if isinstance(payload, dict):
        payload.setdefault("_provider_name", provider_name)

    provider_slug = re.sub(r"[^a-z0-9_-]+", "_", provider_name.lower()).strip("_")
    path = _RAW_RESPONSE_DIR / f"episode_{episode_id}_{provider_slug or 'provider'}.json"
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(_normalize_for_json(payload), f, ensure_ascii=False, indent=2)
    except Exception:
        return None
    return path
