from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Any, Sequence
from uuid import uuid4

from app.domain.news_v2_01_retrieval import RankedArticle
from app.infrastructure.llm import get_openai_client


DEFAULT_CLAIM_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_EXTRACTION_METHOD = "openai-structured-claim-extraction"


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    article_id: str
    source_url: str
    publisher: str | None
    published_at: Any
    claim_text: str
    evidence_span: str
    source_section: str
    entity_mentions: tuple[str, ...]
    claimed_event_time_text: str | None
    directionality_hint: str | None
    magnitude_text: str | None
    extraction_confidence: str
    grounding_confidence: str
    extraction_method: str
    decontextualized_claim_text: str | None = None
    claim_notes: tuple[str, ...] = ()


def extract_claims_from_ranked_articles(
    articles: Sequence[RankedArticle],
    *,
    max_claims_per_article: int = 3,
    model_name: str = DEFAULT_CLAIM_MODEL,
) -> list[ClaimRecord]:
    claims: list[ClaimRecord] = []
    for article in articles:
        claims.extend(
            extract_claims_from_ranked_article(
                article,
                max_claims=max_claims_per_article,
                model_name=model_name,
            )
        )
    return claims


def extract_claims_from_ranked_article(
    article: RankedArticle,
    *,
    max_claims: int = 3,
    model_name: str = DEFAULT_CLAIM_MODEL,
) -> list[ClaimRecord]:
    prompt = _build_claim_extraction_prompt(article=article, max_claims=max_claims)
    client = get_openai_client()
    response = client.responses.create(
        model=model_name,
        input=prompt,
        max_output_tokens=1600,
    )
    payload = _parse_claims_payload(response.output_text)
    return _claim_records_from_payload(
        article=article,
        payload=payload,
    )


def claim_record_to_dict(claim: ClaimRecord) -> dict[str, Any]:
    data = asdict(claim)
    data["entity_mentions"] = list(claim.entity_mentions)
    data["claim_notes"] = list(claim.claim_notes)
    return data


def _build_claim_extraction_prompt(
    *,
    article: RankedArticle,
    max_claims: int,
) -> str:
    summary = article.summary_or_snippet.strip() or "(none)"
    publisher = article.publisher or "(unknown)"
    published_at = (
        article.published_at.isoformat()
        if hasattr(article.published_at, "isoformat")
        else str(article.published_at)
    )

    return f"""
You are a financial-news claim extraction system.

Task:
- Read the ranked article below.
- Extract at most {max_claims} atomic claims that are most relevant to explaining the episode.
- Each claim must be grounded in explicit article text.
- Prefer claims that describe a discrete event, action, announcement, transaction, filing, approval, legal step, operational incident, guidance update, earnings result, analyst action, or market-moving development.
- Do not extract generic market commentary, thematic observations, investment opinions, valuation views, growth outlook, or broad business background unless the article states a concrete event.
- Use headline and summary/snippet only. Do not invent facts not present in the article text.
- If a claim would be unclear outside the original wording, provide a semantically equivalent `decontextualized_claim_text`.
- If the article mainly contains commentary rather than concrete events, return fewer claims or return an empty list.

Priority rules:
1. Prefer claims that answer "what happened".
2. Prefer company-specific events over generic sector commentary.
3. Prefer explicit announced facts over inference or analysis.
4. Avoid duplicating near-identical claims from the same article.

Good examples of claims:
- "Eli Lilly agreed to acquire Centessa Pharmaceuticals."
- "The FDA approved the company's drug."
- "The company raised full-year guidance."

Bad examples of claims:
- "The company has strong long-term growth potential."
- "The stock looks attractive after the recent dip."
- "The market may continue to reward obesity-drug leaders."

Return JSON only, with this exact top-level shape:
{{
  "claims": [
    {{
      "claim_text": "string",
      "evidence_span": "string",
      "source_section": "headline" | "summary",
      "entity_mentions": ["string"],
      "claimed_event_time_text": "string or null",
      "directionality_hint": "positive | negative | neutral | mixed | null",
      "magnitude_text": "string or null",
      "extraction_confidence": "high | medium | low",
      "grounding_confidence": "high | medium | low",
      "decontextualized_claim_text": "string or null",
      "claim_notes": ["string"]
    }}
  ]
}}

Episode query:
{article.episode_query}

Article metadata:
- article_id: {article.article_id}
- ticker: {article.ticker}
- provider: {article.provider}
- publisher: {publisher}
- published_at: {published_at}
- url: {article.url}

Article text:
[headline]
{article.headline}

[summary]
{summary}
""".strip()


def _parse_claims_payload(text: str | None) -> dict[str, Any]:
    if not text:
        return {"claims": []}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse claim extraction JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Claim extraction payload must be a JSON object")

    claims = payload.get("claims")
    if claims is None:
        return {"claims": []}
    if not isinstance(claims, list):
        raise ValueError("Claim extraction payload field 'claims' must be a list")
    return payload


def _claim_records_from_payload(
    *,
    article: RankedArticle,
    payload: dict[str, Any],
) -> list[ClaimRecord]:
    claims_payload = payload.get("claims", [])
    results: list[ClaimRecord] = []

    for raw_claim in claims_payload:
        if not isinstance(raw_claim, dict):
            continue

        claim_text = _clean_text(raw_claim.get("claim_text"))
        evidence_span = _clean_text(raw_claim.get("evidence_span"))
        if not claim_text or not evidence_span:
            continue

        source_section = _normalize_source_section(raw_claim.get("source_section"))
        entity_mentions = _normalize_str_list(raw_claim.get("entity_mentions"))
        claim_notes = _normalize_str_list(raw_claim.get("claim_notes"))

        results.append(
            ClaimRecord(
                claim_id=f"{article.article_id}-claim-{uuid4().hex[:8]}",
                article_id=article.article_id,
                source_url=article.url,
                publisher=article.publisher,
                published_at=article.published_at,
                claim_text=claim_text,
                evidence_span=evidence_span,
                source_section=source_section,
                entity_mentions=tuple(entity_mentions),
                claimed_event_time_text=_optional_text(
                    raw_claim.get("claimed_event_time_text")
                ),
                directionality_hint=_optional_directionality(
                    raw_claim.get("directionality_hint")
                ),
                magnitude_text=_optional_text(raw_claim.get("magnitude_text")),
                extraction_confidence=_normalize_confidence(
                    raw_claim.get("extraction_confidence")
                ),
                grounding_confidence=_normalize_confidence(
                    raw_claim.get("grounding_confidence")
                ),
                extraction_method=DEFAULT_EXTRACTION_METHOD,
                decontextualized_claim_text=_optional_text(
                    raw_claim.get("decontextualized_claim_text")
                ),
                claim_notes=tuple(claim_notes),
            )
        )

    return results


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    text = _clean_text(value)
    return text or None


def _normalize_confidence(value: Any) -> str:
    raw = _clean_text(value).lower()
    if raw in {"high", "medium", "low"}:
        return raw
    return "medium"


def _normalize_source_section(value: Any) -> str:
    raw = _clean_text(value).lower()
    if raw in {"headline", "summary", "snippet", "full_text"}:
        return raw
    return "summary"


def _optional_directionality(value: Any) -> str | None:
    raw = _clean_text(value).lower()
    if raw in {"positive", "negative", "neutral", "mixed"}:
        return raw
    return None


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text:
            cleaned.append(text)
    return cleaned


__all__ = [
    "ClaimRecord",
    "DEFAULT_CLAIM_MODEL",
    "claim_record_to_dict",
    "extract_claims_from_ranked_article",
    "extract_claims_from_ranked_articles",
]
