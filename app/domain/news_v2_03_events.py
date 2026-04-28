from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
import re
from typing import Any, Sequence
from uuid import uuid4

from app.domain.news_v2_02_claims import ClaimRecord
from app.infrastructure.llm import get_openai_client


DEFAULT_EVENT_MODEL = "gpt-4.1-2025-04-14"
ONTOLOGY_VERSION = "v0.1-minimal-finance"

EVENT_TYPE_TO_FAMILY = {
    "earnings_result": "financial_reporting",
    "guidance_update": "financial_reporting",
    "mna_or_strategic_transaction": "corporate_action",
    "capital_return_action": "corporate_action",
    "capital_raise_or_balance_sheet_action": "corporate_action",
    "management_or_governance_change": "corporate_action",
    "product_or_customer_event": "product_commercial",
    "regulatory_approval": "regulation_legal",
    "regulatory_or_legal_action": "regulation_legal",
    "operations_or_security_incident": "operations_incident",
    "analyst_or_investor_view_change": "market_external",
    "macro_or_sector_shock": "market_external",
    "other": "market_external",
    "unknown": "market_external",
}

EVENT_TYPE_TO_LABEL = {
    "earnings_result": "Earnings Result",
    "guidance_update": "Guidance Update",
    "mna_or_strategic_transaction": "M&A or Strategic Transaction",
    "capital_return_action": "Capital Return Action",
    "capital_raise_or_balance_sheet_action": "Capital Raise or Balance Sheet Action",
    "management_or_governance_change": "Management or Governance Change",
    "product_or_customer_event": "Product or Customer Event",
    "regulatory_approval": "Regulatory Approval",
    "regulatory_or_legal_action": "Regulatory or Legal Action",
    "operations_or_security_incident": "Operations or Security Incident",
    "analyst_or_investor_view_change": "Analyst or Investor View Change",
    "macro_or_sector_shock": "Macro or Sector Shock",
    "other": "Other",
    "unknown": "Unknown",
}

RULES: tuple[dict[str, Any], ...] = (
    {
        "canonical_event_type": "mna_or_strategic_transaction",
        "keywords": (
            "acquire",
            "acquisition",
            "buyout",
            "merger",
            "deal",
            "purchase",
            "strategic transaction",
        ),
        "confidence": "high",
        "label": "M&A or Strategic Transaction",
    },
    {
        "canonical_event_type": "earnings_result",
        "keywords": (
            "earnings",
            "quarterly results",
            "reported results",
            "beat estimates",
            "missed estimates",
        ),
        "confidence": "medium",
        "label": "Earnings Result",
    },
    {
        "canonical_event_type": "guidance_update",
        "keywords": (
            "guidance",
            "outlook",
            "forecast",
            "raised guidance",
            "cut guidance",
        ),
        "confidence": "medium",
        "label": "Guidance Update",
    },
    {
        "canonical_event_type": "regulatory_approval",
        "keywords": (
            "approved",
            "approval",
            "clearance",
            "authorized",
        ),
        "confidence": "medium",
        "label": "Regulatory Approval",
    },
    {
        "canonical_event_type": "regulatory_or_legal_action",
        "keywords": (
            "lawsuit",
            "sued",
            "investigation",
            "settlement",
            "penalty",
            "fine",
            "regulator",
        ),
        "confidence": "medium",
        "label": "Regulatory or Legal Action",
    },
    {
        "canonical_event_type": "operations_or_security_incident",
        "keywords": (
            "outage",
            "breach",
            "cyberattack",
            "recall",
            "disruption",
            "incident",
            "shutdown",
        ),
        "confidence": "medium",
        "label": "Operations or Security Incident",
    },
    {
        "canonical_event_type": "capital_return_action",
        "keywords": (
            "buyback",
            "repurchase",
            "dividend",
        ),
        "confidence": "medium",
        "label": "Capital Return Action",
    },
    {
        "canonical_event_type": "capital_raise_or_balance_sheet_action",
        "keywords": (
            "offering",
            "debt sale",
            "bond sale",
            "capital raise",
            "issued shares",
            "share sale",
        ),
        "confidence": "medium",
        "label": "Capital Raise or Balance Sheet Action",
    },
    {
        "canonical_event_type": "management_or_governance_change",
        "keywords": (
            "ceo",
            "cfo",
            "chairman",
            "board",
            "leadership",
            "resigned",
            "appointed",
        ),
        "confidence": "medium",
        "label": "Management or Governance Change",
    },
    {
        "canonical_event_type": "product_or_customer_event",
        "keywords": (
            "launch",
            "product",
            "customer",
            "contract",
            "partnership",
            "commercial agreement",
        ),
        "confidence": "medium",
        "label": "Product or Customer Event",
    },
    {
        "canonical_event_type": "analyst_or_investor_view_change",
        "keywords": (
            "upgraded",
            "downgraded",
            "price target",
            "analyst",
            "rating",
        ),
        "confidence": "medium",
        "label": "Analyst or Investor View Change",
    },
    {
        "canonical_event_type": "macro_or_sector_shock",
        "keywords": (
            "tariff",
            "inflation",
            "interest rate",
            "fed",
            "macro",
            "sector",
            "market-wide",
        ),
        "confidence": "medium",
        "label": "Macro or Sector Shock",
    },
)


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    canonical_event_type: str
    event_family: str
    canonical_label: str
    ontology_version: str
    primary_entity: str | None
    related_entities: tuple[str, ...]
    event_time_start: str | None
    event_time_end: str | None
    event_time_precision: str
    polarity: str | None
    magnitude_bucket: str | None
    event_arguments: dict[str, Any]
    supporting_claim_ids: tuple[str, ...]
    supporting_article_ids: tuple[str, ...]
    normalization_confidence: str
    normalization_method: str
    normalization_notes: tuple[str, ...] = ()
    fallback_reason: str | None = None


def normalize_claims_to_events(
    claims: Sequence[ClaimRecord],
    *,
    model_name: str = DEFAULT_EVENT_MODEL,
) -> list[EventRecord]:
    return [
        normalize_claim_to_event(claim, model_name=model_name)
        for claim in claims
    ]


def normalize_claim_to_event(
    claim: ClaimRecord,
    *,
    model_name: str = DEFAULT_EVENT_MODEL,
) -> EventRecord:
    rule_result = _rule_map_claim(claim)
    if _should_use_llm_fallback(rule_result):
        return _normalize_claim_with_llm_fallback(
            claim,
            rule_result=rule_result,
            model_name=model_name,
        )
    return _event_record_from_rule_result(claim, rule_result)


def event_record_to_dict(event: EventRecord) -> dict[str, Any]:
    data = asdict(event)
    data["related_entities"] = list(event.related_entities)
    data["supporting_claim_ids"] = list(event.supporting_claim_ids)
    data["supporting_article_ids"] = list(event.supporting_article_ids)
    data["normalization_notes"] = list(event.normalization_notes)
    return data


def _rule_map_claim(claim: ClaimRecord) -> dict[str, Any]:
    text = _claim_text_for_matching(claim)
    for rule in RULES:
        if any(keyword in text for keyword in rule["keywords"]):
            event_type = rule["canonical_event_type"]
            return {
                "canonical_event_type": event_type,
                "event_family": EVENT_TYPE_TO_FAMILY[event_type],
                "canonical_label": rule["label"],
                "normalization_confidence": rule["confidence"],
                "normalization_method": "rule",
                "fallback_reason": None,
                "normalization_notes": (
                    f"Matched rule keywords for {event_type}",
                ),
            }

    return {
        "canonical_event_type": "unknown",
        "event_family": EVENT_TYPE_TO_FAMILY["unknown"],
        "canonical_label": EVENT_TYPE_TO_LABEL["unknown"],
        "normalization_confidence": "low",
        "normalization_method": "rule",
        "fallback_reason": "rule_miss",
        "normalization_notes": ("No rule match",),
    }


def _should_use_llm_fallback(rule_result: dict[str, Any]) -> bool:
    return (
        rule_result["canonical_event_type"] in {"other", "unknown"}
        or rule_result["normalization_confidence"] == "low"
    )


def _normalize_claim_with_llm_fallback(
    claim: ClaimRecord,
    *,
    rule_result: dict[str, Any],
    model_name: str,
) -> EventRecord:
    prompt = _build_event_typing_prompt(claim=claim, rule_result=rule_result)
    client = get_openai_client()
    response = client.responses.create(
        model=model_name,
        input=prompt,
        max_output_tokens=1200,
    )
    payload = _parse_event_payload(response.output_text)
    return _event_record_from_llm_payload(
        claim=claim,
        payload=payload,
        fallback_reason=rule_result.get("fallback_reason") or "low_confidence_rule",
    )


def _build_event_typing_prompt(
    *,
    claim: ClaimRecord,
    rule_result: dict[str, Any],
) -> str:
    allowed_types = ", ".join(sorted(EVENT_TYPE_TO_FAMILY))
    return f"""
You are a financial-event normalization system.

Task:
- Map the grounded claim below into the approved SA41 event ontology.
- Choose exactly one `canonical_event_type` from the allowed set.
- Preserve uncertainty instead of overclaiming.
- If the claim does not fit a known type, use `other` or `unknown`.
- Do not force broad performance commentary, long-term growth observations, valuation narratives, or generic business momentum into a concrete event type.
- If the text describes performance or outlook without a discrete event, prefer `other` or `unknown`.
- `canonical_label` must be a short event label, not a full sentence and not a paraphrase of the claim.

Allowed canonical_event_type values:
{allowed_types}

Return JSON only with this exact shape:
{{
  "canonical_event_type": "string",
  "canonical_label": "string",
  "primary_entity": "string or null",
  "related_entities": ["string"],
  "event_time_start": "YYYY-MM-DD or null",
  "event_time_end": "YYYY-MM-DD or null",
  "event_time_precision": "known | inferred | unknown",
  "polarity": "positive | negative | neutral | mixed | null",
  "magnitude_bucket": "large | medium | small | unknown | null",
  "event_arguments": {{}},
  "normalization_confidence": "high | medium | low",
  "normalization_notes": ["string"],
  "fallback_reason": "string or null"
}}

Good canonical_label examples:
- "M&A or Strategic Transaction"
- "Guidance Update"
- "Regulatory Approval"
- "Other"

Bad canonical_label examples:
- "Eli Lilly's obesity portfolio experienced explosive growth in fiscal 2025"
- "The company continued to benefit from strong long-term demand"

Rule-first preclassification:
- canonical_event_type: {rule_result['canonical_event_type']}
- event_family: {rule_result['event_family']}
- confidence: {rule_result['normalization_confidence']}
- fallback_reason: {rule_result['fallback_reason']}

Claim:
- claim_text: {claim.claim_text}
- evidence_span: {claim.evidence_span}
- entity_mentions: {list(claim.entity_mentions)}
- claimed_event_time_text: {claim.claimed_event_time_text}
- directionality_hint: {claim.directionality_hint}
- magnitude_text: {claim.magnitude_text}
- decontextualized_claim_text: {claim.decontextualized_claim_text}
""".strip()


def _parse_event_payload(text: str | None) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty event-normalization response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Event-normalization payload must be a JSON object")
    return payload


def _event_record_from_rule_result(
    claim: ClaimRecord,
    rule_result: dict[str, Any],
) -> EventRecord:
    event_type = rule_result["canonical_event_type"]
    return EventRecord(
        event_id=f"{claim.claim_id}-event-{uuid4().hex[:8]}",
        canonical_event_type=event_type,
        event_family=rule_result["event_family"],
        canonical_label=rule_result["canonical_label"],
        ontology_version=ONTOLOGY_VERSION,
        primary_entity=_primary_entity_from_claim(claim),
        related_entities=_related_entities_from_claim(claim),
        event_time_start=_published_date_iso(claim.published_at),
        event_time_end=_published_date_iso(claim.published_at),
        event_time_precision=_time_precision_from_claim(claim),
        polarity=claim.directionality_hint,
        magnitude_bucket=_magnitude_bucket_from_claim(claim),
        event_arguments={
            "claim_text": claim.claim_text,
            "evidence_span": claim.evidence_span,
            "claimed_event_time_text": claim.claimed_event_time_text,
            "magnitude_text": claim.magnitude_text,
        },
        supporting_claim_ids=(claim.claim_id,),
        supporting_article_ids=(claim.article_id,),
        normalization_confidence=rule_result["normalization_confidence"],
        normalization_method=rule_result["normalization_method"],
        normalization_notes=tuple(rule_result["normalization_notes"]),
        fallback_reason=rule_result["fallback_reason"],
    )


def _event_record_from_llm_payload(
    *,
    claim: ClaimRecord,
    payload: dict[str, Any],
    fallback_reason: str,
) -> EventRecord:
    event_type = _valid_event_type(payload.get("canonical_event_type"))
    return EventRecord(
        event_id=f"{claim.claim_id}-event-{uuid4().hex[:8]}",
        canonical_event_type=event_type,
        event_family=EVENT_TYPE_TO_FAMILY[event_type],
        canonical_label=_clean_text(payload.get("canonical_label")) or EVENT_TYPE_TO_LABEL[event_type],
        ontology_version=ONTOLOGY_VERSION,
        primary_entity=_optional_text(payload.get("primary_entity")) or _primary_entity_from_claim(claim),
        related_entities=tuple(_normalize_str_list(payload.get("related_entities"))) or _related_entities_from_claim(claim),
        event_time_start=_optional_text(payload.get("event_time_start")) or _published_date_iso(claim.published_at),
        event_time_end=_optional_text(payload.get("event_time_end")) or _published_date_iso(claim.published_at),
        event_time_precision=_valid_time_precision(payload.get("event_time_precision")),
        polarity=_optional_polarity(payload.get("polarity")) or claim.directionality_hint,
        magnitude_bucket=_valid_magnitude_bucket(payload.get("magnitude_bucket")) or _magnitude_bucket_from_claim(claim),
        event_arguments=_normalized_event_arguments(payload.get("event_arguments"), claim),
        supporting_claim_ids=(claim.claim_id,),
        supporting_article_ids=(claim.article_id,),
        normalization_confidence=_valid_confidence(payload.get("normalization_confidence")),
        normalization_method="rule+llm_fallback",
        normalization_notes=tuple(_normalize_str_list(payload.get("normalization_notes"))),
        fallback_reason=_optional_text(payload.get("fallback_reason")) or fallback_reason,
    )


def _claim_text_for_matching(claim: ClaimRecord) -> str:
    return " ".join(
        part.lower()
        for part in (
            claim.claim_text,
            claim.evidence_span,
            claim.decontextualized_claim_text or "",
        )
        if part
    )


def _primary_entity_from_claim(claim: ClaimRecord) -> str | None:
    if claim.entity_mentions:
        return claim.entity_mentions[0]
    return None


def _related_entities_from_claim(claim: ClaimRecord) -> tuple[str, ...]:
    if len(claim.entity_mentions) <= 1:
        return ()
    return tuple(claim.entity_mentions[1:])


def _published_date_iso(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return None
    return None


def _time_precision_from_claim(claim: ClaimRecord) -> str:
    if claim.claimed_event_time_text:
        return "inferred"
    if _published_date_iso(claim.published_at):
        return "known"
    return "unknown"


def _magnitude_bucket_from_claim(claim: ClaimRecord) -> str | None:
    text = (claim.magnitude_text or "").lower()
    if not text:
        return None
    if re.search(r"\b(\$?\d+(\.\d+)?\s*billion|\$?\d+(\.\d+)?\s*bn)\b", text):
        return "large"
    if re.search(r"\b(\$?\d+(\.\d+)?\s*million|\$?\d+(\.\d+)?\s*mn)\b", text):
        return "medium"
    return "small"


def _normalized_event_arguments(value: Any, claim: ClaimRecord) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {
        "claim_text": claim.claim_text,
        "evidence_span": claim.evidence_span,
        "claimed_event_time_text": claim.claimed_event_time_text,
        "magnitude_text": claim.magnitude_text,
    }


def _valid_event_type(value: Any) -> str:
    raw = _clean_text(value)
    if raw in EVENT_TYPE_TO_FAMILY:
        return raw
    return "unknown"


def _valid_time_precision(value: Any) -> str:
    raw = _clean_text(value)
    if raw in {"known", "inferred", "unknown"}:
        return raw
    return "unknown"


def _valid_confidence(value: Any) -> str:
    raw = _clean_text(value).lower()
    if raw in {"high", "medium", "low"}:
        return raw
    return "medium"


def _optional_polarity(value: Any) -> str | None:
    raw = _clean_text(value).lower()
    if raw in {"positive", "negative", "neutral", "mixed"}:
        return raw
    return None


def _valid_magnitude_bucket(value: Any) -> str | None:
    raw = _clean_text(value).lower()
    if raw in {"large", "medium", "small", "unknown"}:
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


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    text = _clean_text(value)
    return text or None


__all__ = [
    "DEFAULT_EVENT_MODEL",
    "EVENT_TYPE_TO_FAMILY",
    "EVENT_TYPE_TO_LABEL",
    "EventRecord",
    "event_record_to_dict",
    "normalize_claim_to_event",
    "normalize_claims_to_events",
]
