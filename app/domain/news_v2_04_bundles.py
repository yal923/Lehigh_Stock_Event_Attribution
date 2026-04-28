from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Iterable, Sequence
from uuid import uuid4

from app.config.constants import normalize_publisher
from app.domain.news_v2_01_retrieval import EpisodeContext, RankedArticle
from app.domain.news_v2_02_claims import ClaimRecord
from app.domain.news_v2_03_events import EventRecord


TOP_TIER_PUBLISHERS = frozenset(
    {
        "reuters",
        "dowjones",
        "bloomberg",
        "cnbc",
    }
)

MID_TIER_PUBLISHERS = frozenset(
    {
        "yahoo",
        "yahoofinance",
        "benzinga",
        "marketwatch",
        "gurufocus",
        "zacksinvestmentresearch",
        "themotleyfool",
    }
)


@dataclass(frozen=True)
class EvidenceBundle:
    bundle_id: str
    canonical_event_type: str
    primary_entity: str | None
    event_time_window: dict[str, str | None]
    supporting_event_ids: tuple[str, ...]
    supporting_claim_ids: tuple[str, ...]
    supporting_article_ids: tuple[str, ...]
    representative_claim_ids: tuple[str, ...]
    corroboration_count: int
    source_diversity_count: int
    time_alignment_score: float
    source_quality_score: float
    episode_relevance_score: float
    bundle_score: float
    bundle_status: str
    eligible_for_explanation: bool
    bundle_score_breakdown: dict[str, float]
    merge_method: str
    merge_confidence: str
    bundle_notes: tuple[str, ...] = ()


def build_evidence_bundles(
    events: Sequence[EventRecord],
    claims: Sequence[ClaimRecord],
    ranked_articles: Sequence[RankedArticle],
    episode_context: EpisodeContext,
) -> list[EvidenceBundle]:
    claims_by_id = {claim.claim_id: claim for claim in claims}
    ranked_articles_by_id = {
        article.article_id: article for article in ranked_articles
    }

    groups = _group_events_into_bundles(events)
    bundles = [
        _build_bundle_from_group(
            group=group,
            claims_by_id=claims_by_id,
            ranked_articles_by_id=ranked_articles_by_id,
            episode_context=episode_context,
        )
        for group in groups
    ]
    return sorted(bundles, key=lambda bundle: bundle.bundle_score, reverse=True)


def evidence_bundle_to_dict(bundle: EvidenceBundle) -> dict[str, Any]:
    data = asdict(bundle)
    data["supporting_event_ids"] = list(bundle.supporting_event_ids)
    data["supporting_claim_ids"] = list(bundle.supporting_claim_ids)
    data["supporting_article_ids"] = list(bundle.supporting_article_ids)
    data["representative_claim_ids"] = list(bundle.representative_claim_ids)
    data["bundle_notes"] = list(bundle.bundle_notes)
    return data


def _group_events_into_bundles(
    events: Sequence[EventRecord],
) -> list[list[EventRecord]]:
    groups: list[list[EventRecord]] = []
    ordered_events = sorted(
        events,
        key=lambda event: (
            event.canonical_event_type,
            (event.primary_entity or "").lower(),
            event.event_time_start or "",
            event.event_id,
        ),
    )

    for event in ordered_events:
        matched_group: list[EventRecord] | None = None
        for group in groups:
            if _event_matches_group(event, group):
                matched_group = group
                break

        if matched_group is None:
            groups.append([event])
        else:
            matched_group.append(event)

    return groups


def _event_matches_group(event: EventRecord, group: Sequence[EventRecord]) -> bool:
    anchor = group[0]
    if event.canonical_event_type != anchor.canonical_event_type:
        return False
    if _entity_key(event.primary_entity) != _entity_key(anchor.primary_entity):
        return False
    return _group_time_compatible(event, group)


def _group_time_compatible(event: EventRecord, group: Sequence[EventRecord]) -> bool:
    return all(_events_overlap_or_same_day(event, existing) for existing in group)


def _events_overlap_or_same_day(left: EventRecord, right: EventRecord) -> bool:
    left_start, left_end = _event_window_dates(left)
    right_start, right_end = _event_window_dates(right)
    if left_start is None or left_end is None or right_start is None or right_end is None:
        return False
    latest_start = max(left_start, right_start)
    earliest_end = min(left_end, right_end)
    return latest_start <= earliest_end


def _event_window_dates(event: EventRecord) -> tuple[date | None, date | None]:
    start = _parse_date(event.event_time_start)
    end = _parse_date(event.event_time_end)
    if start is None and end is not None:
        start = end
    if end is None and start is not None:
        end = start
    return start, end


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None


def _entity_key(value: str | None) -> str:
    return (value or "").strip().lower()


def _build_bundle_from_group(
    *,
    group: Sequence[EventRecord],
    claims_by_id: dict[str, ClaimRecord],
    ranked_articles_by_id: dict[str, RankedArticle],
    episode_context: EpisodeContext,
) -> EvidenceBundle:
    anchor = group[0]
    supporting_event_ids = tuple(event.event_id for event in group)
    supporting_claim_ids = _unique_ordered(
        claim_id
        for event in group
        for claim_id in event.supporting_claim_ids
    )
    supporting_article_ids = _unique_ordered(
        article_id
        for event in group
        for article_id in event.supporting_article_ids
    )
    representative_claim_ids = _select_representative_claim_ids(
        supporting_claim_ids=supporting_claim_ids,
        claims_by_id=claims_by_id,
        ranked_articles_by_id=ranked_articles_by_id,
    )

    corroboration_count = len(supporting_claim_ids)
    source_diversity_count = _count_unique_publishers(
        supporting_article_ids=supporting_article_ids,
        ranked_articles_by_id=ranked_articles_by_id,
        claims_by_id=claims_by_id,
    )
    time_alignment_score = _time_alignment_score(group)
    source_quality_score = _source_quality_score(
        supporting_article_ids=supporting_article_ids,
        ranked_articles_by_id=ranked_articles_by_id,
        claims_by_id=claims_by_id,
    )
    episode_relevance_score = _episode_relevance_score(
        supporting_article_ids=supporting_article_ids,
        ranked_articles_by_id=ranked_articles_by_id,
    )
    corroboration_axis = min(corroboration_count / 3.0, 1.0)
    source_diversity_axis = min(source_diversity_count / 3.0, 1.0)
    bundle_score_breakdown = {
        "corroboration": round(corroboration_axis, 4),
        "source_diversity": round(source_diversity_axis, 4),
        "time_alignment": round(time_alignment_score, 4),
        "source_quality": round(source_quality_score, 4),
        "episode_relevance": round(episode_relevance_score, 4),
    }
    bundle_score = round(
        sum(bundle_score_breakdown.values()) / len(bundle_score_breakdown),
        4,
    )
    bundle_status = _bundle_status(bundle_score)
    eligible_for_explanation = _eligible_for_explanation(
        canonical_event_type=anchor.canonical_event_type,
        bundle_status=bundle_status,
    )

    return EvidenceBundle(
        bundle_id=f"bundle-{uuid4().hex[:10]}",
        canonical_event_type=anchor.canonical_event_type,
        primary_entity=anchor.primary_entity,
        event_time_window=_bundle_time_window(group),
        supporting_event_ids=supporting_event_ids,
        supporting_claim_ids=supporting_claim_ids,
        supporting_article_ids=supporting_article_ids,
        representative_claim_ids=representative_claim_ids,
        corroboration_count=corroboration_count,
        source_diversity_count=source_diversity_count,
        time_alignment_score=round(time_alignment_score, 4),
        source_quality_score=round(source_quality_score, 4),
        episode_relevance_score=round(episode_relevance_score, 4),
        bundle_score=bundle_score,
        bundle_status=bundle_status,
        eligible_for_explanation=eligible_for_explanation,
        bundle_score_breakdown=bundle_score_breakdown,
        merge_method="schema-aware-event-identity",
        merge_confidence=_merge_confidence(
            group=group,
            source_diversity_count=source_diversity_count,
            time_alignment_score=time_alignment_score,
        ),
        bundle_notes=_bundle_notes(
            group=group,
            bundle_status=bundle_status,
            episode_context=episode_context,
        ),
    )


def _select_representative_claim_ids(
    *,
    supporting_claim_ids: Sequence[str],
    claims_by_id: dict[str, ClaimRecord],
    ranked_articles_by_id: dict[str, RankedArticle],
) -> tuple[str, ...]:
    ranked_claims = sorted(
        (
            claim_id
            for claim_id in supporting_claim_ids
            if claim_id in claims_by_id
        ),
        key=lambda claim_id: (
            ranked_articles_by_id.get(claims_by_id[claim_id].article_id, None).rank_position
            if claims_by_id[claim_id].article_id in ranked_articles_by_id
            else 9999,
            claim_id,
        ),
    )
    return tuple(ranked_claims[:2])


def _count_unique_publishers(
    *,
    supporting_article_ids: Sequence[str],
    ranked_articles_by_id: dict[str, RankedArticle],
    claims_by_id: dict[str, ClaimRecord],
) -> int:
    publishers = {
        _publisher_for_article(
            article_id=article_id,
            ranked_articles_by_id=ranked_articles_by_id,
            claims_by_id=claims_by_id,
        )
        for article_id in supporting_article_ids
    }
    publishers.discard(None)
    return len(publishers)


def _source_quality_score(
    *,
    supporting_article_ids: Sequence[str],
    ranked_articles_by_id: dict[str, RankedArticle],
    claims_by_id: dict[str, ClaimRecord],
) -> float:
    if not supporting_article_ids:
        return 0.0

    scores = [
        _publisher_quality_score(
            _publisher_for_article(
                article_id=article_id,
                ranked_articles_by_id=ranked_articles_by_id,
                claims_by_id=claims_by_id,
            )
        )
        for article_id in supporting_article_ids
    ]
    return sum(scores) / len(scores)


def _publisher_for_article(
    *,
    article_id: str,
    ranked_articles_by_id: dict[str, RankedArticle],
    claims_by_id: dict[str, ClaimRecord],
) -> str | None:
    article = ranked_articles_by_id.get(article_id)
    if article is not None:
        return article.publisher

    for claim in claims_by_id.values():
        if claim.article_id == article_id:
            return claim.publisher
    return None


def _publisher_quality_score(publisher: str | None) -> float:
    normalized = normalize_publisher(publisher) if publisher else ""
    if normalized in TOP_TIER_PUBLISHERS:
        return 1.0
    if normalized in MID_TIER_PUBLISHERS:
        return 0.6
    return 0.3


def _episode_relevance_score(
    *,
    supporting_article_ids: Sequence[str],
    ranked_articles_by_id: dict[str, RankedArticle],
) -> float:
    if not supporting_article_ids:
        return 0.2

    best_rank = min(
        (
            ranked_articles_by_id[article_id].rank_position
            for article_id in supporting_article_ids
            if article_id in ranked_articles_by_id
        ),
        default=None,
    )
    if best_rank is None:
        return 0.2
    if best_rank <= 3:
        return 1.0
    if best_rank <= 6:
        return 0.7
    if best_rank <= 10:
        return 0.4
    return 0.2


def _time_alignment_score(group: Sequence[EventRecord]) -> float:
    if not group:
        return 0.0

    windows = [_event_window_dates(event) for event in group]
    usable_windows = [
        window for window in windows if window[0] is not None and window[1] is not None
    ]
    if not usable_windows:
        return 0.0

    if all(event.event_time_precision == "known" for event in group):
        return 1.0
    return 0.5


def _bundle_status(bundle_score: float) -> str:
    if bundle_score >= 0.75:
        return "strong"
    if bundle_score >= 0.55:
        return "moderate"
    if bundle_score >= 0.35:
        return "weak"
    return "unknown-boundary"


def _bundle_time_window(group: Sequence[EventRecord]) -> dict[str, str | None]:
    starts = [
        start
        for start, _ in (_event_window_dates(event) for event in group)
        if start is not None
    ]
    ends = [
        end
        for _, end in (_event_window_dates(event) for event in group)
        if end is not None
    ]
    precision = "unknown"
    if group:
        precisions = {event.event_time_precision for event in group}
        if precisions == {"known"}:
            precision = "known"
        elif starts or ends:
            precision = "inferred"

    return {
        "start": min(starts).isoformat() if starts else None,
        "end": max(ends).isoformat() if ends else None,
        "precision": precision,
    }


def _merge_confidence(
    *,
    group: Sequence[EventRecord],
    source_diversity_count: int,
    time_alignment_score: float,
) -> str:
    if len(group) >= 2 and source_diversity_count >= 2 and time_alignment_score >= 1.0:
        return "high"
    if time_alignment_score > 0.0:
        return "medium"
    return "low"


def _bundle_notes(
    *,
    group: Sequence[EventRecord],
    bundle_status: str,
    episode_context: EpisodeContext,
) -> tuple[str, ...]:
    notes: list[str] = []
    if any(event.canonical_event_type == "unknown" for event in group):
        notes.append("Contains unknown event support")
    if bundle_status == "unknown-boundary":
        notes.append("Bundle remains bounded due to weak or sparse support")
    notes.append(f"Episode window={episode_context.episode_start}..{episode_context.episode_end}")
    return tuple(notes)


def _eligible_for_explanation(
    *,
    canonical_event_type: str,
    bundle_status: str,
) -> bool:
    if canonical_event_type in {"other", "unknown"}:
        return False
    if bundle_status == "unknown-boundary":
        return False
    if (
        bundle_status == "weak"
        and canonical_event_type in {"product_or_customer_event", "other", "unknown"}
    ):
        return False
    return True


def _unique_ordered(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)
