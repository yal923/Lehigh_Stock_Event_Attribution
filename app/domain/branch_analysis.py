from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.attribution_router import AttributionRouterResult
from app.domain.branch_rerank import (
    BranchRerankResult,
    branch_rerank_result_to_dict,
)
from app.domain.news_v2_02_claims import ClaimRecord, claim_record_to_dict
from app.domain.news_v2_03_events import EventRecord, event_record_to_dict
from app.domain.news_v2_04_bundles import EvidenceBundle, evidence_bundle_to_dict
from app.domain.news_v2_05_explanations import (
    ExplanationDraft,
    explanation_draft_to_dict,
)


@dataclass(frozen=True)
class BranchAnalysisArtifacts:
    branch_mode: str
    rerank_result: BranchRerankResult
    claims: tuple[ClaimRecord, ...]
    events: tuple[EventRecord, ...]
    bundles: tuple[EvidenceBundle, ...]
    explanation: ExplanationDraft


@dataclass(frozen=True)
class MultiBranchAnalysisPlan:
    routing_mode: str
    primary_mode: str | None
    default_modes: tuple[str, ...]
    lazy_modes: tuple[str, ...]


def build_multi_branch_analysis_plan(
    router_result: AttributionRouterResult,
) -> MultiBranchAnalysisPlan:
    return MultiBranchAnalysisPlan(
        routing_mode=router_result.routing_mode,
        primary_mode=router_result.primary_mode,
        default_modes=tuple(router_result.default_visible_modes),
        lazy_modes=tuple(router_result.lazy_available_modes),
    )


def branch_analysis_artifacts_to_dict(
    artifacts: BranchAnalysisArtifacts,
) -> dict[str, Any]:
    return {
        "branch_mode": artifacts.branch_mode,
        "rerank_result": branch_rerank_result_to_dict(artifacts.rerank_result),
        "claims": [claim_record_to_dict(claim) for claim in artifacts.claims],
        "events": [event_record_to_dict(event) for event in artifacts.events],
        "bundles": [evidence_bundle_to_dict(bundle) for bundle in artifacts.bundles],
        "explanation": explanation_draft_to_dict(artifacts.explanation),
    }


def multi_branch_analysis_plan_to_dict(
    plan: MultiBranchAnalysisPlan,
) -> dict[str, Any]:
    return {
        "routing_mode": plan.routing_mode,
        "primary_mode": plan.primary_mode,
        "default_modes": list(plan.default_modes),
        "lazy_modes": list(plan.lazy_modes),
    }


__all__ = [
    "BranchAnalysisArtifacts",
    "MultiBranchAnalysisPlan",
    "branch_analysis_artifacts_to_dict",
    "build_multi_branch_analysis_plan",
    "multi_branch_analysis_plan_to_dict",
]
