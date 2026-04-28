from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DOMINANT_MIN_RATIO = 0.70
DOMINANT_MIN_GAP = 0.15
_VISIBLE_MODE_ORDER = ("market", "industry", "firm")


@dataclass(frozen=True)
class AttributionRouterInput:
    quality_status: str
    market_ratio: float
    industry_ratio: float
    firm_ratio: float


@dataclass(frozen=True)
class AttributionRouterResult:
    routing_mode: str
    primary_mode: str | None
    mode_weights: dict[str, float]
    default_visible_modes: tuple[str, ...]
    lazy_available_modes: tuple[str, ...]
    reason_codes: tuple[str, ...]


def build_router_input_from_attribution_payload(
    attribution_payload: dict[str, Any],
) -> AttributionRouterInput:
    return AttributionRouterInput(
        quality_status=str(attribution_payload.get("quality_status") or ""),
        market_ratio=_safe_float(attribution_payload.get("market_ratio")),
        industry_ratio=_safe_float(attribution_payload.get("industry_ratio")),
        firm_ratio=_safe_float(attribution_payload.get("firm_ratio")),
    )


def route_attribution(
    router_input: AttributionRouterInput,
) -> AttributionRouterResult:
    mode_weights = {
        "market": router_input.market_ratio,
        "industry": router_input.industry_ratio,
        "firm": router_input.firm_ratio,
    }

    # Only OOS (data acquisition failure) routes to degraded (DEC-174).
    if router_input.quality_status.upper() == "OOS":
        return AttributionRouterResult(
            routing_mode="degraded",
            primary_mode=None,
            mode_weights=mode_weights,
            default_visible_modes=(),
            lazy_available_modes=(),
            reason_codes=("attribution_data_unavailable",),
        )

    ranked_modes = sorted(
        mode_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top1_mode, top1_ratio = ranked_modes[0]
    top2_mode, top2_ratio = ranked_modes[1]
    top3_mode, _ = ranked_modes[2]

    # POOR_FIT: betas are statistically insignificant, so ratios cannot reliably
    # support a dominant claim. Force mixed so all three branches are shown equally
    # rather than asserting a dominant driver the model cannot substantiate.
    if router_input.quality_status.upper() == "POOR_FIT":
        return AttributionRouterResult(
            routing_mode="mixed",
            primary_mode=top1_mode,
            mode_weights=mode_weights,
            default_visible_modes=_VISIBLE_MODE_ORDER,
            lazy_available_modes=(),
            reason_codes=("poor_fit_forced_mixed",),
        )

    # CAUTION and PASS: ratios carry reliable directional signal; route by ratio values.
    if top1_ratio >= DOMINANT_MIN_RATIO and (top1_ratio - top2_ratio) >= DOMINANT_MIN_GAP:
        return AttributionRouterResult(
            routing_mode="dominant",
            primary_mode=top1_mode,
            mode_weights=mode_weights,
            default_visible_modes=(top1_mode,),
            lazy_available_modes=(top2_mode, top3_mode),
            reason_codes=(f"dominant_{top1_mode}",),
        )

    return AttributionRouterResult(
        routing_mode="mixed",
        primary_mode=top1_mode,
        mode_weights=mode_weights,
        default_visible_modes=_VISIBLE_MODE_ORDER,
        lazy_available_modes=(),
        reason_codes=("mixed_split",),
    )


def attribution_router_result_to_dict(
    result: AttributionRouterResult,
) -> dict[str, Any]:
    return asdict(result)


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "AttributionRouterInput",
    "AttributionRouterResult",
    "DOMINANT_MIN_GAP",
    "DOMINANT_MIN_RATIO",
    "attribution_router_result_to_dict",
    "build_router_input_from_attribution_payload",
    "route_attribution",
]
