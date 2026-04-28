from datetime import datetime

from app.domain.episode_pipeline import (
    run_episode_pipeline as _run_episode_pipeline,
    run_episode_secondary_branch_analysis as _run_episode_secondary_branch_analysis,
)
from app.models import Episode


def run_episode_pipeline(episode: Episode, force_recompute: bool = False):
    return _run_episode_pipeline(episode=episode, force_recompute=force_recompute)


def run_episode_secondary_branch_analysis(
    episode: Episode,
    *,
    branch_mode: str,
    force_recompute: bool = False,
):
    return _run_episode_secondary_branch_analysis(
        episode=episode,
        branch_mode=branch_mode,
        force_recompute=force_recompute,
    )


def run_batch_episode_pipeline(
    *,
    limit: int = 20,
    force_recompute: bool = False,
) -> dict:
    """Run pipeline for most recent episodes and return execution summary."""
    episodes = (
        Episode.query
        .order_by(Episode.peak_date.desc(), Episode.id.desc())
        .limit(limit)
        .all()
    )

    started_at = datetime.utcnow().isoformat()
    results: list[dict] = []
    success = 0
    failed = 0

    for ep in episodes:
        try:
            out = _run_episode_pipeline(episode=ep, force_recompute=force_recompute)
            decision = out.get("decision") or {}
            results.append(
                {
                    "episode_id": ep.id,
                    "ticker": ep.company.ticker if ep.company else None,
                    "peak_date": ep.peak_date.isoformat() if ep.peak_date else None,
                    "ok": True,
                    "outward_status": decision.get("outward_status"),
                    "driver_hypothesis": decision.get("driver_hypothesis"),
                    "explanation_confidence": decision.get("explanation_confidence"),
                }
            )
            success += 1
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "episode_id": ep.id,
                    "ticker": ep.company.ticker if ep.company else None,
                    "peak_date": ep.peak_date.isoformat() if ep.peak_date else None,
                    "ok": False,
                    "error": str(exc),
                }
            )
            failed += 1

    return {
        "started_at": started_at,
        "ended_at": datetime.utcnow().isoformat(),
        "total": len(episodes),
        "success": success,
        "failed": failed,
        "results": results,
    }
