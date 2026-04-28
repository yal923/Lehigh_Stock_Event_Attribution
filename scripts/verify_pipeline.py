"""
scripts/verify_pipeline.py

Coverage-driven adaptive verification of the episode analysis pipeline.

Algorithm:
  1. Build candidate pool: Tier 1 (hand-picked) + Tier 2 (auto from DB, sector-diverse)
  2. For each candidate episode (cache-first by default):
       - Run pipeline
       - Record routing mode
       - Run L1 (structural) + L2 (semantic) assertions
  3. Soft stop: all 3 routing modes seen AND episodes run >= MIN_BATCH
  4. Run cap stop: pause once HARD_CAP episodes have been verified in this run
  5. Report L1/L2 assertion results + L3 routing distribution to console and JSON

Usage:
    python scripts/verify_pipeline.py [--force] [--limit N] [--resume-from PATH]

Options:
    --force              Force recompute (ignore cached results)
    --limit N            Override HARD_CAP (default: 100)
    --resume-from PATH   Resume from an existing verification summary JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIER1_TICKERS = [
    "NVDA", "AAPL", "MSFT", "JPM", "V",
    "LLY", "JNJ", "AMZN", "TSLA", "XOM",
    "COST", "WMT", "NEE", "AMT", "CAT", "META",
]

MIN_BATCH = 10
HARD_CAP = 100

VALID_OUTWARD_STATUSES = {"SUPPORTED", "LIMITED_EVIDENCE", "NO_EVIDENCE"}
VALID_QUALITY_STATUSES = {"OOS", "POOR_FIT", "CAUTION", "PASS"}
VALID_ROUTING_MODES = {"dominant", "mixed", "degraded"}
VALID_BRANCH_MODES = {"market", "industry", "firm"}

ALL_THREE_ROUTING_MODES = {"dominant", "mixed", "degraded"}


# ---------------------------------------------------------------------------
# Candidate pool builders
# ---------------------------------------------------------------------------

def _build_tier1_candidates(tier1_tickers: list[str]) -> list:
    """Get attributable Tier 1 episodes using multi-episode round-robin by ticker."""
    from app.models import Company, Episode
    from app.domain.attribution import episode_is_attributable

    ticker_episode_sets: list[list] = []
    for ticker in tier1_tickers:
        company = Company.query.filter_by(ticker=ticker).first()
        if company is None:
            print(f"  [WARN] Tier 1 ticker {ticker!r} not found in DB, skipping.")
            continue
        episodes = (
            Episode.query
            .filter(Episode.company_id == company.id)
            .order_by(Episode.peak_date.desc())
            .all()
        )
        if not episodes:
            print(f"  [WARN] Tier 1 ticker {ticker!r} has no episodes, skipping.")
            continue

        attributable_episodes = [
            episode for episode in episodes if episode_is_attributable(episode)
        ]
        if not attributable_episodes:
            print(
                f"  [WARN] Tier 1 ticker {ticker!r} has no attributable episodes, skipping."
            )
            continue
        ticker_episode_sets.append(attributable_episodes)

    return _round_robin_episode_sets(ticker_episode_sets)


def _build_tier2_candidates(tier1_set: set[str], n_per_sector: int = 2) -> list:
    """Auto-select attributable Tier 2 episodes using multi-episode round-robin."""
    from app.models import Company, Episode
    from app.domain.attribution import episode_is_attributable

    selected_companies_by_sector: dict[str, list] = {}

    companies = (
        Company.query
        .filter(Company.ticker.notin_(list(tier1_set)))
        .filter(Company.gics_sector.isnot(None))
        .order_by(Company.ticker.asc())
        .all()
    )

    for company in companies:
        sector = company.gics_sector
        if sector not in selected_companies_by_sector:
            selected_companies_by_sector[sector] = []
        if len(selected_companies_by_sector[sector]) >= n_per_sector:
            continue
        selected_companies_by_sector[sector].append(company)

    company_episode_sets: list[list] = []
    for companies_in_sector in selected_companies_by_sector.values():
        for company in companies_in_sector:
            episodes = (
                Episode.query
                .filter(Episode.company_id == company.id)
                .order_by(Episode.peak_date.desc())
                .all()
            )
            attributable_episodes = [
                episode for episode in episodes if episode_is_attributable(episode)
            ]
            if attributable_episodes:
                company_episode_sets.append(attributable_episodes)

    return _round_robin_episode_sets(company_episode_sets)


def _round_robin_episode_sets(episode_sets: list[list]) -> list:
    merged: list = []
    depth = 0
    while True:
        added = False
        for episodes in episode_sets:
            if depth < len(episodes):
                merged.append(episodes[depth])
                added = True
        if not added:
            break
        depth += 1
    return merged


def _interleave_candidate_tiers(
    tier1_episodes: list,
    tier2_episodes: list,
) -> list:
    merged: list = []
    seen_episode_ids: set[int] = set()
    max_len = max(len(tier1_episodes), len(tier2_episodes))

    for idx in range(max_len):
        if idx < len(tier1_episodes):
            episode = tier1_episodes[idx]
            if episode.id not in seen_episode_ids:
                merged.append(episode)
                seen_episode_ids.add(episode.id)
        if idx < len(tier2_episodes):
            episode = tier2_episodes[idx]
            if episode.id not in seen_episode_ids:
                merged.append(episode)
                seen_episode_ids.add(episode.id)

    return merged


# ---------------------------------------------------------------------------
# L1: Structural assertions
# ---------------------------------------------------------------------------

def _assert_l1(result: dict) -> list[str]:
    """Check required keys and basic types exist in the pipeline result."""
    failures = []

    for key in ("episode_id", "analysis", "attribution", "decision", "branch_plan"):
        if key not in result:
            failures.append(f"L1: missing top-level key '{key}'")

    decision = result.get("decision") or {}
    for key in ("outward_status", "driver_hypothesis", "headline_explanation", "explanation_confidence"):
        if key not in decision:
            failures.append(f"L1: decision missing key '{key}'")

    attribution = result.get("attribution") or {}
    for key in ("quality_status", "market_ratio", "industry_ratio", "firm_ratio"):
        if key not in attribution:
            failures.append(f"L1: attribution missing key '{key}'")

    branch_plan = result.get("branch_plan") or {}
    for key in ("routing_mode", "primary_mode", "default_modes", "lazy_modes"):
        if key not in branch_plan:
            failures.append(f"L1: branch_plan missing key '{key}'")

    return failures


# ---------------------------------------------------------------------------
# L2: Semantic assertions
# ---------------------------------------------------------------------------

def _assert_l2(result: dict) -> list[str]:
    """Check valid enum values and routing invariants."""
    failures = []

    attribution = result.get("attribution") or {}
    decision = result.get("decision") or {}
    branch_plan = result.get("branch_plan") or {}
    analysis = result.get("analysis") or {}

    quality_status = attribution.get("quality_status")
    routing_mode = branch_plan.get("routing_mode")
    outward_status = decision.get("outward_status")

    # Enum validity
    if quality_status not in VALID_QUALITY_STATUSES:
        failures.append(f"L2: invalid quality_status={quality_status!r}")

    if routing_mode not in VALID_ROUTING_MODES:
        failures.append(f"L2: invalid routing_mode={routing_mode!r}")

    if outward_status is not None and outward_status not in VALID_OUTWARD_STATUSES:
        failures.append(f"L2: invalid outward_status={outward_status!r}")

    # Routing invariants
    if routing_mode == "degraded":
        # Degraded routing is only triggered by OOS attribution quality (DEC-174)
        if quality_status != "OOS":
            failures.append(
                f"L2: degraded routing requires quality_status=OOS, got {quality_status!r}"
            )

    elif routing_mode == "dominant":
        primary_mode = branch_plan.get("primary_mode")
        if primary_mode not in VALID_BRANCH_MODES:
            failures.append(
                f"L2: dominant routing must have primary_mode in {VALID_BRANCH_MODES}, "
                f"got {primary_mode!r}"
            )
        default_modes = branch_plan.get("default_modes") or []
        if len(default_modes) != 1:
            failures.append(
                f"L2: dominant routing must have exactly 1 default_mode, got {default_modes!r}"
            )
        lazy_modes = branch_plan.get("lazy_modes") or []
        if len(lazy_modes) != 2:
            failures.append(
                f"L2: dominant routing must have exactly 2 lazy_modes, got {lazy_modes!r}"
            )

    elif routing_mode == "mixed":
        default_modes = set(branch_plan.get("default_modes") or [])
        if default_modes != VALID_BRANCH_MODES:
            failures.append(
                f"L2: mixed routing must expose all 3 branch modes as default, got {default_modes!r}"
            )

    # Attribution ratio range check (only for non-OOS where regression was fit)
    if quality_status != "OOS":
        for ratio_key in ("market_ratio", "industry_ratio", "firm_ratio"):
            val = attribution.get(ratio_key)
            if val is not None:
                try:
                    fval = float(val)
                    if not (0.0 <= fval <= 1.0):
                        failures.append(
                            f"L2: {ratio_key}={fval:.4f} out of range [0, 1]"
                        )
                except (TypeError, ValueError):
                    failures.append(f"L2: {ratio_key} is not a float: {val!r}")

    # analysis routing_mode must be consistent with branch_plan routing_mode
    analysis_routing = analysis.get("routing_mode")
    if analysis_routing is not None and analysis_routing != routing_mode:
        failures.append(
            f"L2: analysis.routing_mode={analysis_routing!r} "
            f"inconsistent with branch_plan.routing_mode={routing_mode!r}"
        )

    return failures


def _compute_counts(
    episode_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], dict[str, int]]:
    run_ok = [r for r in episode_results if r["ok"]]
    routing_counts: dict[str, int] = {}
    outward_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}

    for r in run_ok:
        rm = r.get("routing_mode") or "unknown"
        routing_counts[rm] = routing_counts.get(rm, 0) + 1
        os_ = r.get("outward_status") or "unknown"
        outward_counts[os_] = outward_counts.get(os_, 0) + 1
        qs = r.get("quality_status") or "unknown"
        quality_counts[qs] = quality_counts.get(qs, 0) + 1

    return run_ok, routing_counts, outward_counts, quality_counts


def _build_summary(
    *,
    run_started_at: str,
    force_recompute: bool,
    hard_cap: int,
    stop_reason: str,
    episode_results: list[dict[str, Any]],
    routing_seen: set[str],
    l1_failures_total: int,
    l2_failures_total: int,
    candidate_pool_size: int,
) -> dict[str, Any]:
    run_ok, routing_counts, outward_counts, quality_counts = _compute_counts(episode_results)
    n_ok = len(run_ok)
    n_total = len(episode_results)
    overall_ok = (l1_failures_total == 0 and l2_failures_total == 0 and n_total - n_ok == 0)

    return {
        "run_at": run_started_at,
        "force_recompute": force_recompute,
        "hard_cap": hard_cap,
        "min_batch": MIN_BATCH,
        "candidate_pool_size": candidate_pool_size,
        "stop_reason": stop_reason,
        "run_cap_reached": stop_reason == "run_cap_reached",
        "total_run": n_total,
        "success": n_ok,
        "errors": n_total - n_ok,
        "l1_failures_total": l1_failures_total,
        "l2_failures_total": l2_failures_total,
        "routing_seen": sorted(routing_seen),
        "all_routing_modes_covered": routing_seen >= ALL_THREE_ROUTING_MODES,
        "routing_counts": routing_counts,
        "outward_counts": outward_counts,
        "quality_counts": quality_counts,
        "overall_pass": overall_ok,
        "episodes": episode_results,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _prepare_output_paths(run_started_at: str) -> tuple[Path, Path, Path]:
    out_dir = Path("instance") / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = run_started_at.replace("-", "").replace(":", "").replace(".", "").replace("Z", "")
    jsonl_path = out_dir / f"verify_{ts}_episodes.jsonl"
    current_summary_path = out_dir / f"verify_{ts}_current.json"
    final_summary_path = out_dir / f"verify_{ts}.json"
    return jsonl_path, current_summary_path, final_summary_path


def _load_resume_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Resume summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Resume summary must be a JSON object: {path}")
    if not payload.get("run_at"):
        raise ValueError(f"Resume summary missing run_at: {path}")
    if not isinstance(payload.get("episodes"), list):
        raise ValueError(f"Resume summary missing episodes list: {path}")
    return payload


# ---------------------------------------------------------------------------
# Main verification runner
# ---------------------------------------------------------------------------

def run_verification(
    force_recompute: bool = False,
    hard_cap: int = HARD_CAP,
    min_batch: int = MIN_BATCH,
    resume_from: str | None = None,
    no_soft_stop: bool = False,
) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app import create_app
    from app.services.pipeline_service import run_episode_pipeline

    app = create_app()
    with app.app_context():
        tier1_set = set(TIER1_TICKERS)

        print("Building candidate pool...")
        tier1_episodes = _build_tier1_candidates(TIER1_TICKERS)
        tier2_episodes = _build_tier2_candidates(tier1_set, n_per_sector=2)
        all_candidates = _interleave_candidate_tiers(tier1_episodes, tier2_episodes)

        if resume_from:
            resume_path = Path(resume_from)
            resume_summary = _load_resume_summary(resume_path)
            run_started_at = str(resume_summary["run_at"])
            episode_log_path, current_summary_path, final_summary_path = _prepare_output_paths(
                run_started_at
            )
            episode_results = list(resume_summary.get("episodes") or [])
            routing_seen: set[str] = set(resume_summary.get("routing_seen") or [])
            l1_failures_total = int(resume_summary.get("l1_failures_total") or 0)
            l2_failures_total = int(resume_summary.get("l2_failures_total") or 0)
            processed_episode_ids = {
                int(record["episode_id"])
                for record in episode_results
                if record.get("episode_id") is not None
            }
            stop_reason = "resumed_in_progress"
        else:
            run_started_at = datetime.utcnow().isoformat() + "Z"
            episode_log_path, current_summary_path, final_summary_path = _prepare_output_paths(
                run_started_at
            )
            episode_results = []
            routing_seen = set()
            l1_failures_total = 0
            l2_failures_total = 0
            processed_episode_ids: set[int] = set()
            stop_reason = "exhausted_pool"

        print(
            f"Pool: {len(tier1_episodes)} Tier 1 + {len(tier2_episodes)} Tier 2 "
            f"= {len(all_candidates)} total candidates"
        )
        print(f"Params: MIN_BATCH={MIN_BATCH}, HARD_CAP={hard_cap}, force={force_recompute}")
        print(f"Episode log: {episode_log_path}")
        print(f"Current summary: {current_summary_path}")
        if resume_from:
            print(
                f"Resume: loaded {len(episode_results)} prior episodes from {resume_from}"
            )
        print()

        for episode in all_candidates:
            if episode.id in processed_episode_ids:
                continue

            n = len(episode_results)

            if n >= hard_cap:
                stop_reason = "run_cap_reached"
                print(
                    f"\n[RUN CAP STOP] Verified {hard_cap} episodes in this run. "
                    "Pausing and summarizing current results."
                )
                break

            ticker = episode.company.ticker if episode.company else f"ep-{episode.id}"
            peak = episode.peak_date.isoformat() if episode.peak_date else "?"
            print(f"[{n + 1:>3}] {ticker:>6}  peak={peak}  ", end="", flush=True)

            try:
                result = run_episode_pipeline(episode, force_recompute=force_recompute)
            except Exception as exc:
                print(f"ERROR: {exc}")
                episode_record = {
                    "index": n + 1,
                    "ticker": ticker,
                    "episode_id": episode.id,
                    "peak_date": peak,
                    "ok": False,
                    "error": str(exc),
                    "routing_mode": None,
                    "outward_status": None,
                    "quality_status": None,
                    "l1_failures": [],
                    "l2_failures": [],
                }
                episode_results.append(episode_record)
                processed_episode_ids.add(episode.id)
                _append_jsonl(episode_log_path, episode_record)
                current_summary = _build_summary(
                    run_started_at=run_started_at,
                    force_recompute=force_recompute,
                    hard_cap=hard_cap,
                    stop_reason=stop_reason,
                    episode_results=episode_results,
                    routing_seen=routing_seen,
                    l1_failures_total=l1_failures_total,
                    l2_failures_total=l2_failures_total,
                    candidate_pool_size=len(all_candidates),
                )
                current_summary["episode_log_path"] = str(episode_log_path)
                current_summary["current_summary_path"] = str(current_summary_path)
                current_summary["final_summary_path"] = str(final_summary_path)
                _write_json(current_summary_path, current_summary)
                time.sleep(60)
                continue

            routing_mode = (result.get("branch_plan") or {}).get("routing_mode")
            outward_status = (result.get("decision") or {}).get("outward_status")
            quality_status = (result.get("attribution") or {}).get("quality_status")

            l1_failures = _assert_l1(result)
            l2_failures = _assert_l2(result)
            l1_failures_total += len(l1_failures)
            l2_failures_total += len(l2_failures)

            status_flag = "OK" if not l1_failures and not l2_failures else "FAIL"
            print(
                f"routing={routing_mode or '?'}  "
                f"outward={outward_status or '?'}  "
                f"quality={quality_status or '?'}  "
                f"[{status_flag}]"
            )
            for msg in l1_failures + l2_failures:
                print(f"         {msg}")

            episode_record = {
                "index": n + 1,
                "ticker": ticker,
                "episode_id": episode.id,
                "peak_date": peak,
                "ok": True,
                "routing_mode": routing_mode,
                "outward_status": outward_status,
                "quality_status": quality_status,
                "l1_failures": l1_failures,
                "l2_failures": l2_failures,
            }
            episode_results.append(episode_record)
            processed_episode_ids.add(episode.id)

            if routing_mode:
                routing_seen.add(routing_mode)

            _append_jsonl(episode_log_path, episode_record)
            current_summary = _build_summary(
                run_started_at=run_started_at,
                force_recompute=force_recompute,
                hard_cap=hard_cap,
                stop_reason=stop_reason,
                episode_results=episode_results,
                routing_seen=routing_seen,
                l1_failures_total=l1_failures_total,
                l2_failures_total=l2_failures_total,
                candidate_pool_size=len(all_candidates),
            )
            current_summary["episode_log_path"] = str(episode_log_path)
            current_summary["current_summary_path"] = str(current_summary_path)
            current_summary["final_summary_path"] = str(final_summary_path)
            _write_json(current_summary_path, current_summary)

            # Throttle: give yfinance time to reset between episodes
            time.sleep(60)

            # Soft stop: all 3 routing modes covered AND minimum batch reached
            if (
                not no_soft_stop
                and len(episode_results) >= min_batch
                and routing_seen >= ALL_THREE_ROUTING_MODES
            ):
                stop_reason = "soft_stop_all_modes_covered"
                print(f"\n[SOFT STOP] All 3 routing modes seen after {len(episode_results)} episodes.")
                break

        summary = _build_summary(
            run_started_at=run_started_at,
            force_recompute=force_recompute,
            hard_cap=hard_cap,
            stop_reason=stop_reason,
            episode_results=episode_results,
            routing_seen=routing_seen,
            l1_failures_total=l1_failures_total,
            l2_failures_total=l2_failures_total,
            candidate_pool_size=len(all_candidates),
        )
        summary["episode_log_path"] = str(episode_log_path)
        summary["current_summary_path"] = str(current_summary_path)
        summary["final_summary_path"] = str(final_summary_path)

        run_ok, routing_counts, outward_counts, quality_counts = _compute_counts(episode_results)
        n_ok = len(run_ok)
        n_total = len(episode_results)

        print()
        print("=" * 64)
        print("L3 ROUTING COVERAGE REPORT")
        print("=" * 64)
        print(f"  Total episodes run : {n_total}")
        print(f"  Successful         : {n_ok}")
        print(f"  Errors             : {n_total - n_ok}")
        print(f"  Stop reason        : {stop_reason}")
        print()
        print("  Routing mode distribution:")
        for mode in ("dominant", "mixed", "degraded"):
            count = routing_counts.get(mode, 0)
            pct = count / n_ok * 100 if n_ok else 0.0
            seen_mark = "seen" if mode in routing_seen else "NOT SEEN"
            print(f"    {mode:<12} {count:>3}  ({pct:5.1f}%)  [{seen_mark}]")
        missing_modes = ALL_THREE_ROUTING_MODES - routing_seen
        if missing_modes:
            print(f"\n  WARNING: routing modes not seen: {', '.join(sorted(missing_modes))}")
            if stop_reason == "run_cap_reached":
                print("  Run paused at the single-run verification cap before full coverage.")
            else:
                print("  Consider running with a larger --limit to cover missing modes.")
        print()
        print("  Outward status distribution:")
        for status in ("SUPPORTED", "LIMITED_EVIDENCE", "NO_EVIDENCE"):
            count = outward_counts.get(status, 0)
            pct = count / n_ok * 100 if n_ok else 0.0
            print(f"    {status:<22} {count:>3}  ({pct:5.1f}%)")
        print()
        print("  Attribution quality distribution:")
        for q in ("PASS", "CAUTION", "POOR_FIT", "OOS"):
            count = quality_counts.get(q, 0)
            pct = count / n_ok * 100 if n_ok else 0.0
            print(f"    {q:<12} {count:>3}  ({pct:5.1f}%)")
        print()
        print(f"  Assertion failures : L1={l1_failures_total}  L2={l2_failures_total}")
        print(f"  Overall result     : {'PASS' if summary['overall_pass'] else 'FAIL'}")
        print("=" * 64)

        _write_json(current_summary_path, summary)
        _write_json(final_summary_path, summary)
        return summary


def save_report(summary: dict) -> Path:
    return Path(summary["final_summary_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coverage-driven adaptive pipeline verification."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute all episodes (ignore cache).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=HARD_CAP,
        help=f"Single-run cap on number of episodes to verify before pausing (default: {HARD_CAP}).",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=MIN_BATCH,
        help=f"Minimum episodes before soft stop can fire (default: {MIN_BATCH}).",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume verification from an existing current/final summary JSON.",
    )
    parser.add_argument(
        "--no-soft-stop",
        action="store_true",
        help="Disable the soft stop (run until hard cap or pool exhausted). Useful when resuming.",
    )
    args = parser.parse_args()

    summary = run_verification(
        force_recompute=args.force,
        hard_cap=args.limit,
        min_batch=args.min_batch,
        resume_from=args.resume_from,
        no_soft_stop=args.no_soft_stop,
    )
    report_path = save_report(summary)
    print(f"\nReport saved: {report_path}")
    sys.exit(0 if summary["overall_pass"] else 1)
