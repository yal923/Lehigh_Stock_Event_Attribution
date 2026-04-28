"""
Run the 2026 release verification episode set.

The set is defined in EPISODES below. This script drives
the same Flask endpoints used by the UI so the V2 pipeline cache and AI
Explanation cache are populated for presentation/demo use.

Outputs:
  - instance/verification/release_verification_<timestamp>.json
  - instance/verification/release_verification_<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


EPISODES = [
    {
        "ticker": "XOM",
        "episode_id": 119,
        "theme": "US-Iran / oil shock relief",
    },
    {
        "ticker": "NVDA",
        "episode_id": 10,
        "theme": "US-Iran relief + AI/product momentum",
    },
    {
        "ticker": "ORCL",
        "episode_id": 481,
        "theme": "AI cloud earnings positive",
    },
    {
        "ticker": "AMZN",
        "episode_id": 69,
        "theme": "Big Tech AI capex fear",
    },
    {
        "ticker": "MSFT",
        "episode_id": 31,
        "theme": "AI capex + cloud growth concern",
    },
    {
        "ticker": "ADBE",
        "episode_id": 379,
        "theme": "AI disruption hits software",
    },
    {
        "ticker": "AAPL",
        "episode_id": 1,
        "theme": "Apple AI delay / mega-cap regulatory risk",
    },
    {
        "ticker": "LLY",
        "episode_id": 44,
        "theme": "New drug / FDA approval",
    },
    {
        "ticker": "ABNB",
        "episode_id": 209,
        "theme": "Earnings / travel demand",
    },
    {
        "ticker": "TSLA",
        "episode_id": 80,
        "theme": "Tesla robotaxi / AI narrative",
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _output_paths(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return (
        out_dir / f"release_verification_{stamp}.json",
        out_dir / f"release_verification_{stamp}.jsonl",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _post_json(client: Any, url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any] | None, float]:
    started = time.monotonic()
    response = client.post(url, json=body)
    elapsed = time.monotonic() - started
    return response.status_code, response.get_json(silent=True), elapsed


class _PhaseTimeout(RuntimeError):
    pass


def _run_with_timeout(label: str, seconds: int, fn: Any) -> Any:
    def _handle_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise _PhaseTimeout(f"{label} exceeded {seconds}s timeout")

    previous = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _summarize_pipeline(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    analysis = payload.get("analysis") or {}
    decision = payload.get("decision") or {}
    branch_plan = (
        payload.get("branch_plan")
        or (payload.get("analysis") or {}).get("branch_plan")
        or {}
    )
    attribution = payload.get("attribution") or analysis.get("attribution_summary") or {}
    return {
        "routing_mode": branch_plan.get("routing_mode") or analysis.get("routing_mode"),
        "primary_mode": branch_plan.get("primary_mode") or analysis.get("primary_mode"),
        "outward_status": decision.get("outward_status"),
        "quality_status": attribution.get("quality_status"),
        "headline_explanation": decision.get("headline_explanation"),
    }


def _summarize_ai(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    ai_payload = payload.get("ai_explanation") or {}
    if not isinstance(ai_payload, dict):
        return {}
    return {
        "provider_name": ai_payload.get("provider_name"),
        "model_name": ai_payload.get("model_name"),
        "evidence_quality": ai_payload.get("evidence_quality"),
        "headline_explanation": ai_payload.get("headline_explanation"),
        "source_count": len(ai_payload.get("sources") or []),
        "cost_estimate_usd": ai_payload.get("cost_estimate_usd"),
        "computed_at": ai_payload.get("computed_at"),
        "raw_response_path": ai_payload.get("raw_response_path"),
    }


def run_release_verification(
    *,
    force_v2: bool,
    force_ai_ids: set[int],
    provider: str,
    model: str,
    sleep_seconds: float,
    dry_run: bool,
    phase_timeout_seconds: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(_repo_root()))
    os.chdir(_repo_root())

    out_path, jsonl_path = _output_paths(_repo_root() / "instance" / "verification")
    summary: dict[str, Any] = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "episode_source": "built-in 2026 release verification set",
        "force_v2": force_v2,
        "force_ai_ids": sorted(force_ai_ids),
        "provider": provider or None,
        "model": model or None,
        "dry_run": dry_run,
        "phase_timeout_seconds": phase_timeout_seconds,
        "output_path": str(out_path),
        "jsonl_path": str(jsonl_path),
        "episodes": EPISODES,
        "results": [],
    }

    print("Release verification episodes:")
    for index, item in enumerate(EPISODES, start=1):
        force_marker = " force-ai" if int(item["episode_id"]) in force_ai_ids else ""
        print(
            f"{index:02d}. {item['ticker']:>5}#{item['episode_id']:<4} "
            f"{item['theme']}{force_marker}"
            ,
            flush=True,
        )

    if dry_run:
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nDry run only. Wrote selection to {out_path}", flush=True)
        return summary

    from app import create_app

    app = create_app()
    with app.app_context():
        client = app.test_client()
        for index, item in enumerate(EPISODES, start=1):
            ticker = str(item["ticker"])
            episode_id = int(item["episode_id"])
            print(f"\n[{index:02d}/{len(EPISODES):02d}] {ticker} episode {episode_id}", flush=True)

            record = {
                **item,
                "pipeline": {
                    "http_status": None,
                    "ok": False,
                    "elapsed_seconds": None,
                    "error": "not started",
                },
                "ai": {
                    "http_status": None,
                    "ok": False,
                    "elapsed_seconds": None,
                    "force_recompute": episode_id in force_ai_ids,
                    "error": "not started",
                },
            }
            _append_jsonl(jsonl_path, {"event": "start", **item})
            out_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            try:
                pipeline_status, pipeline_payload, pipeline_elapsed = _run_with_timeout(
                    f"{ticker}#{episode_id} V2",
                    phase_timeout_seconds,
                    lambda: _post_json(
                        client,
                        f"/api/company/{ticker}/episode/{episode_id}/run",
                        {"force_recompute": force_v2},
                    ),
                )
                pipeline_ok = bool(isinstance(pipeline_payload, dict) and pipeline_payload.get("ok"))
                print(
                    f"  V2: status={pipeline_status} ok={pipeline_ok} elapsed={pipeline_elapsed:.1f}s",
                    flush=True,
                )
                record["pipeline"] = {
                    "http_status": pipeline_status,
                    "ok": pipeline_ok,
                    "elapsed_seconds": round(pipeline_elapsed, 3),
                    "error": None if pipeline_ok else (
                        (pipeline_payload or {}).get("error")
                        if isinstance(pipeline_payload, dict)
                        else "non-json response"
                    ),
                    **_summarize_pipeline(pipeline_payload),
                }
            except Exception as exc:  # noqa: BLE001
                print(f"  V2: failed {exc}", flush=True)
                record["pipeline"]["error"] = str(exc)

            ai_body: dict[str, Any] = {
                "force_recompute": episode_id in force_ai_ids,
            }
            if provider:
                ai_body["provider_name"] = provider
            if model:
                ai_body["model_name"] = model

            try:
                ai_status, ai_payload, ai_elapsed = _run_with_timeout(
                    f"{ticker}#{episode_id} AI",
                    phase_timeout_seconds,
                    lambda: _post_json(
                        client,
                        f"/api/company/{ticker}/episode/{episode_id}/ai_explanation/run",
                        ai_body,
                    ),
                )
                ai_ok = bool(isinstance(ai_payload, dict) and ai_payload.get("ok"))
                ai_summary = _summarize_ai(ai_payload)
                cost = ai_summary.get("cost_estimate_usd")
                cost_text = f"${cost:.3f}" if isinstance(cost, (int, float)) else "n/a"
                print(
                    "  AI: "
                    f"status={ai_status} ok={ai_ok} elapsed={ai_elapsed:.1f}s "
                    f"quality={ai_summary.get('evidence_quality')} "
                    f"sources={ai_summary.get('source_count')} cost={cost_text}",
                    flush=True,
                )
                record["ai"] = {
                    "http_status": ai_status,
                    "ok": ai_ok,
                    "elapsed_seconds": round(ai_elapsed, 3),
                    "force_recompute": episode_id in force_ai_ids,
                    "error": None if ai_ok else (
                        (ai_payload or {}).get("error")
                        if isinstance(ai_payload, dict)
                        else "non-json response"
                    ),
                    **ai_summary,
                }
            except Exception as exc:  # noqa: BLE001
                print(f"  AI: failed {exc}", flush=True)
                record["ai"]["error"] = str(exc)

            summary["results"].append(record)
            _append_jsonl(jsonl_path, record)

            out_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            if index < len(EPISODES) and sleep_seconds > 0:
                time.sleep(sleep_seconds)

    total_cost = 0.0
    has_cost = False
    for record in summary["results"]:
        cost = (record.get("ai") or {}).get("cost_estimate_usd")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
            has_cost = True
    summary["total_ai_cost_estimate_usd"] = round(total_cost, 6) if has_cost else None
    summary["ok"] = all(
        (record.get("pipeline") or {}).get("ok")
        and (record.get("ai") or {}).get("ok")
        for record in summary["results"]
    )
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote summary: {out_path}", flush=True)
    print(f"Wrote JSONL:   {jsonl_path}", flush=True)
    print(f"Overall ok:    {summary['ok']}", flush=True)
    if summary["total_ai_cost_estimate_usd"] is not None:
        print(f"AI cost est.:  ${summary['total_ai_cost_estimate_usd']:.3f}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-v2", action="store_true")
    parser.add_argument(
        "--force-ai-episode",
        action="append",
        type=int,
        default=[10],
        help="Episode ID to force-recompute on the AI path. Repeatable.",
    )
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--phase-timeout-seconds", type=int, default=420)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_release_verification(
        force_v2=args.force_v2,
        force_ai_ids=set(args.force_ai_episode or []),
        provider=args.provider.strip(),
        model=args.model.strip(),
        sleep_seconds=args.sleep_seconds,
        dry_run=args.dry_run,
        phase_timeout_seconds=args.phase_timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
