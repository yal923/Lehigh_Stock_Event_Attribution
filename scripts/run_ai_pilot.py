"""
Run the DEC-203 Day 1 PM AI Explanation pilot batch.

The pilot selects 15 episodes from the latest structural verification artifact:
5 V2 SUPPORTED, 5 V2 LIMITED_EVIDENCE, and 5 V2 NO_EVIDENCE, while covering
dominant / mixed / degraded routing where the artifact has those buckets.

Each selected episode is executed through the same public Flask endpoint used by
the UI:

    POST /api/company/<ticker>/episode/<id>/ai_explanation/run

Outputs:
  - instance/ai_pilot/ai_pilot_<timestamp>.json
  - instance/ai_pilot/ai_pilot_<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TARGET_STATUSES = ("SUPPORTED", "LIMITED_EVIDENCE", "NO_EVIDENCE")
TARGET_ROUTING = ("dominant", "mixed", "degraded")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _latest_verification_file() -> Path:
    candidates = sorted(
        (_repo_root() / "instance" / "verification").glob("verify_*.json")
    )
    candidates = [
        path
        for path in candidates
        if not path.name.endswith("_current.json")
        and not path.name.endswith("_episodes.json")
    ]
    if not candidates:
        raise FileNotFoundError("No verification summary found in instance/verification")
    return candidates[-1]


def _load_verification(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError(f"Invalid verification summary: {path}")
    return payload


def _select_pilot_episodes(
    episodes: list[dict[str, Any]],
    *,
    per_status: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    for status in TARGET_STATUSES:
        bucket = [
            item
            for item in episodes
            if item.get("ok")
            and item.get("outward_status") == status
            and item.get("episode_id") is not None
            and item.get("ticker")
        ]
        by_routing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in bucket:
            by_routing[str(item.get("routing_mode") or "unknown")].append(item)

        status_selected: list[dict[str, Any]] = []

        # First pass: maximize routing coverage inside each V2 status bucket.
        for routing in TARGET_ROUTING:
            if len(status_selected) >= per_status:
                break
            for item in by_routing.get(routing, []):
                episode_id = int(item["episode_id"])
                if episode_id in selected_ids:
                    continue
                status_selected.append(item)
                selected_ids.add(episode_id)
                break

        # Fill remaining slots in artifact order, preserving verification priority.
        for item in bucket:
            if len(status_selected) >= per_status:
                break
            episode_id = int(item["episode_id"])
            if episode_id in selected_ids:
                continue
            status_selected.append(item)
            selected_ids.add(episode_id)

        if len(status_selected) < per_status:
            raise RuntimeError(
                f"Only found {len(status_selected)} episodes for {status}; "
                f"need {per_status}"
            )

        selected.extend(status_selected)

    return selected


def _output_paths(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return (
        out_dir / f"ai_pilot_{stamp}.json",
        out_dir / f"ai_pilot_{stamp}.jsonl",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _live_v2_metadata(episode_id: int) -> dict[str, Any]:
    from app.cache import load_episode_cache

    cached = load_episode_cache(episode_id)
    bundle = cached.get("analysis_v2_bundle") or {}
    if not isinstance(bundle, dict):
        return {}
    return {
        "routing_mode": (bundle.get("branch_plan") or {}).get("routing_mode"),
        "primary_mode": (bundle.get("branch_plan") or {}).get("primary_mode"),
        "outward_status": (bundle.get("decision") or {}).get("outward_status"),
        "quality_status": (bundle.get("attribution") or {}).get("quality_status"),
        "headline_explanation": (bundle.get("decision") or {}).get(
            "headline_explanation"
        ),
    }


def _run_one(client: Any, item: dict[str, Any], *, force_ai: bool) -> dict[str, Any]:
    ticker = str(item["ticker"])
    episode_id = int(item["episode_id"])
    url = f"/api/company/{ticker}/episode/{episode_id}/ai_explanation/run"

    started = time.monotonic()
    response = client.post(url, json={"force_recompute": force_ai})
    elapsed = time.monotonic() - started
    payload = response.get_json(silent=True)

    record: dict[str, Any] = {
        "ticker": ticker,
        "episode_id": episode_id,
        "peak_date": item.get("peak_date"),
        "verification_baseline": {
            "routing_mode": item.get("routing_mode"),
            "outward_status": item.get("outward_status"),
            "quality_status": item.get("quality_status"),
        },
        "http_status": response.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "ok": bool(isinstance(payload, dict) and payload.get("ok")),
        "error": None,
        "live_v2": _live_v2_metadata(episode_id),
        "ai": None,
    }

    if not isinstance(payload, dict):
        record["error"] = response.get_data(as_text=True)[:1000]
        return record

    if not payload.get("ok"):
        record["error"] = payload.get("error") or "endpoint returned ok=false"
        return record

    ai_payload = payload.get("ai_explanation") or {}
    if isinstance(ai_payload, dict):
        record["ai"] = {
            "provider_name": ai_payload.get("provider_name"),
            "model_name": ai_payload.get("model_name"),
            "headline_explanation": ai_payload.get("headline_explanation"),
            "evidence_quality": ai_payload.get("evidence_quality"),
            "source_count": len(ai_payload.get("sources") or []),
            "cost_estimate_usd": ai_payload.get("cost_estimate_usd"),
            "computed_at": ai_payload.get("computed_at"),
            "raw_response_path": ai_payload.get("raw_response_path"),
        }
    return record


def run_pilot(
    *,
    verification_file: Path,
    per_status: int,
    force_ai: bool,
    sleep_seconds: float,
    dry_run: bool,
) -> dict[str, Any]:
    sys.path.insert(0, str(_repo_root()))
    os.chdir(_repo_root())

    verification = _load_verification(verification_file)
    selected = _select_pilot_episodes(
        verification["episodes"],
        per_status=per_status,
    )

    out_path, jsonl_path = _output_paths(_repo_root() / "instance" / "ai_pilot")

    summary: dict[str, Any] = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "verification_file": str(verification_file),
        "force_ai": force_ai,
        "per_status": per_status,
        "dry_run": dry_run,
        "output_path": str(out_path),
        "jsonl_path": str(jsonl_path),
        "selected": selected,
        "results": [],
    }

    print("Selected pilot episodes:")
    for index, item in enumerate(selected, start=1):
        print(
            f"{index:02d}. {item['ticker']:>6} ep={item['episode_id']:<5} "
            f"v2={item['outward_status']:<16} routing={item['routing_mode']:<9} "
            f"quality={item['quality_status']}"
        )

    if dry_run:
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nDry run only. Wrote selection to {out_path}")
        return summary

    from app import create_app

    app = create_app()
    with app.app_context():
        client = app.test_client()

        for index, item in enumerate(selected, start=1):
            print(
                f"\n[{index:02d}/{len(selected):02d}] "
                f"{item['ticker']} episode {item['episode_id']} "
                f"({item['outward_status']} / {item['routing_mode']})"
            )
            record = _run_one(client, item, force_ai=force_ai)
            summary["results"].append(record)
            _append_jsonl(jsonl_path, record)

            ai = record.get("ai") or {}
            if record["ok"]:
                cost = ai.get("cost_estimate_usd")
                cost_text = "n/a" if cost is None else f"${float(cost):.4f}"
                print(
                    "  OK "
                    f"{record['elapsed_seconds']:.1f}s "
                    f"provider={ai.get('provider_name') or '?'} "
                    f"ai_quality={ai.get('evidence_quality')} "
                    f"sources={ai.get('source_count')} "
                    f"cost={cost_text}"
                )
            else:
                print(f"  FAIL {record['http_status']} {record.get('error')}")

            out_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            if sleep_seconds > 0 and index < len(selected):
                time.sleep(sleep_seconds)

    total = len(summary["results"])
    ok = sum(1 for item in summary["results"] if item.get("ok"))
    summary["totals"] = {
        "total": total,
        "ok": ok,
        "failed": total - ok,
        "estimated_cost_usd": round(
            sum(
                float(((item.get("ai") or {}).get("cost_estimate_usd")) or 0.0)
                for item in summary["results"]
                if item.get("ok")
            ),
            6,
        ),
    }
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote summary: {out_path}")
    print(f"Wrote episode log: {jsonl_path}")
    print(f"Totals: {summary['totals']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verification-file",
        type=Path,
        default=None,
        help="Verification summary JSON. Defaults to latest instance/verification/verify_*.json.",
    )
    parser.add_argument("--per-status", type=int, default=5)
    parser.add_argument(
        "--force-ai",
        action="store_true",
        help="Force recompute AI explanations even if an AI cache row exists.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only select and print the 15-episode pilot set.",
    )
    args = parser.parse_args()

    verification_file = args.verification_file or _latest_verification_file()
    run_pilot(
        verification_file=verification_file,
        per_status=args.per_status,
        force_ai=args.force_ai,
        sleep_seconds=args.sleep_seconds,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
