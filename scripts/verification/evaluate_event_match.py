#!/usr/bin/env python3
"""Create the 2026 event-match verification artifact.

This is release/evaluation tooling, not application runtime. It reads the
cached 10-episode release artifact plus raw AI responses, then writes the
curated first-pass event-match labels used by the `/verification` shortcut
table.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "verification_event_match_2026.json"
DEFAULT_VERIFICATION_DIR = REPO_ROOT / "instance" / "verification"


CURATED_EVENT_MATCH: dict[str, dict[str, str]] = {
    "XOM#119": {
        "event_match": "yes",
        "rationale": "AI centers the explanation on the U.S.-Iran ceasefire, lower crude war premium, and energy-sector selloff.",
    },
    "NVDA#10": {
        "event_match": "yes",
        "rationale": "AI matches both parts of the hypothesis: U.S.-Iran risk-on/oil relief and NVIDIA AI/product news.",
    },
    "ORCL#481": {
        "event_match": "yes",
        "rationale": "AI ties the rally to Oracle AI-cloud backlog, earnings, and FY2027 guidance.",
    },
    "AMZN#69": {
        "event_match": "yes",
        "rationale": "AI identifies investor concern over Amazon AI capex and free-cash-flow pressure after earnings.",
    },
    "MSFT#31": {
        "event_match": "yes",
        "rationale": "AI explains the selloff through AI capex, Azure growth concerns, and software-sector derating.",
    },
    "ADBE#379": {
        "event_match": "partial",
        "rationale": "AI includes AI strategy/disruption pressure, but treats CEO-transition and earnings-package uncertainty as major co-drivers.",
    },
    "AAPL#1": {
        "event_match": "yes",
        "rationale": "AI connects the decline to Siri/Apple Intelligence delay concerns plus regulatory pressure.",
    },
    "LLY#44": {
        "event_match": "yes",
        "rationale": "AI identifies a firm-specific FDA approval / oral GLP-1 product catalyst.",
    },
    "ABNB#209": {
        "event_match": "yes",
        "rationale": "AI ties the rally to strong bookings, earnings, guidance, and travel-demand expectations.",
    },
    "TSLA#80": {
        "event_match": "partial",
        "rationale": "AI supports the broader Tesla AI narrative, but emphasizes UBS upgrade, AI-chip/custom-silicon optimism, and risk-on growth rally more than robotaxi specifically.",
    },
}


def latest_release_artifact(directory: Path) -> Path:
    files = sorted(
        directory.glob("release_verification_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"no release_verification_*.json files under {directory}")
    return files[-1]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def repo_relative(path_value: str | Path | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path_value)


def extract_ai_text(raw_path: str | None) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path)
    if not path.exists():
        return ""
    payload = load_json(path)
    chunks: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    if chunks:
        return "\n\n".join(chunks).strip()
    return str(payload.get("output_text") or "").strip()


def build_event_match_artifact(source_path: Path) -> dict[str, Any]:
    source = load_json(source_path)
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for item in source.get("results", []):
        ticker = item["ticker"]
        episode_id = int(item["episode_id"])
        key = f"{ticker}#{episode_id}"
        ai = item.get("ai") or {}
        curated = CURATED_EVENT_MATCH.get(key, {})
        event_match = curated.get("event_match", "unreviewed")
        counts[event_match] = counts.get(event_match, 0) + 1
        ai_text = extract_ai_text(ai.get("raw_response_path"))

        records.append(
            {
                "ticker": ticker,
                "episode_id": episode_id,
                "key": key,
                "event_theme": item.get("theme"),
                "event_match": event_match,
                "rationale": curated.get("rationale", ""),
                "ai_headline": ai.get("headline_explanation"),
                "ai_excerpt": ai_text[:900],
                "ai_raw_response_path": repo_relative(ai.get("raw_response_path")),
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": repo_relative(source_path),
        "review_type": "first-pass manual event-match review",
        "labels": {
            "yes": "AI explanation clearly matches the pre-selected event theme.",
            "partial": "AI mentions the theme but relies on other drivers too.",
            "no": "AI does not match the theme and does not provide a convincing alternative.",
            "better-alt": "AI identifies a more plausible driver than the selection hypothesis.",
            "unreviewed": "No curated label has been recorded yet.",
        },
        "summary": {
            "total": len(records),
            "counts": counts,
        },
        "results": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Release verification JSON. Defaults to latest instance/verification/release_verification_*.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the output file. Without this flag, print JSON to stdout.",
    )
    args = parser.parse_args()

    source_path = args.source or latest_release_artifact(DEFAULT_VERIFICATION_DIR)
    artifact = build_event_match_artifact(source_path)

    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote {args.output}")
    else:
        print(json.dumps(artifact, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
