#!/usr/bin/env python3
"""Build the tracked 2026 release demo cache bundle.

The Flask instance directory is intentionally ignored because it contains local
runtime state. This script copies only the 10 paid release-verification AI
artifacts needed for the presentation/paper demo into `docs/demo_cache/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERIFICATION_DIR = REPO_ROOT / "instance" / "verification"
DEFAULT_BUNDLE_DIR = REPO_ROOT / "docs" / "demo_cache" / "2026_release"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def latest_release_artifact(directory: Path) -> Path:
    files = sorted(
        directory.glob("release_verification_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"no release_verification_*.json files under {directory}")
    return files[-1]


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def rewrite_release_summary(
    *,
    source_path: Path,
    bundle_dir: Path,
    release_summary_path: Path,
    release_jsonl_path: Path | None,
) -> dict[str, Any]:
    payload = load_json(source_path)
    ai_raw_dir = bundle_dir / "ai_raw"
    copied_raw_paths: list[str] = []

    for item in payload.get("results") or []:
        ai = item.get("ai") or {}
        raw_path_value = ai.get("raw_response_path")
        if not raw_path_value:
            continue

        raw_source = Path(raw_path_value)
        if not raw_source.exists():
            raise FileNotFoundError(f"raw AI response not found: {raw_source}")

        raw_target = ai_raw_dir / raw_source.name
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_source, raw_target)
        ai["raw_response_path"] = repo_relative(raw_target)
        copied_raw_paths.append(repo_relative(raw_target))

    payload["output_path"] = repo_relative(release_summary_path)
    payload["jsonl_path"] = repo_relative(release_jsonl_path) if release_jsonl_path else None
    payload["demo_cache_bundle"] = {
        "bundle_dir": repo_relative(bundle_dir),
        "source_artifact": repo_relative(source_path),
        "raw_ai_response_count": len(copied_raw_paths),
        "raw_ai_response_paths": copied_raw_paths,
    }
    return payload


def rewrite_release_jsonl(
    *,
    source_jsonl: Path,
    release_jsonl_path: Path,
    bundle_dir: Path,
) -> None:
    release_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    ai_raw_dir = bundle_dir / "ai_raw"
    with source_jsonl.open("r", encoding="utf-8") as src, release_jsonl_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                dst.write(line)
                continue
            row = json.loads(line)
            ai = row.get("ai") or {}
            raw_path_value = ai.get("raw_response_path")
            if raw_path_value:
                ai["raw_response_path"] = repo_relative(ai_raw_dir / Path(raw_path_value).name)
            dst.write(json.dumps(row, ensure_ascii=False))
            dst.write("\n")


def build_bundle(source_path: Path, bundle_dir: Path) -> dict[str, Path]:
    source_path = source_path.resolve()
    bundle_dir = bundle_dir.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    release_summary_path = bundle_dir / "release_verification.json"
    source_jsonl = source_path.with_suffix(".jsonl")
    release_jsonl_path = bundle_dir / "release_verification.jsonl"

    payload = rewrite_release_summary(
        source_path=source_path,
        bundle_dir=bundle_dir,
        release_summary_path=release_summary_path,
        release_jsonl_path=release_jsonl_path if source_jsonl.exists() else None,
    )
    write_json(release_summary_path, payload)

    if source_jsonl.exists():
        rewrite_release_jsonl(
            source_jsonl=source_jsonl,
            release_jsonl_path=release_jsonl_path,
            bundle_dir=bundle_dir,
        )

    return {
        "bundle_dir": bundle_dir,
        "release_summary": release_summary_path,
        "release_jsonl": release_jsonl_path if source_jsonl.exists() else None,
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
        "--output-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help="Tracked demo cache bundle output directory.",
    )
    args = parser.parse_args()

    source_path = args.source or latest_release_artifact(DEFAULT_VERIFICATION_DIR)
    result = build_bundle(source_path, args.output_dir)
    print(f"wrote {result['release_summary']}")
    if result["release_jsonl"]:
        print(f"wrote {result['release_jsonl']}")
    print(f"bundle {result['bundle_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
