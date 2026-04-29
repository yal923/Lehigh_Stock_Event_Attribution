from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import shutil
import sqlite3

from flask import Flask

from app.models import Company, Episode, EpisodeAiExplanationCache, db


@dataclass(frozen=True)
class DemoSeedResult:
    companies_inserted: int = 0
    companies_updated: int = 0
    episodes_inserted: int = 0
    ai_caches_inserted: int = 0
    pipeline_caches_copied: int = 0
    skipped: bool = False


def install_demo_seed(app: Flask, *, verbose: bool = True) -> DemoSeedResult:
    """
    Install the tracked 10-episode demo seed into local runtime state.

    This is intentionally additive: it does not overwrite an existing database
    or existing runtime cache files.
    """
    seed_dir = Path(app.root_path).parent / "docs" / "demo_cache" / "2026_release"
    seed_db = seed_dir / "demo_seed.sqlite"
    pipeline_cache_dir = seed_dir / "pipeline_cache"

    if not seed_db.exists():
        return DemoSeedResult(skipped=True)

    result = _merge_seed_database(seed_db)
    copied = _copy_pipeline_cache(pipeline_cache_dir)
    result = DemoSeedResult(
        companies_inserted=result.companies_inserted,
        companies_updated=result.companies_updated,
        episodes_inserted=result.episodes_inserted,
        ai_caches_inserted=result.ai_caches_inserted,
        pipeline_caches_copied=copied,
        skipped=False,
    )

    if verbose and _has_work(result):
        print(
            "[startup] Demo seed ready "
            f"(companies_inserted={result.companies_inserted}, "
            f"companies_updated={result.companies_updated}, "
            f"episodes_inserted={result.episodes_inserted}, "
            f"ai_caches_inserted={result.ai_caches_inserted}, "
            f"pipeline_caches_copied={result.pipeline_caches_copied})."
        )
    return result


def _merge_seed_database(seed_db: Path) -> DemoSeedResult:
    companies_inserted = 0
    companies_updated = 0
    episodes_inserted = 0
    ai_caches_inserted = 0

    with sqlite3.connect(seed_db) as conn:
        conn.row_factory = sqlite3.Row
        seed_companies = conn.execute(
            """
            select ticker, name, gics_sector, gics_sub_industry
            from companies
            order by ticker
            """
        ).fetchall()
        seed_episodes = conn.execute(
            """
            select e.*, c.ticker
            from episodes e
            join companies c on c.id = e.company_id
            order by e.id
            """
        ).fetchall()
        seed_ai_caches = conn.execute(
            """
            select episode_id, model_name, payload_json, cost_estimate_usd,
                   raw_response_path, created_at
            from episode_ai_explanation_cache
            order by episode_id
            """
        ).fetchall()

    for row in seed_companies:
        ticker = str(row["ticker"]).upper()
        company = Company.query.filter_by(ticker=ticker).first()
        if company is None:
            db.session.add(Company(
                ticker=ticker,
                name=row["name"],
                gics_sector=row["gics_sector"],
                gics_sub_industry=row["gics_sub_industry"],
            ))
            companies_inserted += 1
            continue

        dirty = False
        for attr in ("name", "gics_sector", "gics_sub_industry"):
            if getattr(company, attr) != row[attr]:
                setattr(company, attr, row[attr])
                dirty = True
        if dirty:
            companies_updated += 1

    db.session.flush()

    companies_by_ticker = {
        company.ticker: company
        for company in Company.query
        .filter(Company.ticker.in_([str(row["ticker"]).upper() for row in seed_companies]))
        .all()
    }

    for row in seed_episodes:
        episode_id = int(row["id"])
        if Episode.query.get(episode_id) is not None:
            continue

        ticker = str(row["ticker"]).upper()
        company = companies_by_ticker.get(ticker)
        if company is None:
            continue

        db.session.add(Episode(
            id=episode_id,
            company_id=company.id,
            mode=row["mode"],
            threshold=float(row["threshold"]),
            lookback_days=int(row["lookback_days"]),
            horizon_n=int(row["horizon_n"]),
            merge_gap_days=int(row["merge_gap_days"]),
            peak_date=_parse_date(row["peak_date"]),
            window_start=_parse_date(row["window_start"]),
            window_end=_parse_date(row["window_end"]),
            pct_move=float(row["pct_move"]),
            direction=row["direction"],
            signal_count=int(row["signal_count"]),
            created_at=_parse_datetime(row["created_at"]),
        ))
        episodes_inserted += 1

    db.session.flush()

    for row in seed_ai_caches:
        episode_id = int(row["episode_id"])
        if Episode.query.get(episode_id) is None:
            continue
        if EpisodeAiExplanationCache.query.filter_by(episode_id=episode_id).first() is not None:
            continue

        db.session.add(EpisodeAiExplanationCache(
            episode_id=episode_id,
            model_name=row["model_name"],
            payload_json=row["payload_json"],
            cost_estimate_usd=row["cost_estimate_usd"],
            raw_response_path=row["raw_response_path"],
            created_at=_parse_datetime(row["created_at"]),
        ))
        ai_caches_inserted += 1

    db.session.commit()
    return DemoSeedResult(
        companies_inserted=companies_inserted,
        companies_updated=companies_updated,
        episodes_inserted=episodes_inserted,
        ai_caches_inserted=ai_caches_inserted,
    )


def _copy_pipeline_cache(source_dir: Path) -> int:
    if not source_dir.exists():
        return 0

    dest_dir = Path(__file__).resolve().parent / "_episode_cache"
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source in sorted(source_dir.glob("episode_*.json")):
        dest = dest_dir / source.name
        if dest.exists():
            continue
        shutil.copy2(source, dest)
        copied += 1
    return copied


def _parse_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value)[:10])


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _has_work(result: DemoSeedResult) -> bool:
    return any((
        result.companies_inserted,
        result.companies_updated,
        result.episodes_inserted,
        result.ai_caches_inserted,
        result.pipeline_caches_copied,
    ))
