from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from app.domain.episode_definition import (
    build_signal_v2_mode,
    detect_signals_from_prices,
    merge_signals_to_episodes,
    validate_episode_params,
)
from app.models import Company, Episode, db


def detect_episodes(
    company: Company,
    mode: str = "signal_v2",
    threshold: float = 0.05,
    lookback_days: int = 365,
    horizon_n: int = 3,
    merge_gap_days: int = 2,
):
    """Detect price-movement episodes for one company on the signal-v2 path."""
    if mode != "signal_v2":
        raise ValueError("Only signal_v2 episode detection is supported")

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days + 10)

    hist = yf.Ticker(company.ticker).history(
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )
    if hist.empty:
        return []

    df = pd.DataFrame({
        "date": [ts.date() for ts in hist.index],
        "close": hist["Close"].values,
    }).set_index("date").sort_index()
    df = df.dropna(subset=["close"]).copy()

    if df.empty:
        return []

    params = validate_episode_params(
        horizon_n=horizon_n,
        threshold=threshold,
        merge_gap_days=merge_gap_days,
        lookback_days=lookback_days,
    )
    signal_input = df.reset_index()[["date", "close"]].copy()
    signals = detect_signals_from_prices(
        signal_input,
        horizon_n=params.horizon_n,
        threshold=params.threshold,
    )
    return merge_signals_to_episodes(signals, merge_gap_days=params.merge_gap_days)


def upsert_episodes_for_company(
    company: Company,
    mode: str = "signal_v2",
    threshold: float = 0.05,
    lookback_days: int = 365,
    horizon_n: int = 3,
    merge_gap_days: int = 2,
) -> int:
    """Detect and upsert signal-v2 episodes for one company."""
    if mode != "signal_v2":
        raise ValueError("Only signal_v2 episode upsert is supported")

    params = validate_episode_params(
        horizon_n=horizon_n,
        threshold=threshold,
        merge_gap_days=merge_gap_days,
        lookback_days=lookback_days,
    )
    mode_for_storage = build_signal_v2_mode(
        horizon_n=params.horizon_n,
        merge_gap_days=params.merge_gap_days,
    )

    candidates = detect_episodes(
        company=company,
        mode=mode,
        threshold=params.threshold,
        lookback_days=params.lookback_days,
        horizon_n=params.horizon_n,
        merge_gap_days=params.merge_gap_days,
    )

    if not candidates:
        return 0

    existing = Episode.query.filter_by(
        company_id=company.id,
        mode=mode_for_storage,
        threshold=params.threshold,
        lookback_days=params.lookback_days,
    ).all()
    existing_by_peak = {e.peak_date: e for e in existing}

    created = 0
    changed = 0

    for ep in candidates:
        peak_date = datetime.fromisoformat(ep["peak_date"]).date()
        window_start = datetime.fromisoformat(ep["start"]).date()
        window_end = datetime.fromisoformat(ep["end"]).date()

        existed = existing_by_peak.get(peak_date)
        if existed is not None:
            dirty = False
            if existed.window_start != window_start:
                existed.window_start = window_start
                dirty = True
            if existed.window_end != window_end:
                existed.window_end = window_end
                dirty = True
            if existed.pct_move != ep["pct"]:
                existed.pct_move = ep["pct"]
                dirty = True
            if existed.direction != ep["direction"]:
                existed.direction = ep["direction"]
                dirty = True
            if dirty:
                changed += 1
            continue

        episode = Episode(
            company_id=company.id,
            mode=mode_for_storage,
            threshold=params.threshold,
            lookback_days=params.lookback_days,
            horizon_n=params.horizon_n,
            merge_gap_days=params.merge_gap_days,
            peak_date=peak_date,
            window_start=window_start,
            window_end=window_end,
            pct_move=ep["pct"],
            direction=ep["direction"],
            signal_count=int(ep.get("signal_count", 1)),
        )
        db.session.add(episode)
        created += 1

    if created > 0 or changed > 0:
        db.session.commit()

    return created


def upsert_episodes_for_all_companies(
    mode: str = "signal_v2",
    threshold: float = 0.05,
    lookback_days: int = 365,
    horizon_n: int = 3,
    merge_gap_days: int = 2,
) -> int:
    """Detect and upsert signal-v2 episodes for all companies in the DB."""
    if mode != "signal_v2":
        raise ValueError("Only signal_v2 episode generation is supported")

    total_created = 0
    companies = Company.query.order_by(Company.ticker.asc()).all()
    for company in companies:
        created = upsert_episodes_for_company(
            company=company,
            mode=mode,
            threshold=threshold,
            lookback_days=lookback_days,
            horizon_n=horizon_n,
            merge_gap_days=merge_gap_days,
        )
        total_created += created

    return total_created
