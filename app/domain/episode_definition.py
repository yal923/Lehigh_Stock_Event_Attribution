from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeParams:
    horizon_n: int = 3
    threshold: float = 0.05
    merge_gap_days: int = 2
    lookback_days: int = 365


def build_signal_v2_mode(horizon_n: int, merge_gap_days: int) -> str:
    # Episode.mode is String(16), so keep mode compact but parameterized.
    return f"v2h{horizon_n}g{merge_gap_days}"[:16]


def validate_episode_params(
    horizon_n: int,
    threshold: float,
    merge_gap_days: int,
    lookback_days: int,
) -> EpisodeParams:
    if horizon_n < 1 or horizon_n > 60:
        raise ValueError("horizon_n must be between 1 and 60")
    if threshold <= 0 or threshold > 0.30:
        raise ValueError("threshold must be in (0, 0.30]")
    if merge_gap_days < 0 or merge_gap_days > 10:
        raise ValueError("merge_gap_days must be between 0 and 10")
    if lookback_days < 30 or lookback_days > 1260:
        raise ValueError("lookback_days must be between 30 and 1260")

    return EpisodeParams(
        horizon_n=horizon_n,
        threshold=threshold,
        merge_gap_days=merge_gap_days,
        lookback_days=lookback_days,
    )


def detect_signals_from_prices(
    price_df: pd.DataFrame,
    horizon_n: int,
    threshold: float,
) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    df = df.sort_values("date").set_index("date")
    # Use h-day inclusive window by trading rows:
    # h=3 means [t-2, t-1, t], so lag=2.
    lag = max(horizon_n - 1, 1)
    df["ret"] = df["close"] / df["close"].shift(lag) - 1
    baseline_series = pd.Series(df.index, index=df.index).shift(lag)
    df["baseline_date"] = baseline_series

    signals = df[df["ret"].abs() >= threshold].copy()
    if signals.empty:
        return pd.DataFrame()

    signals = signals.reset_index().rename(columns={"date": "anchor_date"})
    signals["direction"] = signals["ret"].apply(lambda x: "up" if x > 0 else "down")
    return signals[["anchor_date", "baseline_date", "ret", "direction"]]


def merge_signals_to_episodes(
    signals_df: pd.DataFrame,
    merge_gap_days: int,
) -> list[dict]:
    if signals_df is None or signals_df.empty:
        return []

    signals_df = signals_df.sort_values("anchor_date")
    episodes = []
    current = []
    prev_date = None

    for _, row in signals_df.iterrows():
        dt = row["anchor_date"]
        if prev_date is None:
            current.append(row)
        else:
            gap = (dt - prev_date).days
            if gap <= merge_gap_days:
                current.append(row)
            else:
                episodes.append(current)
                current = [row]
        prev_date = dt

    if current:
        episodes.append(current)

    out = []
    for ep in episodes:
        ep_df = pd.DataFrame(ep).reset_index(drop=True)
        peak_pos = int(np.argmax(ep_df["ret"].abs().to_numpy()))
        peak_anchor = ep_df.iloc[peak_pos]["anchor_date"]
        peak_ret = ep_df.iloc[peak_pos]["ret"]
        peak_direction = ep_df.iloc[peak_pos]["direction"]
        out.append(
            {
                "start": pd.Timestamp(ep_df["baseline_date"].min()).isoformat(),
                "end": pd.Timestamp(ep_df["anchor_date"].max()).isoformat(),
                "peak_date": pd.Timestamp(peak_anchor).isoformat(),
                "pct": round(float(peak_ret) * 100.0, 2),
                "direction": str(peak_direction),
                "signal_count": int(len(ep_df)),
            }
        )

    out.sort(key=lambda x: x["peak_date"], reverse=True)
    return out
