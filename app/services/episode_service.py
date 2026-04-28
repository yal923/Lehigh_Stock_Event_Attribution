from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta

from app.config.settings import get_settings
from app.domain.episode_definition import EpisodeParams, validate_episode_params
from app.services.company_service import (
    ensure_company_episodes_signal_v2,
    get_company_episodes,
)
from app.services.pipeline_service import run_episode_pipeline


def parse_episode_params(args: Mapping[str, object]) -> EpisodeParams:
    settings = get_settings()

    def _get_int(name: str, default: int) -> int:
        raw = str(args.get(name, "") or "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except Exception:
            return default

    def _get_float(name: str, default: float) -> float:
        raw = str(args.get(name, "") or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    horizon_n = _get_int("horizon_n", settings.default_episode_horizon_n)
    threshold = _get_float("threshold", settings.default_episode_threshold)
    merge_gap_days = _get_int("merge_gap_days", settings.default_episode_merge_gap_days)
    lookback_days = _get_int("lookback_days", settings.default_episode_lookback_days)

    return validate_episode_params(
        horizon_n=horizon_n,
        threshold=threshold,
        merge_gap_days=merge_gap_days,
        lookback_days=lookback_days,
    )


def ensure_and_get_company_episodes(company, params: EpisodeParams):
    mode, _created = ensure_company_episodes_signal_v2(
        company,
        horizon_n=params.horizon_n,
        threshold=params.threshold,
        merge_gap_days=params.merge_gap_days,
        lookback_days=params.lookback_days,
    )
    episodes = get_company_episodes(
        company,
        lookback_days=params.lookback_days,
        mode=mode,
        threshold=params.threshold,
    )
    return mode, episodes


def _infer_horizon_from_mode(mode: str) -> int:
    if mode.startswith("v2h") and "g" in mode:
        try:
            middle = mode[3:].split("g", 1)[0]
            return max(1, int(middle))
        except Exception:
            return 1
    return 1


def build_episode_metrics(company_id: int, episode) -> dict:
    if episode is None or episode.window_start is None or episode.window_end is None:
        return {
            "start": None,
            "end": None,
            "peak": None,
            "direction": None,
            "peak_pct": None,
            "cumulative_pct": None,
            "signal_count": 0,
            "signal_anchors": [],
            "daily_pct": [],
        }

    horizon_n = _infer_horizon_from_mode(episode.mode or "")
    lag = max(horizon_n - 1, 1)
    start = episode.window_start
    end = episode.window_end
    pre_start = start - timedelta(days=max(lag, 1) + 7)

    import yfinance as yf
    ticker = episode.company.ticker
    hist = yf.Ticker(ticker).history(
        start=pre_start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
    )

    # Build a simple list of (date, close) in ascending date order.
    raw_rows = []
    if not hist.empty:
        for ts, row in hist.iterrows():
            d = ts.date() if hasattr(ts, "date") else ts
            raw_rows.append((d, float(row["Close"]) if row["Close"] is not None else None))
    raw_rows.sort(key=lambda x: x[0])

    daily_pct: list[dict] = []
    signal_returns: list[float] = []
    signal_anchors: list[dict] = []
    closes: list[float | None] = []
    dates = []
    prev_close = None
    window_closes: list[float] = []

    for row_date, close_val in raw_rows:
        closes.append(close_val)
        dates.append(row_date)

        if close_val is not None and start <= row_date <= end:
            window_closes.append(close_val)
            pct = None
            if prev_close is not None and prev_close != 0:
                pct = (close_val / prev_close - 1.0) * 100.0
            daily_pct.append(
                {
                    "date": row_date.isoformat(),
                    "close": round(float(close_val), 4),
                    "pct": round(float(pct), 2) if pct is not None else None,
                }
            )

        if close_val is not None:
            prev_close = close_val

    threshold = float(episode.threshold) if episode.threshold is not None else 0.0
    if threshold > 0 and len(closes) > lag:
        for idx in range(lag, len(closes)):
            close_now = closes[idx]
            close_prev = closes[idx - lag]
            anchor_date = dates[idx]
            baseline_date = dates[idx - lag]
            if close_now is None or close_prev is None or close_prev == 0:
                continue
            if not (start <= anchor_date <= end):
                continue
            ret = (close_now / close_prev) - 1.0
            if abs(ret) >= threshold:
                signal_returns.append(float(ret))
                signal_anchors.append(
                    {
                        "anchor_date": anchor_date.isoformat(),
                        "baseline_date": baseline_date.isoformat(),
                        "ret_pct": round(float(ret) * 100.0, 2),
                        "direction": "up" if ret >= 0 else "down",
                    }
                )

    cumulative_pct = None
    if len(window_closes) >= 2 and window_closes[0] != 0:
        cumulative_pct = (window_closes[-1] / window_closes[0] - 1.0) * 100.0

    peak_pct = None
    if signal_returns:
        peak_ret = max(signal_returns, key=lambda x: abs(x))
        peak_pct = round(float(peak_ret) * 100.0, 2)
    elif episode.pct_move is not None:
        peak_pct = float(episode.pct_move)

    return {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "peak": episode.peak_date.isoformat() if episode.peak_date else None,
        "direction": episode.direction,
        "peak_pct": peak_pct,
        "cumulative_pct": round(float(cumulative_pct), 2) if cumulative_pct is not None else None,
        "signal_count": len(signal_returns),
        "signal_anchors": signal_anchors,
        "daily_pct": daily_pct,
    }


def build_episode_decision_summary(episode) -> dict:
    decision_summary = {
        "outward_status": None,
        "driver_hypothesis": None,
        "headline_explanation": None,
        "explanation_confidence": None,
        "manual_review": None,
    }

    try:
        pipeline_result = run_episode_pipeline(episode, force_recompute=False)
        decision = pipeline_result.get("decision") or {}
        decision_summary = {
            "outward_status": decision.get("outward_status"),
            "driver_hypothesis": decision.get("driver_hypothesis"),
            "headline_explanation": decision.get("headline_explanation"),
            "explanation_confidence": decision.get("explanation_confidence"),
            "manual_review": decision.get("manual_review"),
        }
    except Exception:
        pass

    return decision_summary
