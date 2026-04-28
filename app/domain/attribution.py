from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import yfinance as yf

from app.models import Episode


DEFAULT_MARKET_PROXY = "SPY"
DEFAULT_ESTIMATION_WINDOW = 756  # ~3 years of trading days (DEC-172)

# EWMA decay factor for weighted-OLS beta estimation (DEC-199).
# lambda = 0.97 -> halflife ≈ 23 trading days ≈ 1 month.
# Weights w_t = lambda^(T-1-t); most recent training day weight = 1.0.
EWMA_DECAY_FACTOR = 0.97

# SPDR Select Sector ETF mapping.
SECTOR_TO_SPDR_PROXY = {
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# GICS sub-industry ETF overrides (DEC-199).
# Takes precedence over SECTOR_TO_SPDR_PROXY when the company's gics_sub_industry matches.
# Sub-industries not listed here fall back to the sector proxy.
# Rationale, ETF selection criteria, and SP500 coverage notes are archived in
# docs/capstone_paper_materials.md under "Sub-Industry Proxy Design".
SUB_INDUSTRY_TO_PROXY = {
    # Information Technology — Software / Semiconductors
    "Application Software": "IGV",
    "Systems Software": "IGV",
    "Semiconductors": "SMH",
    "Semiconductor Materials & Equipment": "SMH",
    # Health Care — Biotech / Pharma / Equipment
    "Biotechnology": "IBB",
    "Pharmaceuticals": "XPH",
    "Health Care Equipment": "IHI",
    # Financials — Banks / Capital Markets / Insurance
    "Diversified Banks": "KBE",
    "Regional Banks": "KRE",
    "Investment Banking & Brokerage": "KCE",
    "Financial Exchanges & Data": "KCE",
    "Insurance Brokers": "KIE",
    "Property & Casualty Insurance": "KIE",
    "Life & Health Insurance": "KIE",
    "Multi-line Insurance": "KIE",
    "Reinsurance": "KIE",
    # Energy
    "Oil & Gas Exploration & Production": "XOP",
    "Oil & Gas Equipment & Services": "OIH",
    # Industrials
    "Aerospace & Defense": "ITA",
    "Passenger Airlines": "JETS",
    # Consumer Discretionary
    "Homebuilding": "ITB",
    "Home Improvement Retail": "XRT",
    # Materials
    "Gold": "GDX",
}

# Plain-English descriptions for every OOS failure reason (DEC-175).
# These are shown directly in the UI — no snake_case codes exposed to analysts.
_OOS_PLAIN_REASONS: dict[str, str] = {
    "company_missing": (
        "This episode has no associated company record. "
        "Attribution cannot determine which ticker to analyze."
    ),
    "missing_gics_sector": (
        "This company is missing a GICS sector classification in the database. "
        "Attribution requires a sector assignment to select an industry proxy ETF."
    ),
    "unsupported_gics_sector": (
        "This company's GICS sector does not have a corresponding SPDR industry ETF "
        "in the current mapping. Attribution cannot identify an industry proxy."
    ),
    "price_data_empty": (
        "The market data provider returned no price data for this ticker and window. "
        "Attribution cannot proceed without price history."
    ),
    "missing_required_series": (
        "Price data for one or more required series could not be retrieved. "
        "Attribution requires the stock, the S&P 500 ETF (SPY), and the "
        "sector ETF to all be available from the market data provider."
    ),
    "episode_window_empty": (
        "No price data was available within the episode window itself. "
        "Attribution cannot decompose returns for an empty episode window."
    ),
    "insufficient_train_data": (
        "Fewer than 10 pre-episode trading days were available for regression estimation. "
        "This is a data availability issue rather than a model quality issue."
    ),
}

# Minimum pre-episode observations needed for a meaningful OLS fit.
# 3 parameters (intercept + beta_mkt + beta_ind) require strictly more observations.
_MIN_TRAIN_OBSERVATIONS = 10

# Quality gate thresholds (DEC-171).
_POOR_FIT_ADJ_R2_THRESHOLD = 0.10
_CAUTION_ADJ_R2_THRESHOLD = 0.20
_SIGNIFICANCE_T_THRESHOLD = 1.96  # two-sided 5% level

# Maximum number of missing pre-episode trading days tolerated before retry.
# Shortfall above this threshold triggers an OOS result without retry.
_ALIGNMENT_SHORTFALL_THRESHOLD = 100

# Below this |episode_return_total|, net-contribution shares are numerically
# unstable (divide-by-near-zero) and communicate no useful direction. Fall back
# to the absolute-magnitude ratios in that regime (DEC-199 G).
_NEAR_ZERO_EPISODE_RETURN_THRESHOLD = 0.005

# VIX macro-shock flag thresholds (DEC-199 H).
# Flag fires when EITHER the episode's max VIX pierces the training p95 level
# OR the episode's average VIX is materially elevated vs the training median.
_VIX_TICKER = "^VIX"
_VIX_SHOCK_AVG_RATIO_THRESHOLD = 1.5


@dataclass(frozen=True)
class AttributionResult:
    ticker: str
    market_ratio: float
    industry_ratio: float
    firm_ratio: float
    alpha_total: float
    market_total: float
    industry_total: float
    firm_total: float
    episode_return_total: float
    adj_r2: float | None
    recon_error_bps: float | None
    # --- quality fields (DEC-173) ---
    quality_status: str           # OOS | POOR_FIT | CAUTION | PASS
    reason: str | None            # plain English OOS failure message; None for PASS/POOR_FIT/CAUTION
    reason_code: str              # quality_status.lower() for non-OOS; specific code for OOS
    # --- new regression diagnostics (DEC-171) ---
    beta_mkt_t_stat: float | None
    beta_ind_t_stat: float | None
    f_stat: float | None
    f_pvalue: float | None
    fit_quality_note: str | None  # plain English fit warning; set for POOR_FIT / CAUTION
    # --- net-contribution shares (DEC-199 G) ---
    # component_total / episode_return_total. Preserves sign: a negative share
    # means that component pushed against the episode's overall direction.
    # None when near_zero_episode_return is True (UI falls back to *_ratio).
    market_share: float | None
    industry_share: float | None
    firm_share: float | None
    alpha_share: float | None
    near_zero_episode_return: bool
    # --- industry proxy provenance (DEC-199 F) ---
    # The actual ETF used for the industry factor, and whether it came from the
    # sub-industry override table ("sub_industry") or the GICS-sector fallback
    # ("sector"). None for OOS results.
    industry_proxy_ticker: str | None
    industry_proxy_source: str | None
    # --- VIX macro-shock diagnostics (DEC-199 H) ---
    # VIX is fetched separately and degrades gracefully: if fetch fails, all four
    # metrics are None and `macro_shock_suspected` is False. The flag is additive
    # and does NOT change `quality_status`; it triggers a UI caveat chip and
    # escalates outward.manual_review at the news-v2 layer.
    vix_estimation_p50: float | None
    vix_estimation_p95: float | None
    vix_episode_max: float | None
    vix_episode_avg: float | None
    macro_shock_suspected: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_ratio(x: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(abs(x) / total)


def _oos_plain_reason(reason_code: str) -> str:
    return _OOS_PLAIN_REASONS.get(
        reason_code,
        f"Attribution could not be computed. Internal reason: {reason_code}.",
    )


def _oos_result(
    ticker: str,
    reason_code: str,
    reason_override: str | None = None,
) -> AttributionResult:
    """Return a fully-populated OOS result with a plain-English reason.

    Pass reason_override to supply a dynamic message (e.g. one that includes
    specific counts or series names). Falls back to _OOS_PLAIN_REASONS otherwise.
    """
    return AttributionResult(
        ticker=ticker,
        market_ratio=0.0,
        industry_ratio=0.0,
        firm_ratio=0.0,
        alpha_total=0.0,
        market_total=0.0,
        industry_total=0.0,
        firm_total=0.0,
        episode_return_total=0.0,
        adj_r2=None,
        recon_error_bps=None,
        quality_status="OOS",
        reason=reason_override if reason_override is not None else _oos_plain_reason(reason_code),
        reason_code=reason_code,
        beta_mkt_t_stat=None,
        beta_ind_t_stat=None,
        f_stat=None,
        f_pvalue=None,
        fit_quality_note=None,
        market_share=None,
        industry_share=None,
        firm_share=None,
        alpha_share=None,
        near_zero_episode_return=False,
        industry_proxy_ticker=None,
        industry_proxy_source=None,
        vix_estimation_p50=None,
        vix_estimation_p95=None,
        vix_episode_max=None,
        vix_episode_avg=None,
        macro_shock_suspected=False,
    )


def _download_single_ticker(
    ticker: str,
    fetch_start,
    fetch_end,
) -> pd.Series | None:
    """Download adjusted close for a single ticker, return a named Series or None.

    Yahoo Finance rate-limits multi-ticker batch downloads more aggressively than
    single-ticker requests (confirmed regression circa 2024). Downloading tickers
    one at a time avoids the batch endpoint and stays within single-ticker limits.
    """
    import time
    raw = yf.download(
        ticker,
        start=fetch_start,
        end=fetch_end,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return None
    col = "Adj Close" if "Adj Close" in raw.columns else ("Close" if "Close" in raw.columns else None)
    if col is None:
        return None
    series = raw[col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series.name = ticker
    time.sleep(1)
    return series


def _fetch_px_subset(
    cols: list[str],
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    calendar_days: int,
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Fetch price data and return an aligned subset for the given columns.

    Downloads each ticker individually to avoid Yahoo Finance batch-endpoint
    rate limits. Returns (px_subset, reason_code, reason_override).
    """
    fetch_start = (episode_start - pd.Timedelta(days=calendar_days)).date()
    fetch_end = (episode_end + pd.Timedelta(days=1)).date()

    series_list: list[pd.Series] = []
    failed: list[str] = []
    for ticker in cols:
        s = _download_single_ticker(ticker, fetch_start, fetch_end)
        if s is None or s.empty:
            failed.append(ticker)
        else:
            series_list.append(s)

    if not series_list:
        return None, "price_data_empty", None

    px = pd.concat(series_list, axis=1)

    if px.empty:
        return None, "price_data_empty", None

    missing_cols = [c for c in cols if c not in px.columns]
    if missing_cols:
        detail = (
            f"The following series could not be retrieved from the market data "
            f"provider: {', '.join(missing_cols)}. Attribution requires the stock, "
            "the S&P 500 ETF (SPY), and the sector ETF to all be available."
        )
        return None, "missing_required_series", detail

    return px[cols].copy(), None, None


def _fetch_vix_series(
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    calendar_days: int,
) -> pd.Series | None:
    """Fail-soft VIX fetch for the macro-shock flag (DEC-199 H).

    Returns a dated pd.Series of VIX closing levels, or None if the fetch fails
    for any reason. A None result must not block attribution — the flag simply
    doesn't fire.
    """
    fetch_start = (episode_start - pd.Timedelta(days=calendar_days)).date()
    fetch_end = (episode_end + pd.Timedelta(days=1)).date()
    try:
        series = _download_single_ticker(_VIX_TICKER, fetch_start, fetch_end)
    except Exception:
        return None
    if series is None or series.empty:
        return None
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.dropna()


def _compute_vix_metrics(
    vix_series: pd.Series | None,
    train_start: pd.Timestamp | None,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
) -> tuple[float | None, float | None, float | None, float | None, bool]:
    """Compute (p50_train, p95_train, ep_max, ep_avg, macro_shock_flag).

    Flag fires when EITHER `ep_max > p95_train` OR `ep_avg / p50_train > 1.5`.
    If the series is missing or either window is empty, all metrics are None
    and the flag is False (graceful no-op).
    """
    if vix_series is None or vix_series.empty:
        return None, None, None, None, False

    train_mask = vix_series.index < episode_start
    if train_start is not None:
        train_mask &= vix_series.index >= train_start
    train_values = vix_series[train_mask].dropna()

    ep_mask = (vix_series.index >= episode_start) & (vix_series.index <= episode_end)
    ep_values = vix_series[ep_mask].dropna()

    if train_values.empty or ep_values.empty:
        return None, None, None, None, False

    p50 = float(train_values.quantile(0.50))
    p95 = float(train_values.quantile(0.95))
    ep_max = float(ep_values.max())
    ep_avg = float(ep_values.mean())

    shock = (ep_max > p95) or (p50 > 0 and (ep_avg / p50) > _VIX_SHOCK_AVG_RATIO_THRESHOLD)
    return p50, p95, ep_max, ep_avg, bool(shock)


def _company_industry_proxy(episode: Episode) -> tuple[str, str]:
    """Resolve industry proxy ETF for a company (DEC-199).

    Returns (proxy_ticker, proxy_source) where proxy_source is either
    "sub_industry" (hit the sub-industry override table) or "sector"
    (fell back to the sector map). The source is surfaced in the UI so
    analysts can tell which mapping was used.

    Resolution order:
    1. SUB_INDUSTRY_TO_PROXY override keyed by company.gics_sub_industry
    2. SECTOR_TO_SPDR_PROXY fallback keyed by company.gics_sector
    """
    company = getattr(episode, "company", None)
    if company is None:
        raise ValueError("company_missing")

    sector = getattr(company, "gics_sector", None)
    if not sector:
        raise ValueError("missing_gics_sector")

    sub_industry = getattr(company, "gics_sub_industry", None)
    if sub_industry:
        override = SUB_INDUSTRY_TO_PROXY.get(sub_industry)
        if override:
            return override, "sub_industry"

    proxy = SECTOR_TO_SPDR_PROXY.get(sector)
    if not proxy:
        raise ValueError("unsupported_gics_sector")

    return proxy, "sector"


def _to_daily_returns(
    px: pd.DataFrame,
    ticker_col: str,
    mkt_col: str,
    ind_col: str,
) -> pd.DataFrame:
    if px.empty:
        return pd.DataFrame()

    rets = px.pct_change().rename_axis("date").reset_index()
    rets = rets.rename(
        columns={
            ticker_col: "stock_ret",
            mkt_col: "mkt_ret",
            ind_col: "ind_ret",
        }
    )
    rets["date"] = pd.to_datetime(rets["date"]).dt.tz_localize(None)
    rets = rets.dropna(subset=["stock_ret", "mkt_ret", "ind_ret"]).copy()
    return rets


def _ols_fit(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _ewma_weights(n: int, lambda_decay: float = EWMA_DECAY_FACTOR) -> np.ndarray:
    """Exponentially decaying weights for weighted-OLS attribution (DEC-199).

    Returns an array of length n where w[-1] = 1.0 (most recent observation)
    and w[t] = lambda^(n-1-t). Older observations contribute with exponentially
    diminishing weight. Halflife ≈ ln(0.5)/ln(lambda_decay).
    """
    return np.power(lambda_decay, np.arange(n - 1, -1, -1, dtype=float))


def _weighted_ols_fit(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted least squares via left-multiplication by sqrt(w).

    Minimizes Σ w_t * (y_t - x_t β)² by fitting OLS on (sqrt(w) * y, sqrt(w) * X).
    """
    sqrt_w = np.sqrt(weights)
    yw = y * sqrt_w
    xw = x * sqrt_w[:, None]
    beta, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    return beta


def _compute_ols_diagnostics(
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    n_predictors: int,
    weights: np.ndarray | None = None,
) -> tuple[float | None, np.ndarray | None, float | None, float | None]:
    """
    Compute adj_R², beta t-statistic vector, F-statistic, and F p-value for an
    (optionally weighted) OLS fit with `n_predictors` regressors (excluding
    intercept).

    When `weights` is provided, sums-of-squares use the weights and standard
    errors are derived from the weighted design matrix (DEC-199).

    Returns (adj_r2, t_stat_vec, f_stat, f_pvalue).
    t_stat_vec has the same length as beta (index 0 = intercept).
    """
    n = len(y)
    k = x.shape[1]  # total parameters including intercept

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)

    y_hat = x @ beta
    resid = y - y_hat

    sse = float(np.sum(w * resid ** 2))
    w_sum = float(np.sum(w))
    y_mean_w = float(np.sum(w * y) / w_sum) if w_sum > 0 else 0.0
    sst = float(np.sum(w * (y - y_mean_w) ** 2))

    rsq = 1.0 - (sse / sst) if sst > 0 else 0.0
    adj_r2: float | None = (
        float(1.0 - (1.0 - rsq) * (n - 1) / (n - k))
        if n > k
        else None
    )

    # Beta standard errors via SVD of the weighted design matrix.
    # For WLS, (X'WX)^{-1} = (Xw' Xw)^{-1} where Xw = sqrt(w) * X.
    t_stat_vec: np.ndarray | None = None
    sigma2 = sse / (n - k) if n > k else None
    if sigma2 is not None and sigma2 > 0:
        try:
            sqrt_w = np.sqrt(w)
            xw = x * sqrt_w[:, None]
            _, sv, Vt = np.linalg.svd(xw, full_matrices=False)
            sv_safe = np.where(sv > 1e-12, sv, np.inf)
            diag_XtX_inv = np.sum((Vt / sv_safe) ** 2, axis=0)
            se_betas = np.sqrt(np.maximum(sigma2 * diag_XtX_inv, 0.0))
            t_stat_vec = np.where(se_betas > 0, beta / se_betas, 0.0)
        except np.linalg.LinAlgError:
            t_stat_vec = None

    # F-statistic (joint significance of all predictors, excluding intercept)
    f_stat: float | None = None
    f_pvalue: float | None = None
    denom_df = n - n_predictors - 1
    if adj_r2 is not None and rsq < 1.0 and denom_df > 0:
        denom = (1.0 - rsq) / denom_df
        if denom > 0:
            f_stat = float((rsq / n_predictors) / denom)
            f_pvalue = float(
                1.0 - scipy_stats.f.cdf(f_stat, n_predictors, denom_df)
            )

    return adj_r2, t_stat_vec, f_stat, f_pvalue


def _determine_quality_status(
    adj_r2: float | None,
    beta_mkt_t_stat: float | None,
    beta_ind_t_stat: float | None,
) -> tuple[str, str | None]:
    """
    Apply the multi-metric quality gate (DEC-171).

    Returns (quality_status, fit_quality_note).
    fit_quality_note is None for PASS; a plain-English warning for POOR_FIT / CAUTION.
    """
    if adj_r2 is None:
        return "POOR_FIT", (
            "Regression quality metrics could not be computed. "
            "The attribution ratios should be treated as unreliable."
        )

    mkt_sig = (
        beta_mkt_t_stat is not None
        and abs(beta_mkt_t_stat) >= _SIGNIFICANCE_T_THRESHOLD
    )
    ind_sig = (
        beta_ind_t_stat is not None
        and abs(beta_ind_t_stat) >= _SIGNIFICANCE_T_THRESHOLD
    )
    any_sig = mkt_sig or ind_sig

    if adj_r2 < _POOR_FIT_ADJ_R2_THRESHOLD or not any_sig:
        if adj_r2 < _POOR_FIT_ADJ_R2_THRESHOLD and any_sig:
            # adj_r2 too low to rely on, but at least one beta is significant.
            # Do not claim "neither factor significant" — that would be factually wrong.
            sig_factor = "market" if mkt_sig else "industry"
            note = (
                f"The regression model has very low explanatory power "
                f"(adj.\u00a0R\u00b2\u00a0=\u00a0{adj_r2:.3f}). "
                f"Although the {sig_factor} factor is statistically significant, "
                "the overall model fit is too weak for the attribution ratios to "
                "reliably reflect the actual drivers of this episode."
            )
        else:
            # Neither beta is significant (regardless of adj_r2).
            note = (
                f"The regression model has low explanatory power "
                f"(adj.\u00a0R\u00b2\u00a0=\u00a0{adj_r2:.3f}). "
                "Neither the market nor the industry factor reached statistical "
                "significance. The market, industry, and firm attribution ratios "
                "may not reliably reflect the actual drivers of this episode."
            )
        return "POOR_FIT", note

    if adj_r2 < _CAUTION_ADJ_R2_THRESHOLD:
        sig_label = (
            "the market factor is statistically significant"
            if mkt_sig
            else "the industry factor is statistically significant"
        )
        note = (
            f"The regression model has limited explanatory power "
            f"(adj.\u00a0R\u00b2\u00a0=\u00a0{adj_r2:.3f}). "
            f"Although {sig_label}, the attribution ratios should be "
            "interpreted with caution."
        )
        return "CAUTION", note

    return "PASS", None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def episode_is_attributable(episode: Episode) -> bool:
    """Return True if the episode has the metadata required for attribution.

    Use as a route-level pre-flight check to avoid serving an analysis page
    for an episode that will unconditionally return OOS due to missing company
    metadata (company_missing / missing_gics_sector / unsupported_gics_sector).
    """
    company = getattr(episode, "company", None)
    if company is None:
        return False
    if not getattr(company, "gics_sector", None):
        return False
    return company.gics_sector in SECTOR_TO_SPDR_PROXY


def run_episode_window_attribution(
    episode: Episode,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
) -> AttributionResult:
    # --- Step 1: resolve industry proxy ETF (also validates company metadata) ---
    # Must run before accessing episode.company.ticker to handle company_missing safely.
    try:
        industry_proxy, industry_proxy_source = _company_industry_proxy(episode)
    except ValueError as exc:
        company = getattr(episode, "company", None)
        ticker = getattr(company, "ticker", f"episode-{episode.id}")
        return _oos_result(ticker, str(exc))

    ticker = episode.company.ticker

    market_proxy = DEFAULT_MARKET_PROXY
    episode_start = pd.Timestamp(episode.window_start)
    episode_end = pd.Timestamp(episode.window_end)

    # Fetch enough calendar days to cover the estimation window.
    # Approximate conversion: 1 trading day ≈ 1.5 calendar days (weekends + holidays).
    calendar_days = int(estimation_window * 1.5) + 30

    # --- Step 2: fetch price data ---
    cols = [ticker, industry_proxy, market_proxy]
    px_subset, rc, rd = _fetch_px_subset(cols, episode_start, episode_end, calendar_days)
    if rc is not None:
        return _oos_result(ticker, rc, rd)

    rets = _to_daily_returns(px_subset, ticker, market_proxy, industry_proxy)

    # --- Step 2b: check alignment shortfall; retry once with extended window ---
    n_pre_episode = len(rets[rets["date"] < episode_start])
    shortfall = max(0, estimation_window - n_pre_episode)

    if shortfall > _ALIGNMENT_SHORTFALL_THRESHOLD:
        return _oos_result(
            ticker,
            "missing_required_series",
            f"The three price series could not be aligned to {estimation_window} "
            f"common trading days. {shortfall} observations are missing, which "
            "exceeds the recoverable threshold. This is likely a data availability "
            "issue with the market data provider.",
        )

    if shortfall > 0:
        extra_days = int(shortfall * 2.0) + 30
        px_subset2, rc2, rd2 = _fetch_px_subset(
            cols, episode_start, episode_end, calendar_days + extra_days
        )
        if rc2 is not None:
            return _oos_result(ticker, rc2, rd2)
        rets2 = _to_daily_returns(px_subset2, ticker, market_proxy, industry_proxy)
        n_pre_episode2 = len(rets2[rets2["date"] < episode_start])
        shortfall2 = max(0, estimation_window - n_pre_episode2)
        if shortfall2 > _ALIGNMENT_SHORTFALL_THRESHOLD:
            return _oos_result(
                ticker,
                "missing_required_series",
                f"The three price series could not be aligned to {estimation_window} "
                f"common trading days after extending the fetch window. "
                f"{shortfall2} observations remain missing. "
                "This is a data availability issue with the market data provider.",
            )
        rets = rets2

    # --- Step 3: split training window and episode window ---
    train = rets[rets["date"] < episode_start].tail(estimation_window).copy()
    win = rets[
        (rets["date"] >= episode_start) & (rets["date"] <= episode_end)
    ].copy()

    if win.empty:
        return _oos_result(ticker, "episode_window_empty")

    if len(train) < _MIN_TRAIN_OBSERVATIONS:
        return _oos_result(ticker, "insufficient_train_data")

    # --- Step 3b: VIX macro-shock diagnostics (DEC-199 H, fail-soft) ---
    # Runs outside the critical path — a failed VIX fetch must not block the
    # main regression, it just leaves the flag False.
    vix_series = _fetch_vix_series(episode_start, episode_end, calendar_days)
    train_start_ts = pd.Timestamp(train["date"].iloc[0]) if not train.empty else None
    (
        vix_estimation_p50,
        vix_estimation_p95,
        vix_episode_max,
        vix_episode_avg,
        macro_shock_suspected,
    ) = _compute_vix_metrics(
        vix_series,
        train_start_ts,
        episode_start,
        episode_end,
    )

    # --- Step 4: orthogonalize industry vs market (EWMA weighted, DEC-199) ---
    train_weights = _ewma_weights(len(train))

    y_ind = train["ind_ret"].to_numpy()
    x_ind = np.column_stack([np.ones(len(train)), train["mkt_ret"].to_numpy()])
    beta_ind_on_mkt = _weighted_ols_fit(y_ind, x_ind, train_weights)

    train["ind_hat"] = x_ind @ beta_ind_on_mkt
    train["ind_orth"] = train["ind_ret"] - train["ind_hat"]

    # --- Step 5: fit main attribution regression (EWMA weighted, DEC-199) ---
    y_stock = train["stock_ret"].to_numpy()
    x_main = np.column_stack([
        np.ones(len(train)),
        train["mkt_ret"].to_numpy(),
        train["ind_orth"].to_numpy(),
    ])
    beta_main = _weighted_ols_fit(y_stock, x_main, train_weights)

    # --- Step 6: compute regression diagnostics (weighted, DEC-199) ---
    adj_r2, t_stat_vec, f_stat, f_pvalue = _compute_ols_diagnostics(
        y=y_stock,
        x=x_main,
        beta=beta_main,
        n_predictors=2,  # market + industry, excluding intercept
        weights=train_weights,
    )
    beta_mkt_t_stat = float(t_stat_vec[1]) if t_stat_vec is not None else None
    beta_ind_t_stat = float(t_stat_vec[2]) if t_stat_vec is not None else None

    # --- Step 7: determine quality status (DEC-171) ---
    quality_status, fit_quality_note = _determine_quality_status(
        adj_r2=adj_r2,
        beta_mkt_t_stat=beta_mkt_t_stat,
        beta_ind_t_stat=beta_ind_t_stat,
    )

    # --- Step 8: apply betas to episode window ---
    xw_ind = np.column_stack([np.ones(len(win)), win["mkt_ret"].to_numpy()])
    win["ind_hat"] = xw_ind @ beta_ind_on_mkt
    win["ind_orth"] = win["ind_ret"] - win["ind_hat"]

    alpha = float(beta_main[0])
    beta_mkt = float(beta_main[1])
    beta_ind_val = float(beta_main[2])

    win["alpha_contrib"] = alpha
    win["market_contrib"] = beta_mkt * win["mkt_ret"]
    win["industry_contrib"] = beta_ind_val * win["ind_orth"]
    win["y_hat"] = (
        win["alpha_contrib"] + win["market_contrib"] + win["industry_contrib"]
    )
    win["firm_contrib"] = win["stock_ret"] - win["y_hat"]

    alpha_total = float(win["alpha_contrib"].sum())
    market_total = float(win["market_contrib"].sum())
    industry_total = float(win["industry_contrib"].sum())
    firm_total = float(win["firm_contrib"].sum())
    episode_return_total = float(win["stock_ret"].sum())

    recon_error_bps = float(
        abs(
            episode_return_total
            - (alpha_total + market_total + industry_total + firm_total)
        )
        * 10000.0
    )

    abs_total = (
        abs(alpha_total) + abs(market_total) + abs(industry_total) + abs(firm_total)
    )
    market_ratio = _safe_ratio(market_total, abs_total)
    industry_ratio = _safe_ratio(industry_total, abs_total)
    firm_ratio = _safe_ratio(firm_total, abs_total)

    # --- Step 9: net-contribution shares (DEC-199 G) ---
    near_zero = abs(episode_return_total) < _NEAR_ZERO_EPISODE_RETURN_THRESHOLD
    if near_zero:
        market_share = industry_share = firm_share = alpha_share = None
    else:
        market_share = market_total / episode_return_total
        industry_share = industry_total / episode_return_total
        firm_share = firm_total / episode_return_total
        alpha_share = alpha_total / episode_return_total

    return AttributionResult(
        ticker=ticker,
        market_ratio=market_ratio,
        industry_ratio=industry_ratio,
        firm_ratio=firm_ratio,
        alpha_total=alpha_total,
        market_total=market_total,
        industry_total=industry_total,
        firm_total=firm_total,
        episode_return_total=episode_return_total,
        adj_r2=adj_r2,
        recon_error_bps=recon_error_bps,
        quality_status=quality_status,
        reason=None,
        reason_code=quality_status.lower(),
        beta_mkt_t_stat=beta_mkt_t_stat,
        beta_ind_t_stat=beta_ind_t_stat,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        fit_quality_note=fit_quality_note,
        market_share=market_share,
        industry_share=industry_share,
        firm_share=firm_share,
        alpha_share=alpha_share,
        near_zero_episode_return=near_zero,
        industry_proxy_ticker=industry_proxy,
        industry_proxy_source=industry_proxy_source,
        vix_estimation_p50=vix_estimation_p50,
        vix_estimation_p95=vix_estimation_p95,
        vix_episode_max=vix_episode_max,
        vix_episode_avg=vix_episode_avg,
        macro_shock_suspected=macro_shock_suspected,
    )
