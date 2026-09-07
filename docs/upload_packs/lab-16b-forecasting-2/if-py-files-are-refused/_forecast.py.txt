"""Forecasting helpers for MBA 775, Chapter 16.

Every function here implements a formula box from Donnelly's Chapter 16, with
the textbook's conventions kept on purpose:

  * A forecast for period t uses only data through period t-1. The first
    three-period moving average therefore belongs to period 4, not period 3.
  * Exponential smoothing is "primed" with F1 = A1, and period 1 is excluded
    from the MAD so that the method gets no credit for a forecast it did not
    make.
  * MAD averages the absolute errors over the periods that actually have a
    forecast. RMSE and MAPE are supplied as well, because MAD is not the only
    defensible yardstick and the forecasting challenge scores on more than one.

Each forecasting function returns a Series of *forecasts* aligned to the
periods they are forecasts FOR, with one extra entry at the end: the forecast
for the next period after the data end. That last entry is the number you
would submit.

You do not need to read this file to do the coursework. It is here so that
the definitions in the note and the numbers in the output cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "naive_forecast", "sma_forecast", "wma_forecast",
    "exp_smooth", "exp_smooth_trend",
    "mad", "rmse", "mape", "error_table", "accuracy",
    "tune_sma", "tune_alpha", "tune_alpha_beta",
    "trend_projection", "durbin_watson", "dw_critical", "dw_test",
    "centered_moving_average", "seasonal_factors", "decompose",
    "seasonal_dummy_regression",
    "backtest", "forecast_plot", "fill_gaps",
    "CAR_SALES",
]

# Donnelly Table 16.1 -- Brandywine Ford new car sales, 2016Q1 to 2018Q4.
CAR_SALES = pd.Series(
    [97, 142, 108, 135, 120, 164, 126, 150, 123, 151, 141, 142],
    index=pd.period_range("2016Q1", periods=12, freq="Q"),
    name="sales",
)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def _as_series(y) -> pd.Series:
    """Coerce input to a float Series with a clean, sequential index."""
    s = pd.Series(y, dtype="float64")
    if s.isna().any():
        n = int(s.isna().sum())
        raise ValueError(
            f"{n} missing value(s) in the series. Every method in this module "
            f"needs consecutive periods. Decide how to treat the gap first -- "
            f"see fill_gaps() -- and say what you did."
        )
    return s


def _next_label(index):
    """Label for the period after the last one in `index`."""
    if isinstance(index, pd.PeriodIndex):
        return index[-1] + 1
    if isinstance(index, pd.DatetimeIndex) and index.freq is not None:
        return index[-1] + index.freq
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        step = index[-1] - index[-2]
        return index[-1] + step
    if isinstance(index, pd.RangeIndex) or np.issubdtype(np.asarray(index).dtype, np.integer):
        return int(index[-1]) + 1
    return len(index) + 1


def _with_next(values, index, name):
    """Build a forecast Series covering every historical period plus one more."""
    out = pd.Series(values, index=list(index) + [_next_label(index)], name=name,
                    dtype="float64")
    return out


def fill_gaps(y, method="interpolate") -> pd.Series:
    """Fill missing observations, and SAY how many were filled.

    A time series with a hole in it cannot be smoothed. The choice of how to
    fill the hole is a modelling decision, not a data-cleaning chore: it should
    be stated in your write-up. `interpolate` draws a straight line between the
    neighbours; `previous` carries the last value forward.
    """
    s = pd.Series(y, dtype="float64")
    n_missing = int(s.isna().sum())
    if n_missing == 0:
        return s
    if method == "interpolate":
        filled = s.interpolate(limit_area="inside")
    elif method == "previous":
        filled = s.ffill()
    else:
        raise ValueError("method must be 'interpolate' or 'previous'")
    where = ", ".join(str(i) for i in s.index[s.isna()])
    print(f"NOTE: filled {n_missing} missing value(s) by '{method}' at: {where}")
    return filled


# ---------------------------------------------------------------------------
# Smoothing forecasts (Section 16.2, plus the naive forecast the book skips)
# ---------------------------------------------------------------------------

def naive_forecast(y) -> pd.Series:
    """Next period's forecast is this period's actual: F_t = A_{t-1}.

    Not in the textbook, but it is the benchmark every other method has to
    beat. A method that cannot outperform "same as last time" has not earned
    its complexity.
    """
    s = _as_series(y)
    vals = [np.nan] + list(s.values)
    return _with_next(vals, s.index, "naive")


def sma_forecast(y, p: int) -> pd.Series:
    """Simple moving average of the most recent p values (Section 16.2).

    The first forecast is for period p+1. Increasing p smooths more and
    responds less; whether that lowers the MAD depends on the data.
    """
    s = _as_series(y)
    if p < 1 or p > len(s):
        raise ValueError(f"p must be between 1 and {len(s)}")
    vals = [np.nan] * p
    for t in range(p, len(s) + 1):
        vals.append(s.values[t - p:t].mean())
    return _with_next(vals, s.index, f"sma_{p}")


def wma_forecast(y, weights) -> pd.Series:
    """Weighted moving average. `weights` run OLDEST to NEWEST, as in the
    textbook's "3, 2, 1" (which the book lists newest first -- here the last
    weight applies to the most recent period). The denominator is the sum of
    the weights, so they need not add to one.
    """
    s = _as_series(y)
    w = np.asarray(weights, dtype="float64")
    p = len(w)
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive number")
    vals = [np.nan] * p
    for t in range(p, len(s) + 1):
        vals.append(float(np.dot(w, s.values[t - p:t]) / w.sum()))
    label = "wma_" + "_".join(f"{x:g}" for x in w)
    return _with_next(vals, s.index, label)


def exp_smooth(y, alpha: float, start=None) -> pd.Series:
    """Exponential smoothing: F_t = F_{t-1} + alpha * (A_{t-1} - F_{t-1}).

    Formula 16.2. The forecast is primed with F_1 = A_1 (or `start`), so the
    first forecast that was actually *calculated* is for period 2. Period 1
    carries the primed value and is excluded from error calculations by
    error_table() / mad().
    """
    s = _as_series(y)
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    a = s.values
    F = np.empty(len(a) + 1)
    F[0] = a[0] if start is None else float(start)
    for t in range(1, len(a) + 1):
        F[t] = F[t - 1] + alpha * (a[t - 1] - F[t - 1])
    out = _with_next(F, s.index, f"es_{alpha:g}")
    out.attrs["primed"] = 1     # first period is not a real forecast
    return out


def exp_smooth_trend(y, alpha: float, beta: float, start=None) -> pd.DataFrame:
    """Exponential smoothing with trend adjustment (Formulas 16.3-16.5).

        F_t   = FIT_{t-1} + alpha * (A_{t-1} - FIT_{t-1})
        T_t   = beta * (F_t - F_{t-1}) + (1 - beta) * T_{t-1}
        FIT_t = F_t + T_t

    Returns a DataFrame with columns F, T, FIT for every period plus the next.
    Use the FIT column as the forecast. As in the book, F_1 = A_1 and T_1 = 0.
    """
    s = _as_series(y)
    for name, v in (("alpha", alpha), ("beta", beta)):
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    a = s.values
    n = len(a)
    F = np.empty(n + 1); T = np.empty(n + 1); FIT = np.empty(n + 1)
    F[0] = a[0] if start is None else float(start)
    T[0] = 0.0
    FIT[0] = F[0] + T[0]
    for t in range(1, n + 1):
        F[t] = FIT[t - 1] + alpha * (a[t - 1] - FIT[t - 1])
        T[t] = beta * (F[t] - F[t - 1]) + (1 - beta) * T[t - 1]
        FIT[t] = F[t] + T[t]
    idx = list(s.index) + [_next_label(s.index)]
    out = pd.DataFrame({"F": F, "T": T, "FIT": FIT}, index=idx)
    out.attrs["primed"] = 1
    out.attrs["label"] = f"es_trend_{alpha:g}_{beta:g}"
    return out


# ---------------------------------------------------------------------------
# Accuracy (Formula 16.1 and two companions)
# ---------------------------------------------------------------------------

def _forecast_column(forecast):
    """Accept a Series, or the DataFrame from exp_smooth_trend()."""
    if isinstance(forecast, pd.DataFrame):
        f = forecast["FIT"].copy()
        f.name = forecast.attrs.get("label", "FIT")
        f.attrs["primed"] = forecast.attrs.get("primed", 0)
        return f
    return forecast


def error_table(actual, forecast) -> pd.DataFrame:
    """Actual, forecast, error and absolute error, period by period.

    Rows without a forecast (the warm-up periods of a moving average, or the
    primed first period of exponential smoothing) are shown but excluded from
    the MAD, exactly as in Tables 16.2-16.6.
    """
    a = _as_series(actual)
    f = _forecast_column(forecast)
    hist = f.iloc[: len(a)].copy()
    hist.index = a.index
    primed = int(f.attrs.get("primed", 0)) if hasattr(f, "attrs") else 0
    if primed:
        hist.iloc[:primed] = np.nan
    err = a - hist
    return pd.DataFrame({
        "actual": a, "forecast": hist,
        "error": err, "abs_error": err.abs(),
        "pct_error": (err.abs() / a.abs() * 100).where(a != 0),
    })


def mad(actual, forecast) -> float:
    """Mean absolute deviation (Formula 16.1)."""
    t = error_table(actual, forecast)
    return float(t["abs_error"].mean())


def rmse(actual, forecast) -> float:
    """Root mean squared error. Punishes large misses more than MAD does."""
    t = error_table(actual, forecast)
    return float(np.sqrt((t["error"] ** 2).mean()))


def mape(actual, forecast) -> float:
    """Mean absolute percentage error. Unit-free, so it can compare a
    forecast of the CPI (hundreds) with one of jobless claims (hundreds of
    thousands). Undefined when the actual is zero."""
    t = error_table(actual, forecast)
    return float(t["pct_error"].mean())


def accuracy(actual, forecasts: dict) -> pd.DataFrame:
    """Summary table in the spirit of Tables 16.7 and 16.22: one row per
    method with its next-period forecast, MAD, RMSE, MAPE and the number of
    errors averaged."""
    rows = []
    a = _as_series(actual)
    for name, f in forecasts.items():
        col = _forecast_column(f)
        t = error_table(a, col)
        rows.append({
            "method": name,
            "next_forecast": float(col.iloc[-1]),
            "MAD": t["abs_error"].mean(),
            "RMSE": np.sqrt((t["error"] ** 2).mean()),
            "MAPE": t["pct_error"].mean(),
            "n_errors": int(t["abs_error"].notna().sum()),
        })
    return pd.DataFrame(rows).set_index("method")


# ---------------------------------------------------------------------------
# Tuning: let the data pick the parameter, then say so
# ---------------------------------------------------------------------------

def tune_sma(y, max_p=None) -> pd.DataFrame:
    """MAD for every window length from 1 (which is the naive forecast) up."""
    s = _as_series(y)
    max_p = max_p or max(2, len(s) // 3)
    rows = [{"p": p, "MAD": mad(s, sma_forecast(s, p)),
             "RMSE": rmse(s, sma_forecast(s, p))} for p in range(1, max_p + 1)]
    return pd.DataFrame(rows).set_index("p")


def tune_alpha(y, grid=None) -> pd.DataFrame:
    """MAD across a grid of alpha values -- the U-shaped curve of Figure 16.5."""
    s = _as_series(y)
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2) if grid is None else grid
    rows = [{"alpha": a, "MAD": mad(s, exp_smooth(s, a)),
             "RMSE": rmse(s, exp_smooth(s, a))} for a in grid]
    return pd.DataFrame(rows).set_index("alpha")


def tune_alpha_beta(y, alphas=None, betas=None) -> pd.DataFrame:
    """MAD over a grid of (alpha, beta) for trend-adjusted smoothing."""
    s = _as_series(y)
    alphas = np.round(np.arange(0.1, 1.0, 0.1), 2) if alphas is None else alphas
    betas = np.round(np.arange(0.1, 1.0, 0.1), 2) if betas is None else betas
    rows = []
    for a in alphas:
        for b in betas:
            rows.append({"alpha": a, "beta": b,
                         "MAD": mad(s, exp_smooth_trend(s, a, b))})
    return pd.DataFrame(rows).pivot(index="alpha", columns="beta", values="MAD")


# ---------------------------------------------------------------------------
# Regression (Section 16.3)
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray):
    """Plain least squares with the usual standard errors. numpy only."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    dof = n - k
    s2 = float(resid @ resid) / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    try:
        from scipy import stats
        pvals = 2 * stats.t.sf(np.abs(tstat), dof)
    except Exception:                           # scipy not installed
        pvals = np.full(k, np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else np.nan
    return dict(beta=beta, se=se, t=tstat, p=pvals, fitted=fitted, resid=resid,
                r2=r2, adj_r2=adj_r2, std_error=np.sqrt(s2), n=n, k=k - 1)


def trend_projection(y, horizon: int = 1) -> dict:
    """Fit y_t = b0 + b1 * t by simple regression on the period number
    (Formulas 16.6-16.8) and project it `horizon` periods ahead.

    Returns a dict with the coefficients, a table of fitted values and
    residuals (Table 16.9), the forecasts, and the regression summary numbers
    Excel would print.
    """
    s = _as_series(y)
    n = len(s)
    t = np.arange(1, n + 1, dtype="float64")
    X = np.column_stack([np.ones(n), t])
    fit = _ols(X, s.values)
    b0, b1 = fit["beta"]
    future_t = np.arange(n + 1, n + horizon + 1)
    forecasts = b0 + b1 * future_t
    table = pd.DataFrame({"t": t.astype(int), "actual": s.values,
                          "fitted": fit["fitted"], "residual": fit["resid"],
                          "abs_error": np.abs(fit["resid"])}, index=s.index)
    # The forecast Series in the same shape as the smoothing functions return
    forecast_series = _with_next(list(fit["fitted"]) + [forecasts[0]], s.index,
                                 "trend")
    return dict(b0=float(b0), b1=float(b1), table=table,
                forecasts=pd.Series(forecasts, index=future_t, name="forecast"),
                forecast=forecast_series,
                MAD=float(np.abs(fit["resid"]).mean()),
                se=fit["se"], t=fit["t"], p=fit["p"], r2=fit["r2"],
                adj_r2=fit["adj_r2"], std_error=fit["std_error"])


def durbin_watson(residuals) -> float:
    """Formula 16.10: sum of squared successive differences over sum of
    squared residuals. 2 means no autocorrelation; 0 perfect positive."""
    e = np.asarray(residuals, dtype="float64")
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


# Lower and upper critical values, one-tail test for POSITIVE autocorrelation,
# alpha = 0.05, k = 1 independent variable. These match Donnelly's Appendix A
# Table 9 (n = 16 gives 1.10 and 1.37, the values used in the text). For k > 1
# or another alpha, look the values up in the appendix.
_DW_K1_05 = {
    15: (1.08, 1.36), 16: (1.10, 1.37), 17: (1.13, 1.38), 18: (1.16, 1.39),
    19: (1.18, 1.40), 20: (1.20, 1.41), 21: (1.22, 1.42), 22: (1.24, 1.43),
    23: (1.26, 1.44), 24: (1.27, 1.45), 25: (1.29, 1.45), 26: (1.30, 1.46),
    27: (1.32, 1.47), 28: (1.33, 1.48), 29: (1.34, 1.48), 30: (1.35, 1.49),
    31: (1.36, 1.50), 32: (1.37, 1.50), 33: (1.38, 1.51), 34: (1.39, 1.51),
    35: (1.40, 1.52), 36: (1.41, 1.52), 37: (1.42, 1.53), 38: (1.43, 1.54),
    39: (1.43, 1.54), 40: (1.44, 1.54), 45: (1.48, 1.57), 50: (1.50, 1.59),
    55: (1.53, 1.60), 60: (1.55, 1.62), 65: (1.57, 1.63), 70: (1.58, 1.64),
    75: (1.60, 1.65), 80: (1.61, 1.66), 85: (1.62, 1.67), 90: (1.63, 1.68),
    95: (1.64, 1.69), 100: (1.65, 1.69),
}


def dw_critical(n: int, k: int = 1, alpha: float = 0.05):
    """(d_L, d_U) from the table above. Only k = 1, alpha = 0.05 is built in;
    for larger n the nearest smaller tabled n is used (conservative)."""
    if k != 1 or alpha != 0.05:
        raise ValueError("Only k=1, alpha=0.05 is tabled here; use Appendix A.")
    if n < 15:
        raise ValueError("The Durbin-Watson test needs at least 15 observations.")
    key = max(m for m in _DW_K1_05 if m <= n)
    return _DW_K1_05[key]


def dw_test(residuals, k: int = 1, alpha: float = 0.05) -> dict:
    """One-tail test for positive autocorrelation with the textbook's rules:
    d < d_L reject H0; d_L <= d <= d_U inconclusive; d > d_U fail to reject."""
    e = np.asarray(residuals, dtype="float64")
    d = durbin_watson(e)
    dl, du = dw_critical(len(e), k, alpha)
    if d < dl:
        verdict = "reject H0: positive autocorrelation is present"
    elif d > du:
        verdict = "fail to reject H0: no evidence of positive autocorrelation"
    else:
        verdict = "inconclusive"
    return dict(d=d, d_L=dl, d_U=du, n=len(e), verdict=verdict)


# ---------------------------------------------------------------------------
# Seasonality (Section 16.4)
# ---------------------------------------------------------------------------

def centered_moving_average(y, season_length: int) -> pd.Series:
    """Step 1 of decomposition. For an even number of seasons the plain
    moving average lands between periods, so two neighbouring averages are
    averaged again to centre it (Tables 16.14-16.15)."""
    s = _as_series(y)
    L = season_length
    ma = s.rolling(L).mean()               # lands on the LAST period of the window
    if L % 2 == 0:
        cma = (ma + ma.shift(-1)) / 2      # average of windows ending at t and t+1
        cma = cma.shift(-(L // 2 - 1))     # centre on the middle period
    else:
        cma = ma.shift(-(L // 2))
    cma.name = "CMA"
    return cma


def seasonal_factors(y, season_length: int, season_of=None) -> pd.DataFrame:
    """Steps 1-4 of Tables 16.14-16.17: CMA, ratio-to-moving-average, the
    seasonal factor for each season (mean of its ratios) and the normalised
    factors that sum to the number of seasons."""
    s = _as_series(y)
    L = season_length
    cma = centered_moving_average(s, L)
    rma = s / cma
    if season_of is None:
        season = pd.Series((np.arange(len(s)) % L) + 1, index=s.index)
    else:
        season = pd.Series(season_of, index=s.index)
    table = pd.DataFrame({"actual": s, "CMA": cma, "RMA": rma, "season": season})
    sf = table.groupby("season")["RMA"].mean()
    nsf = sf * (L / sf.sum())
    factors = pd.DataFrame({"SF": sf, "NSF": nsf})
    factors.attrs["sum_SF"] = float(sf.sum())
    return table, factors


def decompose(y, season_length: int, horizon: int = 1, season_of=None) -> dict:
    """Multiplicative decomposition forecast (Formulas 16.11-16.14):
    deseasonalise with the normalised factors, fit a trend line to the
    deseasonalised series, and multiply the projected trend by the seasonal
    factor for each future period."""
    s = _as_series(y)
    L = season_length
    table, factors = seasonal_factors(s, L, season_of)
    S = table["season"].map(factors["NSF"])
    deseason = s / S
    trend = trend_projection(deseason, horizon=horizon)
    n = len(s)
    fitted = trend["table"]["fitted"] * S.values
    future_t = np.arange(n + 1, n + horizon + 1)
    future_season = ((np.arange(n, n + horizon) % L) + 1) if season_of is None \
        else [((int(table["season"].iloc[-1]) + i - 1) % L) + 1
              for i in range(1, horizon + 1)]
    future_S = factors["NSF"].reindex(future_season).values
    forecasts = trend["forecasts"].values * future_S
    full = pd.DataFrame({
        "actual": s, "season": table["season"], "S": S,
        "deseasonalized": deseason, "T": trend["table"]["fitted"],
        "forecast": fitted, "error": s - fitted, "abs_error": (s - fitted).abs(),
    })
    fut = pd.DataFrame({"t": future_t, "season": future_season,
                        "T": trend["forecasts"].values, "S": future_S,
                        "forecast": forecasts})
    forecast_series = _with_next(list(fitted) + [forecasts[0]], s.index, "decomposition")
    return dict(steps=table, factors=factors, b0=trend["b0"], b1=trend["b1"],
                table=full, future=fut, forecast=forecast_series,
                MAD=float(full["abs_error"].mean()), trend_fit=trend)


def seasonal_dummy_regression(y, season_length: int, base_season: int = 1,
                              horizon: int = 1, season_of=None,
                              drop=None) -> dict:
    """Regress y on the period number and season dummies (Table 16.23-16.24).

    `drop` lets you leave dummies out to mimic the textbook's best-subsets
    result, e.g. drop=[2] omits SD2. Returns coefficients, the regression
    summary, fitted values and residuals, and `horizon` forecasts.
    """
    s = _as_series(y)
    n = len(s)
    L = season_length
    season = np.array(season_of) if season_of is not None else (np.arange(n) % L) + 1
    t = np.arange(1, n + 1, dtype="float64")
    seasons = [q for q in range(1, L + 1) if q != base_season]
    if drop:
        seasons = [q for q in seasons if q not in drop]
    cols = ["intercept", "t"] + [f"SD{q}" for q in seasons]
    X = np.column_stack([np.ones(n), t] + [(season == q).astype(float) for q in seasons])
    fit = _ols(X, s.values)
    coef = pd.DataFrame({"coef": fit["beta"], "se": fit["se"], "t": fit["t"],
                         "p": fit["p"]}, index=cols)
    future_t = np.arange(n + 1, n + horizon + 1)
    future_season = ((np.arange(n, n + horizon) % L) + 1) if season_of is None \
        else [((int(season[-1]) + i - 1) % L) + 1 for i in range(1, horizon + 1)]
    Xf = np.column_stack([np.ones(horizon), future_t.astype(float)] +
                         [(np.array(future_season) == q).astype(float) for q in seasons])
    forecasts = Xf @ fit["beta"]
    table = pd.DataFrame({"t": t.astype(int), "season": season, "actual": s.values,
                          "fitted": fit["fitted"], "residual": fit["resid"],
                          "abs_error": np.abs(fit["resid"])}, index=s.index)
    forecast_series = _with_next(list(fit["fitted"]) + [forecasts[0]], s.index,
                                 "seasonal_dummies")
    return dict(coef=coef, r2=fit["r2"], adj_r2=fit["adj_r2"],
                std_error=fit["std_error"], table=table,
                future=pd.DataFrame({"t": future_t, "season": future_season,
                                     "forecast": forecasts}),
                forecast=forecast_series, MAD=float(np.abs(fit["resid"]).mean()))


# ---------------------------------------------------------------------------
# Honest evaluation: how would this method have done, period after period?
# ---------------------------------------------------------------------------

def backtest(y, method, last: int = 12, **kwargs) -> pd.DataFrame:
    """Rolling-origin evaluation.

    For each of the `last` periods, refit `method` on the data available
    BEFORE that period and record the one-step forecast, then compare with
    what actually happened. This is what the forecasting challenge does to you
    every week, so it is the fairest way to choose a method.

    `method` is a function like sma_forecast or exp_smooth (kwargs are passed
    through); it must return a Series whose last entry is the next-period
    forecast, or a DataFrame with a FIT column.
    """
    s = _as_series(y)
    rows = []
    for i in range(len(s) - last, len(s)):
        history = s.iloc[:i]
        f = method(history, **kwargs)
        f = f["FIT"] if isinstance(f, pd.DataFrame) else f
        rows.append({"period": s.index[i], "actual": s.iloc[i],
                     "forecast": float(f.iloc[-1])})
    out = pd.DataFrame(rows).set_index("period")
    out["error"] = out["actual"] - out["forecast"]
    out["abs_error"] = out["error"].abs()
    out["pct_error"] = out["abs_error"] / out["actual"].abs() * 100
    return out


def backtest_summary(y, methods: dict, last: int = 12) -> pd.DataFrame:
    """MAD / RMSE / MAPE of several methods over the same held-out periods."""
    rows = []
    for name, (fn, kw) in methods.items():
        b = backtest(y, fn, last=last, **kw)
        rows.append({"method": name, "MAD": b["abs_error"].mean(),
                     "RMSE": np.sqrt((b["error"] ** 2).mean()),
                     "MAPE": b["pct_error"].mean(), "periods": len(b)})
    return pd.DataFrame(rows).set_index("method").sort_values("MAD")


__all__.append("backtest_summary")


# ---------------------------------------------------------------------------
# One chart
# ---------------------------------------------------------------------------

def forecast_plot(actual, forecasts: dict, title=None, ylab=None, last=None,
                  figsize=(10, 4.8)):
    """Actual series in scarlet, each forecast in gray tones, the next-period
    forecast marked. `last` restricts the window to the final N periods."""
    import matplotlib.pyplot as plt
    try:
        from _charts import UNLV_SCARLET, UNLV_GRAY
    except Exception:
        UNLV_SCARLET, UNLV_GRAY = "#a03123", "#666666"

    a = pd.Series(actual, dtype="float64")
    if last:
        a = a.iloc[-last:]
    x_a = _plot_x(a.index)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_a, a.values, color=UNLV_SCARLET, linewidth=1.8, label="actual")
    grays = ["#333333", UNLV_GRAY, "#9FA1A4", "#6A737B", "#b5b8bb"]
    for i, (name, f) in enumerate(forecasts.items()):
        f = _forecast_column(f)
        if last:
            f = f.iloc[-(last + 1):]
        x_f = _plot_x(f.index)
        col = grays[i % len(grays)]
        ax.plot(x_f[:-1], f.values[:-1], color=col, linewidth=1.2,
                linestyle="--", label=name)
        ax.scatter([x_f[-1]], [f.values[-1]], color=col, zorder=5, s=28)
    ax.set_title(title or "Actual vs. forecast", color=UNLV_SCARLET)
    if ylab:
        ax.set_ylabel(ylab)
    ax.grid(axis="y", color="#e6e6e6")
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig


def _plot_x(index):
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp()
    try:
        return pd.PeriodIndex(index).to_timestamp()
    except Exception:
        pass
    try:
        return pd.to_datetime(index)
    except Exception:
        return np.arange(len(index))
