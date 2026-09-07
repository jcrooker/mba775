"""MBA 775 - Chapter 16, Script B
The forecasting challenge, part 2: exponential smoothing, with and without trend.

    python 16b_exponential_smoothing.py

What it does, in order:

    1. Reproduces Table 16.5 (exponential smoothing, alpha = 0.6) and
       Table 16.6 (with trend adjustment, alpha = 0.3, beta = 0.1) for the
       textbook's car sales, so every number can be checked against the book.
    2. Draws the U-shaped curve of Figure 16.5: MAD against alpha.
    3. Tunes alpha for the CPI and for weekly claims, and shows that the
       tuned alpha is not the same for the two series.
    4. Adds the trend component and tunes alpha and beta together.
    5. Backtests everything from Scripts A and B side by side, so the
       method you submit has earned its place.

Data files it reads (from the course repository's data/ folder):

    cpi.csv        icsa.csv        nvsthpi.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                # noqa: E402
from _forecast import (CAR_SALES, naive_forecast, sma_forecast,     # noqa: E402
                       exp_smooth, exp_smooth_trend, error_table, mad,
                       accuracy, tune_alpha, tune_alpha_beta,
                       backtest_summary, fill_gaps, forecast_plot)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

# ---------------------------------------------------------------------------
banner("1. Textbook check: Tables 16.5 and 16.6")

y = CAR_SALES
es6 = exp_smooth(y, alpha=0.6)
print("Table 16.5  exponential smoothing, alpha = 0.6 (book: forecast 142, MAD 19.0)")
print(error_table(y, es6).round(1)[["actual", "forecast", "error", "abs_error"]])
print(f"Forecast for 2019Q1: {es6.iloc[-1]:.1f}    MAD: {mad(y, es6):.1f}")
print("Period 1 has no error: F1 = A1 was assigned, not forecast, so it is left out")
print("of the MAD -- otherwise the method gets credit for a forecast it never made.")

est = exp_smooth_trend(y, alpha=0.3, beta=0.1)
print("\nTable 16.6  with trend adjustment, alpha = 0.3, beta = 0.1 (book: 148, MAD 16.6)")
tbl = est.round(1)
tbl["actual"] = list(y.values) + [np.nan]
tbl["abs_error"] = error_table(y, est)["abs_error"].round(1).reindex(tbl.index)
print(tbl[["actual", "F", "T", "FIT", "abs_error"]])
print(f"Forecast for 2019Q1: {est['FIT'].iloc[-1]:.1f}    MAD: {mad(y, est):.1f}")

# ---------------------------------------------------------------------------
banner("2. Figure 16.5: the MAD is U-shaped in alpha")

curve = tune_alpha(y, grid=np.round(np.arange(0.05, 1.0, 0.05), 2))
print(curve.round(2).T)
print(f"\nMAD is smallest at alpha = {curve['MAD'].idxmin():.2f} "
      f"(MAD {curve['MAD'].min():.1f}); the book eyeballs 'around 0.35'.")

# ---------------------------------------------------------------------------
banner("3. Tune alpha on real series: CPI and weekly claims")

cpi = pd.read_csv(find_data("cpi.csv"), parse_dates=["date"]).set_index("date")["cpi"]
cpi = fill_gaps(cpi, "interpolate").loc["2022-01-01":]
icsa = pd.read_csv(find_data("icsa.csv"), parse_dates=["date"]).set_index("date")["initial_claims"]
claims = icsa.loc["2023-01-01":]

cpi_curve = tune_alpha(cpi)
cl_curve = tune_alpha(claims)
side = pd.DataFrame({"CPI MAD": cpi_curve["MAD"], "claims MAD": cl_curve["MAD"]})
print(side.round(2).T)
a_cpi, a_cl = cpi_curve["MAD"].idxmin(), cl_curve["MAD"].idxmin()
print(f"\nBest alpha:  CPI {a_cpi:.2f}    claims {a_cl:.2f}")
print("A trending series wants a LARGE alpha (chase the level); a noisy, flat series")
print("wants a smaller one (average the noise). The parameter is a description of")
print("the data, which is why you tune it rather than copy it from the book.")

# ---------------------------------------------------------------------------
banner("4. Add the trend component and tune alpha and beta together")

grid = tune_alpha_beta(cpi, alphas=np.round(np.arange(0.1, 1.0, 0.2), 1),
                       betas=np.round(np.arange(0.1, 1.0, 0.2), 1))
print("CPI: MAD by alpha (rows) and beta (columns)")
print(grid.round(3))
best = grid.stack().idxmin()
print(f"\nSmallest MAD at alpha = {best[0]}, beta = {best[1]}: {grid.loc[best]:.3f}")

next_month = (cpi.index[-1] + pd.offsets.MonthBegin(1)).strftime("%B %Y")
cpi_methods = {
    "naive":                     naive_forecast(cpi),
    "SMA p=3":                   sma_forecast(cpi, 3),
    f"ES alpha={a_cpi:.2f}":     exp_smooth(cpi, a_cpi),
    f"ES+trend {best[0]},{best[1]}": exp_smooth_trend(cpi, best[0], best[1]),
}
print(f"\nCPI forecast for {next_month}:")
print(accuracy(cpi, cpi_methods).round(3))

# ---------------------------------------------------------------------------
banner("5. Backtest everything: the last 24 months of CPI, one month at a time")

bt = backtest_summary(cpi, {
    "naive":        (naive_forecast, {}),
    "SMA p=3":      (sma_forecast, {"p": 3}),
    "ES 0.5":       (exp_smooth, {"alpha": 0.5}),
    f"ES {a_cpi:.2f}": (exp_smooth, {"alpha": a_cpi}),
    "ES+trend 0.5,0.3": (exp_smooth_trend, {"alpha": 0.5, "beta": 0.3}),
    f"ES+trend {best[0]},{best[1]}": (exp_smooth_trend, {"alpha": best[0], "beta": best[1]}),
}, last=24)
print(bt.round(3))
print(f"\nWinner over the last 24 months: {bt['MAD'].idxmin()}.")
print("Notice whether the tuned parameters (chosen on the whole sample) still win")
print("when only the past is available. If they do not, you have over-fit.")

print("\nSame contest for weekly claims, last 26 weeks:")
pd.set_option("display.float_format", lambda v: f"{v:,.1f}")
btc = backtest_summary(claims, {
    "naive":      (naive_forecast, {}),
    "SMA p=2":    (sma_forecast, {"p": 2}),
    "SMA p=4":    (sma_forecast, {"p": 4}),
    f"ES {a_cl:.2f}": (exp_smooth, {"alpha": a_cl}),
    "ES+trend 0.3,0.1": (exp_smooth_trend, {"alpha": 0.3, "beta": 0.1}),
}, last=26)
print(btc.round(1))

# ---------------------------------------------------------------------------
banner("6. A quarterly series: the Nevada house price index (NVSTHPI)")

hpi = pd.read_csv(find_data("nvsthpi.csv"), parse_dates=["date"]).set_index("date")["nv_hpi"]
hpi.index = hpi.index.to_period("Q")
recent = hpi.loc["2015Q1":]
print(f"{len(recent)} quarters, {recent.index[0]} to {recent.index[-1]}; next: {recent.index[-1] + 1}")
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
hc = tune_alpha(recent)
a_h = hc["MAD"].idxmin()
hpi_methods = {
    "naive":               naive_forecast(recent),
    "SMA p=2":             sma_forecast(recent, 2),
    f"ES {a_h:.2f}":       exp_smooth(recent, a_h),
    "ES+trend 0.5,0.3":    exp_smooth_trend(recent, 0.5, 0.3),
    "ES+trend 0.9,0.1":    exp_smooth_trend(recent, 0.9, 0.1),
}
print(accuracy(recent, hpi_methods).round(2))
print("\nBacktest, last 8 quarters:")
print(backtest_summary(recent, {
    "naive":            (naive_forecast, {}),
    "SMA p=2":          (sma_forecast, {"p": 2}),
    f"ES {a_h:.2f}":    (exp_smooth, {"alpha": a_h}),
    "ES+trend 0.5,0.3": (exp_smooth_trend, {"alpha": 0.5, "beta": 0.3}),
    "ES+trend 0.9,0.1": (exp_smooth_trend, {"alpha": 0.9, "beta": 0.1}),
}, last=8).round(2))

forecast_plot(recent, {"naive": naive_forecast(recent),
                       "ES+trend 0.9,0.1": exp_smooth_trend(recent, 0.9, 0.1)},
              title="Nevada house price index: actual vs. forecast", ylab="Index, 1980Q1 = 100",
              last=24)

# ---------------------------------------------------------------------------
banner("Writing assignment (do not have the assistant write this for you)")
print("""
For the series you chose in Script A: did exponential smoothing beat the
method you submitted? Report the backtested MAD of both. Then explain, in plain
English, what a large alpha means about the series and whether the trend
component earned its keep. Two paragraphs at most.
""")
