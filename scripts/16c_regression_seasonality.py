"""MBA 775 - Chapter 16, Script C
The forecasting challenge, part 3: trend projection, autocorrelation, seasonality.

    python 16c_regression_seasonality.py

What it does, in order:

    1. Trend projection on the textbook's car sales (Tables 16.8-16.9) and
       the Excel regression output of Figure 16.11C, reproduced.
    2. The Durbin-Watson test on the union-membership series (Table 16.12,
       Figure 16.13C): d, the critical values, the verdict.
    3. Multiplicative decomposition, step by step (Tables 16.14-16.21):
       centered moving average, ratio-to-moving-average, seasonal factors,
       deseasonalized trend, forecast, MAD.
    4. Seasonality with dummy variables (Table 16.24, Figure 16.18).
    5. Table 16.22: every Chapter 16 method on the car sales, one table.
    6. The same tools on Las Vegas visitor volume (twelve seasons) and on
       the Nevada house price index (trend, and whether its residuals are
       autocorrelated -- they are).

Data files it reads (from the course repository's data/ folder):

    lv_visitors.csv     nvsthpi.csv     houst.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                # noqa: E402
from _forecast import (CAR_SALES, naive_forecast, sma_forecast,     # noqa: E402
                       wma_forecast, exp_smooth, exp_smooth_trend,
                       trend_projection, durbin_watson, dw_test,
                       decompose, seasonal_dummy_regression, accuracy,
                       backtest_summary, forecast_plot)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# ---------------------------------------------------------------------------
banner("1. Trend projection: Tables 16.8-16.9 and Figure 16.11C")

y = CAR_SALES
tp = trend_projection(y, horizon=2)
print(f"y_hat = {tp['b0']:.4f} + {tp['b1']:.4f} t        (book: 115.5453 + 2.7238 t)")
print(f"Forecast, period 13: {tp['forecasts'].iloc[0]:.2f}   period 14: {tp['forecasts'].iloc[1]:.2f}")
print("\nTable 16.9  fitted values and residuals")
print(tp["table"].round(1))
print(f"MAD: {tp['MAD']:.1f}   (book 13.8)")
print("\nWhat Excel's Regression tool prints (Figure 16.11C):")
print(pd.DataFrame({"coef": [tp["b0"], tp["b1"]], "std error": tp["se"],
                    "t": tp["t"], "p-value": tp["p"]}, index=["Intercept", "Period"]).round(4))
print(f"R Square {tp['r2']:.4f}   Adjusted {tp['adj_r2']:.4f}   Standard Error {tp['std_error']:.4f}")
print("\nThe slope's p-value is 0.086: at alpha = 0.05 the trend is not significant,")
print("yet the MAD (13.8) is one of the best so far. Significance and forecast")
print("accuracy are different questions. Twelve observations is not many.")

# ---------------------------------------------------------------------------
banner("2. Autocorrelation: the Durbin-Watson test on union membership (Table 16.12)")

union = pd.Series([13.0, 12.6, 12.5, 12.1, 12.2, 12.5, 12.4, 12.5,
                   11.9, 11.8, 11.3, 11.3, 11.1, 11.1, 10.7, 10.7],
                  index=range(2002, 2018), name="pct_union")
u = trend_projection(union)
print(f"y_hat = {u['b0']:.4f} + ({u['b1']:.4f}) t      (book: 13.0850 - 0.1446 t)")
res = u["table"]["residual"]
print("\nFigure 16.13C  residuals and the pieces of Formula 16.10")
pieces = pd.DataFrame({"residual": res, "prior": res.shift(1),
                       "(e_t - e_t-1)^2": (res - res.shift(1)) ** 2, "e_t^2": res ** 2})
print(pieces.round(4))
print(f"Sums: {pieces['(e_t - e_t-1)^2'].sum():.4f} / {pieces['e_t^2'].sum():.4f}")
t = dw_test(res)
print(f"\nd = {t['d']:.4f}   d_L = {t['d_L']}   d_U = {t['d_U']}   n = {t['n']}   (book: d = 1.0085)")
print(f"Verdict: {t['verdict']}")
print("\nH0: no positive autocorrelation.  d < d_L, so reject H0. The residuals move")
print("in runs: a miss this year predicts a miss of the same sign next year. That")
print("violates the regression assumption of independent errors, and it means the")
print("t-test on the slope is not to be trusted -- though the trend line may still")
print("forecast tolerably. It also means there is information in the residuals a")
print("better model could use.")

# ---------------------------------------------------------------------------
banner("3. Multiplicative decomposition, one step at a time (Tables 16.14-16.21)")

dec = decompose(y, season_length=4, horizon=4)
print("Steps 1-3  centered moving average and ratio-to-moving-average")
print(dec["steps"].round(4))
print("\nStep 4  seasonal factors (SF) and normalized seasonal factors (NSF)")
print(dec["factors"].round(4))
print(f"Sum of SF = {dec['factors'].attrs['sum_SF']:.4f}; NSF rescaled to sum to 4.")
print(f"\nDeseasonalized trend: T_t = {dec['b0']:.4f} + {dec['b1']:.4f} t   "
      f"(book: 116.8545 + 2.5224 t, from rounded deseasonalized values)")
print("\nTable 16.19/16.21  deseasonalized series, trend, forecast, error")
print(dec["table"].round(2))
print(f"MAD: {dec['MAD']:.1f}    (book 7.0)")
print("\nTable 16.20  forecasts for 2019 (T x S)")
print(dec["future"].round(2))

# ---------------------------------------------------------------------------
banner("4. Seasonality with dummy variables (Table 16.24, Figure 16.18)")

# The book's best-subsets search keeps Period, SD1 (= Q2) and SD3 (= Q4) and
# drops the Q3 dummy. Here dummies are named by quarter: SD2, SD3, SD4, with
# Q1 as the base. drop=[3] reproduces the book's final model.
full = seasonal_dummy_regression(y, season_length=4, base_season=1, horizon=4)
print("All three dummies:")
print(full["coef"].round(4))
print(f"Adjusted R2 {full['adj_r2']:.4f}   standard error {full['std_error']:.4f}")

book = seasonal_dummy_regression(y, season_length=4, base_season=1, horizon=4, drop=[3])
print("\nThe book's model (Q3 dummy dropped):")
print(book["coef"].round(4))
print(f"Adjusted R2 {book['adj_r2']:.4f}   standard error {book['std_error']:.4f}   "
      f"(book: 77%, 9.18)")
print("\nForecasts for 2019 (book: 137, 172, 142, 162):")
print(book["future"].round(1))
print(f"MAD: {book['MAD']:.1f}")
print("\nReading a coefficient: Q2 sells 33.2 more cars than Q1 (the base) at the same")
print("point on the trend. That is the seasonal factor of decomposition (1.14) said")
print("in cars instead of ratios.")

# ---------------------------------------------------------------------------
banner("5. Table 16.22: every Chapter 16 method on the car sales")

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
everything = {
    "naive":                              naive_forecast(y),
    "SMA p=3":                            sma_forecast(y, 3),
    "SMA p=4":                            sma_forecast(y, 4),
    "WMA 3-2-1":                          wma_forecast(y, [1, 2, 3]),
    "Exp. smoothing 0.6":                 exp_smooth(y, 0.6),
    "Exp. smoothing w/ trend 0.3, 0.1":   exp_smooth_trend(y, 0.3, 0.1),
    "Trend projection":                   tp["forecast"],
    "Multiplicative decomposition":       dec["forecast"],
    "Seasonal dummies":                   book["forecast"],
}
print(accuracy(y, everything).round(2))
print("\nThe two methods that model BOTH trend and season win by a wide margin, and")
print("a four-period moving average -- one that spans all four seasons -- is third.")
print("Complexity is not what wins; matching the method to the components is.")

# ---------------------------------------------------------------------------
banner("6a. Las Vegas visitor volume: twelve seasons")

lv = pd.read_csv(find_data("lv_visitors.csv"), parse_dates=["date"]).set_index("date")["visitors"]
lv.index = lv.index.to_period("M")
print(f"{len(lv)} months, {lv.index[0]} to {lv.index[-1]}")
print("\n2020 and 2021 are not a season, they are a pandemic. Fitting seasonal factors")
print("across them would average a closed city with an open one. We start in 2022.")
post = lv.loc["2022-01":]
print(f"Using {len(post)} months from {post.index[0]}; next period: {post.index[-1] + 1}")

month_of = post.index.month
d12 = decompose(post, season_length=12, horizon=3, season_of=month_of)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
print("\nNormalized seasonal factors by calendar month:")
print(d12["factors"]["NSF"].round(3).to_frame().T)
print(f"\nDeseasonalized trend: T_t = {d12['b0']:,.0f} {'+' if d12['b1'] >= 0 else '-'} "
      f"{abs(d12['b1']):,.0f} t   (t in months; the slope is the trend per month)")
print("\nForecasts, next three months (T x S):")
fut = d12["future"].copy()
for c in ("T", "forecast"):
    fut[c] = fut[c].map(lambda v: f"{v:,.0f}")
fut["S"] = fut["S"].map(lambda v: f"{v:.3f}")
print(fut)
pd.set_option("display.float_format", lambda v: f"{v:,.0f}")

sd12 = seasonal_dummy_regression(post, season_length=12, base_season=1, horizon=3,
                                 season_of=month_of)
print("\nSeasonal-dummy regression, next three months:")
print(sd12["future"])

lv_methods = {
    "naive":            naive_forecast(post),
    "SMA p=3":          sma_forecast(post, 3),
    "SMA p=12":         sma_forecast(post, 12),
    "ES 0.3":           exp_smooth(post, 0.3),
    "Trend projection": trend_projection(post)["forecast"],
    "Decomposition":    d12["forecast"],
    "Seasonal dummies": sd12["forecast"],
}
print("\nAll methods, whole-sample accuracy:")
print(accuracy(post, lv_methods).round(0))

dw_lv = dw_test(sd12["table"]["residual"])
print(f"\nDurbin-Watson on the dummy model's residuals: d = {dw_lv['d']:.3f} "
      f"(d_L {dw_lv['d_L']}, d_U {dw_lv['d_U']}, n = {dw_lv['n']}) -> {dw_lv['verdict']}")

print("\nBacktest, last 12 months (seasonal methods refit each month):")
print(backtest_summary(post, {
    "naive":    (naive_forecast, {}),
    "SMA p=12": (sma_forecast, {"p": 12}),
    "ES 0.3":   (exp_smooth, {"alpha": 0.3}),
    "Decomposition": (lambda s: decompose(s, 12, season_of=s.index.month)["forecast"], {}),
    "Seasonal dummies": (lambda s: seasonal_dummy_regression(
        s, 12, season_of=s.index.month)["forecast"], {}),
}, last=12).round(0))

forecast_plot(post, {"Seasonal dummies": sd12["forecast"], "naive": naive_forecast(post)},
              title="Las Vegas visitor volume: actual vs. forecast", ylab="Visitors per month",
              last=30)

# ---------------------------------------------------------------------------
banner("6b. Nevada house price index: a trend line and its autocorrelated residuals")

hpi = pd.read_csv(find_data("nvsthpi.csv"), parse_dates=["date"]).set_index("date")["nv_hpi"]
hpi.index = hpi.index.to_period("Q")
recent = hpi.loc["2015Q1":]
th = trend_projection(recent, horizon=2)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
print(f"{len(recent)} quarters from {recent.index[0]}.  T_t = {th['b0']:.2f} + {th['b1']:.3f} t")
print(f"Slope p-value {th['p'][1]:.2e}, R2 {th['r2']:.3f}.  Forecast {recent.index[-1] + 1}: "
      f"{th['forecasts'].iloc[0]:.1f};  naive: {recent.iloc[-1]:.2f}")
dw_h = dw_test(th["table"]["residual"])
print(f"Durbin-Watson d = {dw_h['d']:.3f} (d_L {dw_h['d_L']}, d_U {dw_h['d_U']}) -> {dw_h['verdict']}")
print("\nA very significant slope, a high R2, and residuals that run in long streaks:")
print("the line is straight and the index is not. The last residuals tell you which")
print("side of the line the next quarter is likely to fall on -- information the")
print("trend projection ignores and the naive forecast uses without knowing it.")
print(th["table"]["residual"].tail(8).round(2).to_frame().T)

# ---------------------------------------------------------------------------
banner("6c. Housing starts (HOUST): read the units line before you fit a seasonal model")

houst = pd.read_csv(find_data("houst.csv"), parse_dates=["date"]).set_index("date")["housing_starts"]
houst.index = houst.index.to_period("M")
h_recent = houst.loc["2019-01":]
hd = decompose(h_recent, season_length=12, season_of=h_recent.index.month)
print("FRED units: 'Thousands of Units, Seasonally Adjusted Annual Rate'.")
print("Normalized seasonal factors a decomposition finds anyway:")
print(hd["factors"]["NSF"].round(3).to_frame().T)
print(f"They range {hd['factors']['NSF'].min():.3f} to {hd['factors']['NSF'].max():.3f}: the "
      f"Census Bureau already removed the season before FRED published the number.")
print("A seasonal model here is fitting noise. The same is true of CPIAUCSL and ICSA")
print("(both SA) but NOT of NVSTHPI (NSA) or visitor volume. Check the units line.")
pd.set_option("display.float_format", lambda v: f"{v:,.1f}")
print("\nBacktest, last 12 months:")
print(backtest_summary(h_recent, {
    "naive":   (naive_forecast, {}),
    "SMA p=3": (sma_forecast, {"p": 3}),
    "SMA p=6": (sma_forecast, {"p": 6}),
    "ES 0.3":  (exp_smooth, {"alpha": 0.3}),
    "Trend":   (lambda s: trend_projection(s)["forecast"], {}),
}, last=12).round(1))

# ---------------------------------------------------------------------------
banner("Writing assignment (do not have the assistant write this for you)")
print("""
For Las Vegas visitor volume: which method would you submit for next month, and
what is its backtested MAD? Explain what the seasonal factor for that month
says in plain English ("<month> runs about x% above/below an average month"), and
name one thing that happened in the last year that a seasonal model built on
2022-2025 cannot know about. Two paragraphs.
""")
