"""MBA 775 - Chapter 16, Script A
The forecasting challenge, part 1: naive, simple and weighted moving averages.

    python 16a_naive_sma_wma.py

What it does, in order:

    1. Reproduces Donnelly's Brandywine Ford tables (16.2, 16.3, 16.4) so
       you can check every number against the book.
    2. Adds the forecast the book skips: the NAIVE forecast, "same as last
       period", which every other method has to beat.
    3. Loads the CPI (FRED: CPIAUCSL), checks it the way Chapter 1 taught,
       finds the hole in it, and says what we did about the hole.
    4. Produces next month's CPI forecast by naive, SMA and WMA, with the
       MAD, RMSE and MAPE of each over the last 24 months.
    5. Backtests: which method would have done best had you been submitting
       forecasts every month for the past two years?
    6. Repeats step 4-5 for weekly initial claims (ICSA), where the choice
       of window matters a great deal more.

Data files it reads (from the course repository's data/ folder):

    cpi.csv        icsa.csv

Every number printed was computed by the code above it. The last line of
each forecast block is the number you would submit to the challenge.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                # noqa: E402
from _forecast import (CAR_SALES, naive_forecast, sma_forecast,     # noqa: E402
                       wma_forecast, error_table, mad, accuracy,
                       tune_sma, backtest, backtest_summary, fill_gaps)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 12)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

# ---------------------------------------------------------------------------
banner("1. The textbook's car sales (Table 16.1) -- check us against the book")

y = CAR_SALES
print(y.to_frame().T)

sma3 = sma_forecast(y, 3)
print("\nTable 16.2  three-period SMA (book: forecast 145, MAD 15.9)")
print(error_table(y, sma3).round(1)[["actual", "forecast", "error", "abs_error"]])
print(f"Forecast for 2019Q1: {sma3.iloc[-1]:.1f}    MAD: {mad(y, sma3):.1f}")

sma4 = sma_forecast(y, 4)
print("\nTable 16.3  four-period SMA (book: forecast 139, MAD 11.1)")
print(f"Forecast for 2019Q1: {sma4.iloc[-1]:.1f}    MAD: {mad(y, sma4):.1f}")

wma = wma_forecast(y, [1, 2, 3])      # oldest -> newest; the book writes it 3, 2, 1
print("\nTable 16.4  three-period WMA, weights 3-2-1 (book: forecast 143, MAD 15.3)")
print(error_table(y, wma).round(1)[["actual", "forecast", "error", "abs_error"]])
print(f"Forecast for 2019Q1: {wma.iloc[-1]:.1f}    MAD: {mad(y, wma):.1f}")

# ---------------------------------------------------------------------------
banner("2. The forecast the book skips: naive (same as last period)")

nv = naive_forecast(y)
print(error_table(y, nv).round(1)[["actual", "forecast", "error", "abs_error"]])
print(f"Naive forecast for 2019Q1: {nv.iloc[-1]:.0f}    MAD: {mad(y, nv):.1f}")
print("\nThe naive forecast is a one-period SMA. It is also the benchmark: a method")
print("that cannot beat 'same as last time' has not earned its complexity.")

print("\nMAD by window length p (p = 1 is naive):")
print(tune_sma(y, max_p=6).round(2))

# ---------------------------------------------------------------------------
banner("3. Now a real series: the CPI (CPIAUCSL). First, look before you forecast")

cpi_raw = pd.read_csv(find_data("cpi.csv"), parse_dates=["date"])
cpi_raw = cpi_raw.set_index("date")["cpi"]
print(f"Observations : {len(cpi_raw):,}")
print(f"First / last : {cpi_raw.index.min().date()}  /  {cpi_raw.index.max().date()}")
print(f"Type         : {cpi_raw.dtype}")
print(f"Missing      : {int(cpi_raw.isna().sum())}")
if cpi_raw.isna().any():
    print("Missing at   :", ", ".join(str(d.date()) for d in cpi_raw.index[cpi_raw.isna()]))
    print("\nThe BLS did not publish an October 2025 CPI (the survey was not run during")
    print("the federal shutdown). FRED shows a '.' -- a hole, not a zero. A moving")
    print("average needs consecutive months, so we must decide what to do. We will")
    print("interpolate the one month and SAY SO. That is a modelling choice, and it")
    print("belongs in your write-up.")

cpi = fill_gaps(cpi_raw, "interpolate")

# Work with the recent past. The 1970s are real data, but they are not the
# regime you are forecasting. How far back to look is itself a choice.
recent = cpi.loc["2022-01-01":]
print(f"\nUsing {len(recent)} months, {recent.index.min().date()} to {recent.index.max().date()}")
print(recent.tail(6).to_frame("cpi").T)

# ---------------------------------------------------------------------------
banner("4. Next month's CPI: naive, SMA and WMA")

next_month = (recent.index[-1] + pd.offsets.MonthBegin(1)).strftime("%B %Y")
methods = {
    "naive":         naive_forecast(recent),
    "SMA p=2":       sma_forecast(recent, 2),
    "SMA p=3":       sma_forecast(recent, 3),
    "SMA p=6":       sma_forecast(recent, 6),
    "WMA 1-2-3":     wma_forecast(recent, [1, 2, 3]),
    "WMA 1-2-3-4-5-6": wma_forecast(recent, [1, 2, 3, 4, 5, 6]),
}
summary = accuracy(recent, methods)
print(f"Forecast for {next_month}, and accuracy over the months that have a forecast:\n")
print(summary.round(3))

best = summary["MAD"].idxmin()
print(f"\nRead the table. The smallest MAD belongs to: {best}.")
print("The CPI has a trend: it rises almost every month, so every average of PAST")
print("values sits below the current level, and the wider the window the further")
print("behind it lags. Donnelly warns you about exactly this on p. 770. Compare the")
print("MAD of SMA p=6 with the naive forecast's to see the size of the lag.")

# ---------------------------------------------------------------------------
banner("5. Backtest: submitting a forecast every month for the last 24 months")

bt = backtest_summary(recent, {
    "naive":     (naive_forecast, {}),
    "SMA p=2":   (sma_forecast, {"p": 2}),
    "SMA p=3":   (sma_forecast, {"p": 3}),
    "SMA p=6":   (sma_forecast, {"p": 6}),
    "WMA 1-2-3": (wma_forecast, {"weights": [1, 2, 3]}),
}, last=24)
print(bt.round(3))
print("\nEach row refits the method on the data available at the time, forecasts one")
print("month ahead, and scores the miss -- which is what the challenge does to you.")

detail = backtest(recent, naive_forecast, last=6)
print("\nThe last six naive forecasts, one month at a time:")
print(detail.round(3))

# ---------------------------------------------------------------------------
banner("6. Weekly initial claims (ICSA): where the window really matters")

icsa = pd.read_csv(find_data("icsa.csv"), parse_dates=["date"]).set_index("date")["initial_claims"]
print(f"Observations : {len(icsa):,}   {icsa.index.min().date()} to {icsa.index.max().date()}")
print(f"Missing      : {int(icsa.isna().sum())}")
claims = icsa.loc["2023-01-01":]
print(f"Using {len(claims)} weeks from {claims.index.min().date()}")
print(claims.tail(8).to_frame("claims").T)

next_week = (claims.index[-1] + pd.Timedelta(days=7)).date()
cm = {
    "naive":      naive_forecast(claims),
    "SMA p=2":    sma_forecast(claims, 2),
    "SMA p=4":    sma_forecast(claims, 4),
    "SMA p=8":    sma_forecast(claims, 8),
    "WMA 1-2-3-4": wma_forecast(claims, [1, 2, 3, 4]),
}
pd.set_option("display.float_format", lambda v: f"{v:,.1f}")
print(f"\nForecast for the week ending {next_week}:\n")
print(accuracy(claims, cm).round(1))

print("\nBacktest over the last 26 weeks:")

bt_claims = backtest_summary(claims, {
    "naive":   (naive_forecast, {}),
    "SMA p=2": (sma_forecast, {"p": 2}),
    "SMA p=4": (sma_forecast, {"p": 4}),
    "SMA p=8": (sma_forecast, {"p": 8}),
    "WMA 1-2-3-4": (wma_forecast, {"weights": [1, 2, 3, 4]}),
}, last=26)
print(bt_claims.round(1))
spread = bt_claims["MAD"].max() / bt_claims["MAD"].min() - 1
print(f"\nBest over the last 26 weeks: {bt_claims['MAD'].idxmin()}; the worst is only "
      f"{spread:.0%} behind it.")
print("Claims have no trend to speak of, only noise, so averaging costs little and")
print("the methods finish close together -- unlike the CPI, where every average lagged.")
print("Same tools, different verdict. The data decide, which is why the challenge")
print("makes you look every week rather than pick a method once.")

# ---------------------------------------------------------------------------
banner("Writing assignment (do not have the assistant write this for you)")
print("""
Pick ONE of the two series. In four or five sentences, say which method you
would submit this week, what its backtested MAD is, why the naive forecast does
(or does not) beat the moving averages for this series, and what you would
want to know before trusting the forecast next month. Keep the script; you will
rerun it when the next observation is published.
""")
