"""MBA 775 - Chapter 1, Script A
Inspecting a downloaded data set before you analyze it.

The Federal Funds Effective Rate (FRED series DFF) is the interest rate at
which depository institutions lend reserve balances to each other overnight.

Data file needed: dff.csv
Source:           https://github.com/jcrooker/mba775  (data/ folder)

A program that runs without an error has not verified anything. This script
performs the checks that should precede any analysis: what did we actually
get, what type is it, what is missing, and does the coverage match what we
believe we asked for.
"""

import pandas as pd

from _course import banner, load_dff

full = load_dff("dff.csv")

# The lecture note requests this series starting in January 2020. The data file
# carries the whole published history, so we can look at both and see how much
# the answer depends on the window we chose.
LECTURE_START = "2020-01-01"
recent = full.loc[LECTURE_START:]

rate = full  # the checks below run on the full history

# ---------------------------------------------------------------------------
banner("1. What did we get?")

print(f"{'':<14}{'full history':>16}{'since ' + LECTURE_START:>18}")
print(f"{'Observations':<14}{len(full):>16,}{len(recent):>18,}")
print(f"{'First date':<14}{str(full.index.min().date()):>16}"
      f"{str(recent.index.min().date()):>18}")
print(f"{'Last date':<14}{str(full.index.max().date()):>16}"
      f"{str(recent.index.max().date()):>18}")
print(f"{'Missing':<14}{int(full.isna().sum()):>16,}{int(recent.isna().sum()):>18,}")

print("\nThe lecture note works with the shorter window. Note how different")
print("the two columns are -- every summary statistic you report depends on")
print("a choice of window that is easy to make without noticing.")

# ---------------------------------------------------------------------------
banner("2. Are the data types right?")

print(f"Values are stored as : {rate.dtype}")
print(f"Index is stored as   : {type(rate.index).__name__}")

if rate.dtype != "float64":
    print("\nWARNING: the rate column is not numeric. Any average you compute")
    print("from it would be meaningless or would fail outright.")

# ---------------------------------------------------------------------------
banner("3. What is the publication calendar?")

# DFF is published on a 7-day daily calendar, so there should be a value on
# every single calendar day -- weekends and holidays included. Many other
# daily series (DGS10, SP500) are business-day only. Do not assume: check.
expected_days = (rate.index.max() - rate.index.min()).days + 1
coverage = len(rate) / expected_days

print(f"Calendar days spanned : {expected_days:,}")
print(f"Observations present  : {len(rate):,}")
print(f"Coverage              : {coverage:.1%}")

if coverage > 0.99:
    print("\nEssentially every calendar day has a value, so this is a 7-day")
    print("daily series. Weekend and holiday dates will return data.")
else:
    print("\nA meaningful share of calendar days have no observation, so this")
    print("is probably a business-day series. Selecting by an arbitrary date")
    print("will sometimes come back empty.")

# ---------------------------------------------------------------------------
banner("4. Descriptive statistics")

comparison = pd.DataFrame({
    "full history": full.describe(),
    f"since {LECTURE_START}": recent.describe(),
})
print(comparison.round(3).to_string())

print("\nSame series, same units, two defensible samples, and almost nothing")
print("in common. A summary statistic is only as meaningful as the sample")
print("behind it -- so the sample belongs in the write-up, every time.")

# ---------------------------------------------------------------------------
banner("5. The most recent observations")

print(rate.tail(10).to_string())

# ---------------------------------------------------------------------------
banner("Provenance -- record this with your submission")

print("Series      : DFF - Federal Funds Effective Rate")
print("Source      : Federal Reserve Economic Data (FRED),")
print("              Federal Reserve Bank of St. Louis")
print("URL         : https://fred.stlouisfed.org/series/DFF")
print("Units       : Percent, not seasonally adjusted")
print(f"Date range  : {full.index.min().date()} to {full.index.max().date()}")
print(f"Rows        : {len(full):,}")
print(f"Retrieved   : see data/README.md in the course repository")
