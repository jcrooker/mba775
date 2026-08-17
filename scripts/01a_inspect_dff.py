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

rate = load_dff("dff.csv")

# ---------------------------------------------------------------------------
banner("1. What did we get?")

print(f"Observations : {len(rate):,}")
print(f"First date   : {rate.index.min().date()}")
print(f"Last date    : {rate.index.max().date()}")
print(f"Missing      : {int(rate.isna().sum()):,}")

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

print(rate.describe().round(3).to_string())

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
print(f"Date range  : {rate.index.min().date()} to {rate.index.max().date()}")
print(f"Rows        : {len(rate):,}")
print(f"Retrieved   : see data/README.md in the course repository")
