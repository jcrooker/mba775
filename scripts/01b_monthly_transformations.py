"""MBA 775 - Chapter 1, Script B
Selecting an observation is not the same as calculating a statistic.

Data file needed: dff.csv

We have a daily interest rate and we want a monthly one. There is no single
correct answer, and the differences are not rounding error. This script shows
three defensible monthly measures and how far apart they can get.
"""

import matplotlib.pyplot as plt
import pandas as pd

from _course import banner, load_dff, month_coverage

rate = load_dff("dff.csv")

START = "2020-01-01"
recent = rate.loc[START:]

# ---------------------------------------------------------------------------
banner("1. Selecting the first of each month")

# This picks out particular DAILY observations whose date happens to be the
# first of a month. It does not CALCULATE anything about the month.
month_starts = pd.date_range(recent.index.min(), recent.index.max(), freq="MS")
selected = recent.reindex(month_starts)

print(f"Month-start dates requested : {len(month_starts)}")
print(f"Dates with no observation   : {int(selected.isna().sum())}")
print("\nFirst twelve:")
print(selected.head(12).to_string())

# ---------------------------------------------------------------------------
banner("2. Three ways to calculate a monthly rate")

monthly = pd.DataFrame({
    "mean":  recent.resample("MS").mean(),
    "first": recent.resample("MS").first(),
    "last":  recent.resample("MS").last(),
})

# Resampling always emits a bucket for the current month, however few days have
# elapsed. A 13-day "monthly mean" sitting beside 31-day neighbours is not a
# comparable number, and nothing in the output says so unless we say it.
coverage = month_coverage(recent)
monthly["days"] = coverage["observed"]
complete = coverage["complete"]

print(monthly.head(12).round(4).to_string())

if (~complete).any():
    partial = monthly.index[~complete]
    print(f"\nIncomplete month(s): "
          f"{', '.join(d.strftime('%B %Y') for d in partial)} -- "
          f"excluded from the comparisons below.")

print("\n  mean  = average conditions DURING the month")
print("  first = the rate the month OPENED with")
print("  last  = the rate the month CLOSED with")

# ---------------------------------------------------------------------------
banner("3. How much does the choice matter?")

# Compare only months that actually ran their full length.
full = monthly.loc[complete]
gap = (full["mean"] - full["last"]).abs()
worst = gap.idxmax()

print(f"Largest gap between the monthly mean and the month-end rate:")
print(f"  {gap.max():.3f} percentage points, in {worst.strftime('%B %Y')}")
print(f"    mean       : {full.loc[worst, 'mean']:.3f}")
print(f"    month-end  : {full.loc[worst, 'last']:.3f}")
print(f"\nMonths where the two differ by more than 0.25 points: "
      f"{int((gap > 0.25).sum())} of {len(gap)}")

print("\nThe gaps are largest when the Fed moves mid-month. If you are")
print("describing borrowing conditions over a month, the mean is right. If")
print("you are describing the rate a decision faced at month end, it is not.")
print("Choosing without stating the question is an unstated assumption.")

# ---------------------------------------------------------------------------
banner("4. The same three measures, plotted")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(full.index, full["mean"], label="monthly mean", linewidth=1.8)
ax.plot(full.index, full["first"], label="first of month",
        linewidth=1.0, linestyle="--")
ax.plot(full.index, full["last"], label="last of month",
        linewidth=1.0, linestyle=":")
ax.set_title("Federal Funds Effective Rate: three monthly measures")
ax.set_ylabel("Percent")
ax.legend(frameon=False)
ax.grid(color="#e6e6e6")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

banner("Something to answer in writing")
print("In one paragraph: give a business question for which the monthly mean")
print("is the correct measure, and a different question for which the")
print("month-end observation is correct. Be specific about the decision.")
