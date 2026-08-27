"""MBA 775 - Chapter 1, Script C
A cross section is a claim about timing, not just a table of numbers.

Data file needed: state_unemployment.csv

Cross-sectional data are values collected from many subjects during the SAME
time period. Downloading fifty state unemployment rates does not by itself
produce a cross section: the states have to actually be reporting the same
period. This script checks that before comparing anything.
"""

import matplotlib.pyplot as plt
import pandas as pd

from _course import banner, load_state_unemployment

panel = load_state_unemployment("state_unemployment.csv")

# ---------------------------------------------------------------------------
banner("1. What is in the file?")

print(f"Rows    : {len(panel):,}")
print(f"States  : {panel['state'].nunique()}")
print(f"Dates   : {panel['date'].min().date()} to {panel['date'].max().date()}")
print(f"Missing : {int(panel['unemployment_rate'].isna().sum())}")

# ---------------------------------------------------------------------------
banner("2. Take the latest observation for each state")

latest = (panel.dropna(subset=["unemployment_rate"])
                .sort_values("date")
                .groupby("state", as_index=False)
                .last())

print(f"One observation for each of {len(latest)} states.")

# ---------------------------------------------------------------------------
banner("3. Is this actually a cross section?")

date_counts = latest["date"].value_counts().sort_index()
print(date_counts.to_string())

if len(date_counts) == 1:
    reference = date_counts.index[0]
    print(f"\nAll {len(latest)} states report {reference.date()}.")
    print("This is a clean cross section.")
else:
    reference = date_counts.index.max()
    lagging = sorted(latest.loc[latest["date"] != reference, "state"])
    print(f"\nWARNING: {len(date_counts)} distinct dates. These states are NOT")
    print("all measured in the same period.")
    print(f"Not reporting {reference.date()}: {', '.join(lagging)}")
    print("\nDo not quietly compare them. Either restrict to a common date or")
    print("say plainly in your write-up that the periods differ.")

# ---------------------------------------------------------------------------
banner("4. Census regions")

state_regions = {
    "New England":        ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Mid-Atlantic":       ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic":     ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain":           ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific":            ["AK", "CA", "HI", "OR", "WA"],
}

lookup = {st: region for region, members in state_regions.items() for st in members}
latest["region"] = latest["state"].map(lookup)

unassigned = latest.loc[latest["region"].isna(), "state"].tolist()
print(f"States assigned to a region: {latest['region'].notna().sum()} "
      f"of {len(latest)}")
if unassigned:
    print(f"Unassigned: {unassigned}")

by_region = (latest.groupby("region")["unemployment_rate"]
                   .agg(["count", "mean", "min", "max"])
                   .round(2)
                   .sort_values("mean"))
print("\n" + by_region.to_string())

# ---------------------------------------------------------------------------
banner("5. Highest and lowest states")

ranked = latest.sort_values("unemployment_rate", ascending=False)
print("Highest five:")
print(ranked.head(5)[["state", "region", "unemployment_rate"]].to_string(index=False))
print("\nLowest five:")
print(ranked.tail(5)[["state", "region", "unemployment_rate"]].to_string(index=False))

nv = latest.loc[latest["state"] == "NV", "unemployment_rate"]
if len(nv):
    rank = int((latest["unemployment_rate"] > nv.iloc[0]).sum()) + 1
    print(f"\nNevada: {nv.iloc[0]:.1f}% -- {rank} of {len(latest)} states, "
          f"highest to lowest.")

# ---------------------------------------------------------------------------
banner("6. Distribution within each region")

order = by_region.index.tolist()
data = [latest.loc[latest["region"] == r, "unemployment_rate"].dropna()
        for r in order]

fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
           boxprops={"facecolor": "#a03123", "alpha": 0.75},
           medianprops={"color": "white"})
# Set the labels separately: boxplot's own keyword was renamed in matplotlib
# 3.9, and this works on every version.
ax.set_yticks(range(1, len(order) + 1))
ax.set_yticklabels(order)
ax.set_xlabel("Unemployment rate (percent)")
ax.set_title(f"State unemployment by Census region, {reference.date()}")
ax.grid(axis="x", color="#e6e6e6")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

banner("Something to answer in writing")
print("The regional averages differ. In two or three sentences, say what this")
print("comparison can and cannot tell you about why. What would you need to")
print("know before treating a regional gap as evidence about policy?")
