"""MBA 775 - Chapter 2
Displaying descriptive statistics, and reading the display honestly.

Data file needed: state_hpi_ur_pop.csv
Also needs:       _charts.py
Source:           https://github.com/jcrooker/mba775

Fifty US states, each with a housing price index, an unemployment rate, a
population, and population growth over the past decade.

Every chart below is technically correct. Your job is to say what each one
supports -- and, harder, what it does not.
"""

import numpy as np
import pandas as pd

from _course import banner, find_data
from _charts import (frequency_table, contingency_table, histogram, binned_bar,
                     heatmap, ogive, scatter)

states = pd.read_csv(find_data("state_hpi_ur_pop.csv"))

# ---------------------------------------------------------------------------
banner("1. What are we looking at?")

print(f"{len(states)} states")
print(states[["HPI", "UR", "POPN", "gPOP"]].describe().round(2).to_string())

print("\nAs-of dates for each variable:")
for col, label in [("HPI_date", "Housing price index"),
                   ("UR_date", "Unemployment rate"),
                   ("POP_date", "Population")]:
    d = pd.to_datetime(states[col])
    print(f"  {label:<22} {d.max().date()}")
print("\nThese are not all measured at the same moment. Every comparison")
print("below puts them side by side anyway -- a limitation to state, not hide.")

# ---------------------------------------------------------------------------
banner("2. Distribution of population growth")

table = frequency_table(states["gPOP"], bins=np.arange(-5, 30, 5), cumulative=True)
print(table.to_string(index=False))

g = states["gPOP"]
print(f"\nmean   {g.mean():6.2f}")
print(f"median {g.median():6.2f}")
print(f"The mean sits {'above' if g.mean() > g.median() else 'below'} the median "
      f"by {abs(g.mean() - g.median()):.2f} points, which is the skew "
      f"expressed as a number.")
print(f"\nFastest: {states.loc[g.idxmax(), 'Member']} at {g.max():.1f}%")
print(f"Slowest: {states.loc[g.idxmin(), 'Member']} at {g.min():.1f}%")

_ = histogram(g, bins=np.arange(-5, 30, 5),
              title="Distribution of state population growth",
              xlab="Population growth over the decade (%)")

_ = ogive(g, bins=np.arange(-5, 30, 5),
          title="Cumulative distribution of state population growth",
          xlab="Population growth over the decade (%)")

# ---------------------------------------------------------------------------
banner("3. Does population growth go with housing prices?")

print("Binned averages first:\n")
summary = binned_bar(states["gPOP"], states["HPI"], bins=6,
                     title="Average housing price index by population growth",
                     xlab="Population growth over the decade (%)",
                     ylab="Housing price index")
print(summary.to_string())

print("\nCorrelations:")
for col, label in [("UR", "unemployment rate"),
                   ("POPN", "population"),
                   ("gPOP", "population growth")]:
    print(f"  housing price index vs {label:<20} r = {states['HPI'].corr(states[col]):+.3f}")

print("\nThe bar chart looks like a clean upward pattern. Now the same data")
print("without any averaging:")

_ = scatter(states["gPOP"], states["HPI"], labels=states["Member"],
            title="Housing price index and population growth, US states",
            xlab="Population growth over the decade (%)",
            ylab="Housing price index")

print("\nBinning averages away the disagreement. Both charts are honest;")
print("the scatter is more informative about how much confidence the")
print("pattern deserves.")

# ---------------------------------------------------------------------------
banner("4. Two variables at once")

table = contingency_table(states["gPOP"], states["HPI"], x_bins=5, y_bins=5,
                          x_name="Population growth (%)",
                          y_name="Housing price index")
print(table.to_string())

cells = table.size
empty = int((table == 0).to_numpy().sum())
print(f"\n{len(states)} states spread over {cells} cells. "
      f"{empty} cells are empty.")
print("A heatmap of a small sample looks authoritative and carries very")
print("little information. Check the counts before reading the colours.")

_ = heatmap(table, title="Housing prices and population growth, 50 states",
            xlab="Population growth over the decade (%)",
            ylab="Housing price index")

# ---------------------------------------------------------------------------
banner("5. How much does bin width decide the story?")

for k in [3, 6, 15]:
    t = frequency_table(states["gPOP"], bins=k)
    counts = ", ".join(str(c) for c in t["Frequency"])
    print(f"{k:>3} bins -> counts: {counts}")

print("\nThree bins say the distribution is smooth. Fifteen say it is lumpy.")
print("Same fifty states. Nobody looking at one histogram can tell which")
print("bin width you tried first.")

# ---------------------------------------------------------------------------
banner("Answer in writing")

print("1. Describe the population growth distribution to someone who cannot")
print("   see the chart. Use the mean, the median, and the shape.")
print()
print("2. Housing prices and population growth correlate at "
      f"{states['HPI'].corr(states['gPOP']):+.2f}. Write the strongest claim")
print("   this supports -- and a claim it does NOT support that someone")
print("   might make from the bar chart.")
print()
print("3. Pick one chart above and change one thing -- axis, bin width, chart")
print("   type -- so it tells a different story. Say what you changed and why")
print("   it worked.")
