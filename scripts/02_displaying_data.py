"""MBA 775 - Chapter 2
Displaying descriptive statistics, and reading the display honestly.

Data file needed: state_hpi_ur_pop.csv
Also needs:       _charts.py
Source:           https://github.com/jcrooker/mba775

Fifty US states, each with a house price index, the ten-year CHANGE in that
index, an unemployment rate, a population, and population growth over the past
decade.

Section 3 is the one to read slowly. It is about what happens when you compare
two variables that are not measured in comparable units, and how to notice.

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
print(states[["HPI", "gHPI", "UR", "POPN", "gPOP"]].describe().round(2).to_string())

print("\nAs-of dates for each variable:")
for col, label in [("HPI_date", "House price index"),
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
banner("3. Does population growth go with house prices?")

print("Binned averages first:\n")
summary = binned_bar(states["gPOP"], states["HPI"], bins=6,
                     title="Average house price index by population growth",
                     xlab="Population growth over the decade (%)",
                     ylab="House price index")
print(summary.to_string())

print(f"\n  house price INDEX vs population growth   "
      f"r = {states['HPI'].corr(states['gPOP']):+.3f}")
print("\nWeak, but there. Most analyses stop here. Look at the tails first.\n")

print("Highest house price index")
print(states.nlargest(6, "HPI")[["Member", "HPI", "gPOP"]].to_string(index=False))
print("\nFastest-growing states, and where they rank on that index")
print(states.nlargest(6, "gPOP")[["Member", "gPOP", "HPI"]].to_string(index=False))

print("\nMassachusetts, New York and Rhode Island lead a chart supposedly")
print("about housing -- and they are among the SLOWEST-growing states.")
print("Texas grew 15.4% and sits in the bottom half. Read the units:\n")
print("   Series : {ST}STHPI, All-Transactions House Price Index (FHFA)")
print("   Units  : Index, 1980 Q1 = 100")
print("\nEvery state starts at 100 in 1980 BY CONSTRUCTION. So the index")
print("LEVEL does not measure what a house costs. It measures how much")
print("prices have risen since 1980 -- forty-five years of appreciation,")
print("being compared against ten years of population growth.")

# ---------------------------------------------------------------------------
banner("4. The fix: compare like with like")

print("gPOP is a ten-year change, so compare it to the ten-year change in")
print("the index, which is a percentage and IS comparable across states.\n")

summary = binned_bar(states["gPOP"], states["gHPI"], bins=6,
                     title="Average ten-year house price growth by population growth",
                     xlab="Population growth over the decade (%)",
                     ylab="House price growth over the decade (%)")
print(summary.to_string())

print()
for ycol, xcol, ylab, xlab in [
        ("HPI", "gPOP", "house price INDEX LEVEL", "population growth"),
        ("gHPI", "gPOP", "house price CHANGE (10y)", "population growth"),
        ("gHPI", "UR", "house price CHANGE (10y)", "unemployment rate")]:
    print(f"  {ylab:<26} vs {xlab:<20} "
          f"r = {states[ycol].corr(states[xcol]):+.3f}")

print("\nSame fifty states, same source, same download. The correlation went")
print("from +0.30 to +0.71 because the UNITS changed and nothing else did.")
print("Unemployment is still flat against the new variable, which matters:")
print("one pairing got much stronger and the other did not move, so this is")
print("not an artefact of making everything correlate with everything.")

print("\nNow the same data with no averaging at all:")

_ = scatter(states["gPOP"], states["gHPI"], labels=states["Member"],
            title="Ten-year house price growth and population growth, US states",
            xlab="Population growth over the decade (%)",
            ylab="House price growth over the decade (%)")

print("\nBinning averages away the disagreement. Both charts are honest;")
print("the scatter is more informative about how much confidence the")
print("pattern deserves -- and the strongest claim either supports is still")
print("that the two moved TOGETHER over this decade, not that one caused")
print("the other.")

# ---------------------------------------------------------------------------
banner("5. Two variables at once")

table = contingency_table(states["gPOP"], states["gHPI"], x_bins=5, y_bins=5,
                          x_name="Population growth (%)",
                          y_name="House price growth (%)")
print(table.to_string())

cells = table.size
empty = int((table == 0).to_numpy().sum())
print(f"\n{len(states)} states spread over {cells} cells. "
      f"{empty} cells are empty.")
print("A heatmap of a small sample looks authoritative and carries very")
print("little information. Check the counts before reading the colours.")

_ = heatmap(table, title="House price growth and population growth, 50 states",
            xlab="Population growth over the decade (%)",
            ylab="House price growth over the decade (%)")

# ---------------------------------------------------------------------------
banner("6. How much does bin width decide the story?")

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
print("2. Ten-year house price growth and population growth correlate at "
      f"{states['gHPI'].corr(states['gPOP']):+.2f}.")
print("   Write the strongest claim this supports -- and a claim it does")
print("   NOT support that someone might make from the bar chart.")
print()
print("3. The index LEVEL correlates with population growth at "
      f"{states['HPI'].corr(states['gPOP']):+.2f}.")
print("   Explain, in two sentences a manager would understand, why that")
print("   number is not a weaker version of the same finding but an answer")
print("   to a different question.")
print()
print("4. Pick one chart above and change one thing -- axis, bin width, chart")
print("   type -- so it tells a different story. Say what you changed and why")
print("   it worked.")
