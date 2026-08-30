"""MBA 775 - Chapter 2
Displaying descriptive statistics, and reading the display honestly.

Data files needed: world_poverty_income.csv, gdp_per_capita_panel.csv,
                   state_hpi_ur_pop.csv
Also needs:        _course.py, _charts.py
Source:           https://github.com/jcrooker/mba775

Two data sets, deliberately.

First, two centuries of world income and extreme poverty. This is what a chart
is FOR: a pattern so clear that nobody has to be talked into it.

Then fifty US states, with a house price index, the ten-year change in that
index, a count of the quarters in which the index fell, an unemployment rate,
a population, and population growth. This is a chart that LOOKS like a pattern
and is measuring the wrong thing until you read the units.

Every chart below is technically correct. Your job is to say what each one
supports -- and, harder, what it does not.
"""

import numpy as np
import pandas as pd

from _course import banner, find_data
from _charts import (frequency_table, contingency_table, histogram, binned_bar,
                     discrete_histogram, heatmap, ogive, scatter,
                     connected_scatter)

world = pd.read_csv(find_data("world_poverty_income.csv"))
gdp_panel = pd.read_csv(find_data("gdp_per_capita_panel.csv"))
states = pd.read_csv(find_data("state_hpi_ur_pop.csv"))

# ---------------------------------------------------------------------------
banner("1. Two centuries of data, as a table")

print("The Maddison Project's GDP per capita, every country, every year:\n")
print(f"  {len(gdp_panel):,} rows")
print(f"  {gdp_panel['entity'].nunique()} countries and regions")
print(f"  years {gdp_panel['year'].min()} to {gdp_panel['year'].max()}\n")
print(gdp_panel.head(10).to_string(index=False))

print("\nTen of twenty-one thousand rows. Nobody will read this, and printing")
print("more would not help. So narrow it all the way down to the world.\n")

show = pd.DataFrame({
    "Year": world.year,
    "GDP per capita": world.gdp_pc,
    "% in extreme poverty": world.share_poor,
    "In poverty (millions)": world.n_poor / 1e6,
    "World population (millions)": (world.n_poor + world.n_notpoor) / 1e6,
})
print(show.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

print("\nTEN ROWS. This one you can read, and it tells you two things at once:")
print("income went up, and the share in poverty went down.")
print("\nNow say what SHAPE that relationship has. Steady? Accelerating?")
print("Flattening at high incomes? Tight, or loose with exceptions?")
print("\nYou cannot answer that from ten rows. The table is not hiding the")
print("facts. It is hiding the STRUCTURE.")

# ---------------------------------------------------------------------------
banner("2. The same ten rows, drawn")

_ = connected_scatter(world["gdp_pc"], world["share_poor"],
                      labels=world["year"], logx=True,
                      title="Extreme poverty and income, the world 1820 to 2015",
                      xlab="GDP per capita (international-$, 2011 prices, log scale)",
                      ylab="Share living in extreme poverty (%)")

print("Neither axis is time. Both are measured quantities, and the points are")
print("joined in the order they happened -- so the line is the path the world")
print("has travelled through two centuries. It goes one way.\n")

first, last = world.iloc[0], world.iloc[-1]
for row in (first, last):
    pop = row.n_poor + row.n_notpoor
    print(f"  {int(row.year)}   {row.share_poor:5.1f}% in extreme poverty   "
          f"GDP per capita ${row.gdp_pc:,.0f}   "
          f"world population {pop/1e9:.2f} billion")

print("\nIn 1820, NINE PEOPLE IN TEN lived in extreme poverty. By 2015, one in")
print("ten did. There is no precedent for that in recorded history, and no")
print("table of ten rows makes you feel it.\n")

print(f"  correlation of poverty share with GDP per capita      "
      f"r = {world['share_poor'].corr(world['gdp_pc']):+.3f}")
print(f"  correlation of poverty share with LOG GDP per capita  "
      f"r = {world['share_poor'].corr(np.log(world['gdp_pc'])):+.3f}")

print("\nBoth strong; the second stronger. That is the justification for the")
print("log axis -- income spans more than an order of magnitude here, and the")
print("relationship is closer to straight when income is read in MULTIPLES")
print("rather than in dollars. On a linear axis the points bunch at the left")
print("and the last century swamps the picture. Neither is false. They answer")
print("different questions, and the axis is where you choose which.")

# ---------------------------------------------------------------------------
banner("3. The number that did not fall")

peak = world.loc[world["n_poor"].idxmax()]
print("Number of people in extreme poverty")
print(f"  1820   {first.n_poor/1e6:>7,.0f} million")
print(f"  {int(peak.year)}   {peak.n_poor/1e6:>7,.0f} million   <- the peak")
print(f"  2015   {last.n_poor/1e6:>7,.0f} million\n")
print(f"The SHARE fell from {first.share_poor:.1f}% to {last.share_poor:.1f}%.")
print(f"The COUNT fell by only {100*(1 - last.n_poor/first.n_poor):.0f}%.")
print(f"World population grew "
      f"{(last.n_poor+last.n_notpoor)/(first.n_poor+first.n_notpoor):.1f} times.")

print("\nBoth of these are true, from the same two columns:")
print("   'Extreme poverty has almost been eliminated -- 89% to 10%.'")
print("   '733 million people still live in extreme poverty, barely fewer")
print("    than in 1820.'")
print("\nWhich one you put in the caption decides what your reader believes,")
print("and neither is a lie. The arithmetic is not the argument. The choice")
print("of DENOMINATOR is.")

print("\nProvenance, before anyone quotes a number off this:")
print("  Poverty : Ravallion (2016) updated with World Bank (2019), via OWID")
print("  Line    : $1.90 per day, international-$ at 2011 prices")
print("            -- since superseded by $2.15 (2017) and $3.00 (2021)")
print("  Income  : Maddison Project Database 2023, constant 2011 int'l $")
print(f"  Coverage: {int(world.year.min())} to {int(world.year.max())}, "
      f"{len(world)} observations")
print("\nThe pre-1950 figures are RECONSTRUCTIONS, not measurements -- nobody")
print("surveyed household consumption in 1820, and different teams publish")
print("different reconstructions. The level is uncertain. The direction is not.")
print("\nThe finding survives all four caveats. That is what a robust result")
print("looks like. The next chart does not survive its first one.")

# ---------------------------------------------------------------------------
banner("4. And here is a chart that looks like a pattern")

print(f"A different data set: {len(states)} US states.\n")
print(states[["HPI", "gHPI", "nDOWN", "UR", "POPN", "gPOP"]]
      .describe().round(2).to_string())

print("\nAs-of dates for each variable:")
for col, label in [("HPI_date", "House price index"),
                   ("UR_date", "Unemployment rate"),
                   ("POP_date", "Population")]:
    d = pd.to_datetime(states[col])
    print(f"  {label:<22} {d.max().date()}")
print("\nThese are not all measured at the same moment. Every comparison")
print("below puts them side by side anyway -- a limitation to state, not hide.")

# ---------------------------------------------------------------------------
banner("5. Does population growth go with house prices?")

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
banner("6. The fix: compare like with like")

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

pair = states[states["Member"].isin(["NH", "ND"])][["Member", "gPOP", "gHPI"]]
print("\nTwo states with IDENTICAL population growth:")
print(pair.to_string(index=False))

tx = states[states["Member"] == "TX"].iloc[0]
rank = int((states["gPOP"] > tx["gPOP"]).sum()) + 1
print(f"\nTexas grew {tx['gPOP']:.1f}% ({rank}th fastest) and appreciated "
      f"{tx['gHPI']:.1f}%, against a median of {states['gHPI'].median():.1f}%.")

print("\nBinning averages away the disagreement. Both charts are honest;")
print("the scatter is more informative about how much confidence the")
print("pattern deserves -- and the strongest claim either supports is still")
print("that the two moved TOGETHER over this decade, not that one caused")
print("the other.")

# ---------------------------------------------------------------------------
banner("7. Two kinds of number: counted or measured?")

print("DISCRETE data are COUNTED. Whole numbers only. There is no 3.4th")
print("quarter and no 2.8th iPad.")
print("CONTINUOUS data are MEASURED. Any real number in the range, limited")
print("only by the precision of the instrument.\n")
print("The acid test: how many values are possible inside an interval?")
print("Between 0 and 3 a count can take four (0,1,2,3); a measurement can")
print("take infinitely many. Finite means discrete. Infinite means continuous.\n")

for col, label in [("nDOWN", "Falling quarters (counted)"),
                   ("gPOP", "Population growth % (measured)")]:
    print(f"  {label:<34} {states[col].nunique():>2} distinct values "
          f"among {len(states)} states")

print("\nThat count is what decides whether each value can be its own class.")

# ---------------------------------------------------------------------------
banner("8. An UNGROUPED frequency distribution (discrete data)")

print("For each state: the last ten years of the quarterly house price index,")
print("forty quarterly changes, and a count of how many were NEGATIVE. How")
print("many times in a decade did that state's housing market go backwards?\n")

table = frequency_table(states["nDOWN"], bins="each", cumulative=True)
print(table.to_string(index=False))

v = states["nDOWN"]
print(f"\nn = {len(v)} states, {int(v.min())} to {int(v.max())} falling quarters")
print(f"mean {v.mean():.2f}   median {v.median():.0f}")
print("Never fell:  " + ", ".join(states.loc[v == v.min(), "Member"]))
print("Fell most:   " + ", ".join(states.loc[v == v.max(), "Member"])
      + f" ({int(v.max())} of 40 quarters)")

print("\nNo bins were chosen. No width was decided. No boundary had to be")
print("argued over. The values ARE the classes.")

print("\nNow look at the row for 6. Its frequency is ZERO, and it is still")
print("there. An empty class is a FINDING -- it says the distribution has a")
print("gap in it. Delete the row and the 5 and the 7 sit next to each other")
print("as though nothing were missing. Same rule as Tukey's: do not cut your")
print("leafless stems.")

_ = discrete_histogram(states["nDOWN"],
                       title="Quarters with falling house prices, last ten years",
                       xlab="Number of falling quarters (out of 40)")

print("\nNotice the GAPS between the bars. Discrete data: bars separated,")
print("because nothing exists between 2 and 3 to draw. Continuous data: bars")
print("touching, because every value in between exists and belongs to a class.")
print("Excel left alone will get this backwards. Fixing it is on you.")

print("\nThis outlives chapter 2. Chapter 5 builds probability distributions")
print("for DISCRETE variables, where 'what is the probability of exactly 3?'")
print("has a real answer. Chapter 6 does CONTINUOUS variables, where that")
print("probability is zero and every question is about an INTERVAL instead.")

# ---------------------------------------------------------------------------
banner("9. A GROUPED frequency distribution (continuous data)")

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
banner("10. Two variables at once")

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
banner("11. How much does bin width decide the story?")

for k in [3, 6, 15]:
    t = frequency_table(states["gPOP"], bins=k)
    counts = ", ".join(str(c) for c in t["Frequency"])
    print(f"{k:>3} bins -> counts: {counts}")

print("\nThree bins say the distribution is smooth. Fifteen say it is lumpy.")
print("Same fifty states. Nobody looking at one histogram can tell which")
print("bin width you tried first.")

# ---------------------------------------------------------------------------
banner("Answer in writing")

print("1. The world poverty chart and the housing chart are both correct.")
print("   One is worth showing and one needed repair first. In two sentences,")
print("   say what you would check before trusting ANY chart of two variables.")
print()
print("2. Extreme poverty fell from 89% to 10% of the world, while the NUMBER")
print("   of people in extreme poverty fell only 24%. Write the caption you")
print("   would put on the chart, and say what it leaves out.")
print()
print("3. Describe the population growth distribution to someone who cannot")
print("   see the chart. Use the mean, the median, and the shape.")
print()
print("4. One of the two distributions above was drawn with gaps between the")
print("   bars and one without. Say which, say why, and say what a reader")
print("   would wrongly conclude if the two were swapped.")
print()
print("5. Ten-year house price growth and population growth correlate at "
      f"{states['gHPI'].corr(states['gPOP']):+.2f}.")
print("   Write the strongest claim this supports -- and a claim it does")
print("   NOT support that someone might make from the bar chart.")
print()
print("6. The index LEVEL correlates with population growth at "
      f"{states['HPI'].corr(states['gPOP']):+.2f}.")
print("   Explain, in two sentences a manager would understand, why that")
print("   number is not a weaker version of the same finding but an answer")
print("   to a different question.")
print()
print("7. Pick one chart above and change one thing -- axis, bin width, chart")
print("   type -- so it tells a different story. Say what you changed and why")
print("   it worked.")
