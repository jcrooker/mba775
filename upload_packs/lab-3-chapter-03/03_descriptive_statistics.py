"""MBA 775 - Chapter 3: Calculating Descriptive Statistics

Run this and read the output. Nothing here needs editing to work.

    python 03_descriptive_statistics.py

What it does, in order:

    1. Central tendency        mean, weighted mean, median, mode
    2. Shape                   symmetric, left-skewed, right-skewed
    3. Variability             range, variance, standard deviation
    4. Mean and sd together    coefficient of variation, z-scores
    5. Empirical rule and Chebyshev's Theorem
    6. Grouped data            mean and variance from counts by class
    7. Relative position       percentiles, quartiles, IQR, box plots
    8. Association             covariance and correlation

Data files it reads (all from the course repository's data/ folder):

    state_population.csv          state_hpi_ur_pop.csv
    unemployment_rate.csv         cpi_inflation.csv
    grad_student_gpa.csv          annual_returns.csv
    nevada_agi_2016_sample.csv

Every number printed below was computed from those files by the line of code
above it. If you are asked where a number came from, the answer is in the
section header.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                # noqa: E402
from _stats import (describe, five_number_summary, shape_report,     # noqa: E402
                    percentile, percentile_methods, percentile_rank,
                    quartiles, outlier_fences, z_score,
                    coefficient_of_variation, empirical_rule,
                    chebyshev, chebyshev_table, grouped_mean,
                    grouped_variance, grouped_summary, association)

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 20)

WEST = ["WA", "ID", "MT", "WY", "OR", "NV", "CA", "UT", "CO", "NM", "AZ"]
NEW_ENGLAND = ["CT", "ME", "MA", "NH", "RI", "VT"]


def load(name, **kwargs):
    return pd.read_csv(find_data(name), **kwargs)


pops = load("state_population.csv")
states = load("state_hpi_ur_pop.csv")
unrate = load("unemployment_rate.csv", parse_dates=["date"])
cpi = load("cpi_inflation.csv", parse_dates=["date"])
gpa = load("grad_student_gpa.csv")
returns = load("annual_returns.csv")
agi = load("nevada_agi_2016_sample.csv")

# FRED reports state population in thousands.
pops["People"] = pops["Population"] * 1000
west = pops[pops["Member"].isin(WEST)]
newengland = pops[pops["Member"].isin(NEW_ENGLAND)]


# ===========================================================================
banner("1. CENTRAL TENDENCY: the mean is not always the middle")
# ===========================================================================

print("Population of the Western states, ascending:\n")
print(west.sort_values("People")[["Member", "People"]]
          .assign(People=lambda d: d["People"].map("{:,.0f}".format))
          .to_string(index=False))

w_mean = west["People"].mean()
w_median = west["People"].median()
print(f"\n  mean   {w_mean:>14,.0f}")
print(f"  median {w_median:>14,.0f}")
print(f"  gap    {w_mean - w_median:>14,.0f}")
print(f"\n  States above the mean: {int((west['People'] > w_mean).sum())} "
      f"of {len(west)}")
print("  A 'central' value that most observations fall below is describing")
print("  its largest member, not its centre.")

banner("1b. The median with an even number of observations")

ne = newengland.sort_values("People")
print(ne[["Member", "People"]]
      .assign(position=range(1, len(ne) + 1),
              People=lambda d: d["People"].map("{:,.0f}".format))
      .to_string(index=False))

vals = ne["People"].to_numpy()
print(f"\n  n = {len(vals)}, so the index point is 0.5 * {len(vals)} = "
      f"{0.5*len(vals):g}, a whole number.")
print(f"  Median = average of positions 3 and 4 "
      f"= ({vals[2]:,.0f} + {vals[3]:,.0f}) / 2 = {0.5*(vals[2]+vals[3]):,.0f}")
print(f"  Mean   = {newengland['People'].mean():,.0f}")
print(f"  Gap    = {newengland['People'].mean() - np.median(vals):,.0f}"
      f"   (compare {w_mean - w_median:,.0f} out West)")

banner("1c. The weighted mean: which unemployment rate is the country's?")

plain = states["UR"].mean()
weighted = (states["UR"] * states["POPN"]).sum() / states["POPN"].sum()
print(f"  Unweighted mean of the 50 state rates : {plain:.2f}%")
print(f"  Population-weighted mean              : {weighted:.2f}%")
print(f"  Difference                            : {weighted - plain:+.2f} "
      f"percentage points")

big = states.nlargest(1, "POPN").iloc[0]
small = states.nsmallest(1, "POPN").iloc[0]
print(f"\n  The unweighted average gives {small['Member']} "
      f"({small['POPN']*1000:,.0f} people) the same weight as "
      f"{big['Member']} ({big['POPN']*1000:,.0f}).")
print("  Only one of these two numbers is the unemployment rate of the")
print("  country. Decide which question you were asked before averaging.")


# ===========================================================================
banner("2. SHAPE: compare the mean and the median, always")
# ===========================================================================

print(shape_report(unrate["unemployment_rate"], "US unemployment rate"))
print(shape_report(gpa["GPA"], "Graduate GPA               "))
print(shape_report(agi["AGI"], "Nevada AGI ($10,000s)      "))
print("""
  Right skew: the mean sits above the median, dragged there by a long
              upper tail. Income, claim sizes, waiting times.
  Left skew:  the mean sits below the median. Anything with a ceiling --
              GPA, exam scores, on-time delivery rates.
  Symmetric:  they sit close together.
""")

print(f"Only {100*(agi['AGI'] > agi['AGI'].mean()).mean():.1f}% of Nevada "
      f"households earn more than the AVERAGE household.")
print("That is what right skew does to the word 'average'.")


# ===========================================================================
banner("3. VARIABILITY: range, variance, standard deviation")
# ===========================================================================

sp = returns["sp500"].dropna()
wm = returns["wmt"].dropna()

print(describe(sp, "S&P 500 annual return").to_string())

best = returns.loc[returns["sp500"].idxmax()]
worst = returns.loc[returns["sp500"].idxmin()]
print(f"\n  Range = {sp.max():.4f} - {sp.min():.4f} = {sp.max()-sp.min():.4f}")
print(f"  Best year : {int(best['year'])}  {100*best['sp500']:+.1f}%")
print(f"  Worst year: {int(worst['year'])}  {100*worst['sp500']:+.1f}%")
print(f"\n  The range used 2 of {len(sp)} observations and ignored "
      f"{len(sp)-2}.")

print(f"\n  Sample sd     (n-1, Excel STDEV.S): {sp.std(ddof=1):.6f}")
print(f"  Population sd (n,   Excel STDEV.P): {sp.std(ddof=0):.6f}")
print("  pandas defaults to the first; numpy defaults to the second.")


# ===========================================================================
banner("4. COEFFICIENT OF VARIATION: the headline result of this chapter")
# ===========================================================================

table = pd.DataFrame({"S&P 500": describe(sp), "Walmart": describe(wm)})
print(table.loc[["n", "mean", "median", "std_dev", "range", "CV_percent"]]
           .round(4).to_string())

sp_cv, wm_cv = coefficient_of_variation(sp), coefficient_of_variation(wm)
print(f"\n  CV(S&P 500) = {sp.std(ddof=1):.4f} / {sp.mean():.4f} * 100 "
      f"= {sp_cv:.1f}")
print(f"  CV(Walmart) = {wm.std(ddof=1):.4f} / {wm.mean():.4f} * 100 "
      f"= {wm_cv:.1f}")
print(f"""
  Walmart's standard deviation is {wm.std(ddof=1)/sp.std(ddof=1):.2f} times
  the index's, so by standard deviation alone it is the more volatile
  holding. Its coefficient of variation is {'LOWER' if wm_cv < sp_cv else 'HIGHER'}, because the mean
  return it deviates around is {wm.mean()/sp.mean():.1f} times larger.

  Standard deviation answers "how much does it move?"
  Coefficient of variation answers "how much does it move, per unit of
  return earned?" -- which is the question a committee is asking.
""")


# ===========================================================================
banner("5. Z-SCORES, THE EMPIRICAL RULE, AND CHEBYSHEV")
# ===========================================================================

infl = cpi.dropna(subset=["inflation"])
mu, sd = infl["inflation"].mean(), infl["inflation"].std(ddof=1)
peak = infl.loc[infl["inflation"].idxmax()]
z_peak = z_score(peak["inflation"], mu, sd)

print(f"  12-month inflation, {infl['date'].min():%b %Y} to "
      f"{infl['date'].max():%b %Y}   (n = {len(infl):,})")
print(f"  mean {mu:.4f}   sd {sd:.4f}")
print(f"\n  Highest month: {peak['date']:%B %Y} at "
      f"{100*peak['inflation']:.2f}%")
print(f"  z = ({peak['inflation']:.4f} - {mu:.4f}) / {sd:.4f} = {z_peak:.4f}")
print(f"  {'Above' if abs(z_peak) > 3 else 'Below'} the |z| > 3 outlier "
      f"threshold.")
print(f"  Months higher than that one: "
      f"{int((infl['inflation'] > peak['inflation']).sum())} of {len(infl):,}")

banner("5b. Does the empirical rule apply? Check, do not assume.")

print("US unemployment rate:")
print(empirical_rule(unrate["unemployment_rate"]).to_string())
print("\nNevada AGI:")
er = empirical_rule(agi["AGI"])
print(er.to_string())
print(f"""
  Look at interval_low for 3 sd on the AGI table: {er.loc['3 sd','interval_low']:,.2f}.
  A negative income, in data where nothing can be below zero. The empirical
  rule did not fail -- it was never applicable. It describes a symmetric
  bell curve, and income is not one.
""")

banner("5c. Chebyshev: the bound that always holds, and is always loose")

gpa_mu, gpa_sd = gpa["GPA"].mean(), gpa["GPA"].std(ddof=1)
z_gpa = z_score(3.0, gpa_mu, gpa_sd)
print(f"  Graduate GPA: mean {gpa_mu:.4f}, sd {gpa_sd:.4f}")
print(f"  z for a 3.0 GPA = (3.0 - {gpa_mu:.4f}) / {gpa_sd:.4f} = "
      f"{z_gpa:.4f}\n")
print(chebyshev_table(z_gpa).to_string())

actual = 100 * (gpa["GPA"] < 3.0).mean()
bound = (100 - chebyshev(z_gpa)) / 2
print(f"\n  Chebyshev bound on the share below 3.0: at most {bound:.2f}%")
print(f"  Actual share below 3.0                 : {actual:.2f}%")
print(f"""
  The bound held and it held by a wide margin. That is the normal outcome.
  Chebyshev tells you what CANNOT be true, not what is likely, and it is
  the only tool available when you do not know the distribution's shape.

  Two cautions. It bounds both tails together, so halving the remainder
  assumes a symmetry the theorem never promised -- report a one-tail figure
  as an upper bound, never as an estimate. And for |z| <= 1 it returns
  nothing usable at all.
""")


# ===========================================================================
banner("6. GROUPED DATA: when you never see the individual observations")
# ===========================================================================

# IRS Statistics of Income, tax year 2016, zip code 89015 (Henderson, NV).
# Counts of returns by adjusted gross income bracket.
brackets = pd.DataFrame({
    "class": ["$1 - $25,000", "$25,000 - $50,000", "$50,000 - $75,000",
              "$75,000 - $100,000", "$100,000 - $200,000",
              "$200,000 - $300,000"],
    "low": [1, 25_000, 50_000, 75_000, 100_000, 200_000],
    "high": [25_000, 50_000, 75_000, 100_000, 200_000, 300_000],
    "frequency": [6_600, 4_820, 2_550, 1_600, 1_790, 310],
})
brackets["midpoint"] = (brackets["low"] + brackets["high"]) / 2

print(grouped_summary(brackets["frequency"], brackets["midpoint"],
                      labels=brackets["class"]).round(2).to_string(index=False))

n = int(brackets["frequency"].sum())
g_mean = grouped_mean(brackets["frequency"], brackets["midpoint"])
g_var = grouped_variance(brackets["frequency"], brackets["midpoint"])
print(f"\n  n              = {n:,}")
print(f"  grouped mean   = {g_mean:,.2f}")
print(f"  grouped var    = {g_var:,.2f}")
print(f"  grouped sd     = {np.sqrt(g_var):,.2f}")

z_agi = z_score(250_000, g_mean, np.sqrt(g_var))
print(f"\n  Share above $250,000, by Chebyshev:")
print(f"  z = (250,000 - {g_mean:,.2f}) / {np.sqrt(g_var):,.2f} = {z_agi:.4f}")
print(chebyshev_table(z_agi).to_string())
top = brackets["frequency"].iloc[-1]
print(f"\n  Bound: at most {(100-chebyshev(z_agi))/2:.2f}% above $250,000.")
print(f"  The published table: {top:,} of {n:,} returns "
      f"({100*top/n:.2f}%) are in the top bracket at all.")
print("  When you have the counts, count. Chebyshev is for when you do not.")
print("\n  Note: every observation has been replaced by its class midpoint,")
print("  so these are approximations. An open-ended top class ('$200,000 or")
print("  more') would force you to invent a midpoint, and the standard")
print("  deviation would inherit whatever you invented.")


# ===========================================================================
banner("7. RELATIVE POSITION: percentiles, quartiles, the IQR")
# ===========================================================================

p80 = percentile(gpa["GPA"], 0.80)
print(f"  n = {len(gpa):,} students")
print(f"  Index point for the 80th percentile: 0.80 * {len(gpa)} = "
      f"{0.80*len(gpa):g}")
print(f"  80th percentile GPA = {p80:.4f}\n")
print("  The same percentile, three different rules:")
print(percentile_methods(gpa["GPA"], 0.80).to_string())
print("\n  This course uses the first (Excel PERCENTILE.EXC). They agree on")
print("  large data sets and disagree on small ones. Neither is wrong; being")
print("  able to say which one you used is the point.")

rank = percentile_rank(gpa["GPA"], 3.75)
print(f"\n  Percentile RANK of a 3.75 GPA = {rank:.2f}")
print(f"  ({int((gpa['GPA'] < 3.75).sum()):,} of {len(gpa):,} students are "
      f"below it)")
print("  Percentile takes a percentage and returns a value.")
print("  Percentile rank takes a value and returns a percentage.")

q1, q2, q3 = quartiles(gpa["GPA"])
lo, hi = outlier_fences(gpa["GPA"])
print(f"\n  Q1 {q1:.4f}   median {q2:.4f}   Q3 {q3:.4f}   IQR {q3-q1:.4f}")
print(f"  Box plot fences: [{lo:.4f}, {hi:.4f}]")
outliers = gpa["GPA"][(gpa["GPA"] < lo) | (gpa["GPA"] > hi)]
print(f"  Observations outside the fences: {len(outliers)}")

banner("7b. The five-number summary")

print(pd.DataFrame({
    "Graduate GPA": five_number_summary(gpa["GPA"]),
    "S&P 500": five_number_summary(sp),
    "Walmart": five_number_summary(wm),
    "Unemployment": five_number_summary(unrate["unemployment_rate"]),
}).to_string())


# ===========================================================================
banner("8. ASSOCIATION: covariance and correlation")
# ===========================================================================

pair = returns.dropna(subset=["sp500", "wmt"])
result = association(pair["sp500"], pair["wmt"], "sp500", "wmt")
print(result.round(6).to_string())

print(f"""
  The covariance is {result['covariance']:.6f}. Its SIGN is informative: the
  two {'move together' if result['covariance'] > 0 else 'move in opposite directions'}. Its SIZE is not, because it carries the units of
  both variables multiplied together. Restate returns as percentages instead
  of decimals and the covariance changes by a factor of 10,000 while the
  relationship does not change at all.

  Dividing by both standard deviations strips the units out and leaves
  {result['correlation']:+.4f}, on a scale from -1 to +1 that means the same thing whatever
  the variables were measured in.
""")

print("Correlation matrix, state panel:\n")
print(states[["HPI", "UR", "POPN", "gPOP"]].dropna().corr().round(3).to_string())

print("""
  Four things this matrix does not tell you:
    1. Causation -- in either direction, or a third variable driving both.
    2. Non-linear relationships. A perfect U-shape correlates at zero.
       Draw the scatter plot before trusting the coefficient.
    3. What outliers did to it. One extreme pair can create or destroy a
       correlation, and the number gives no hint.
    4. Whether the sample is large enough for the number to mean anything.
       That is a Chapter 9 question.
""")

banner("Done")
print("Every number above came from the CSV files named at the top of this")
print("script. If a figure in your write-up is not in this output, say where")
print("it came from.")
