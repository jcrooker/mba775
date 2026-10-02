"""MBA 775 - Chapter 6: Continuous Probability Distributions

Run this and read the output. Nothing here needs editing to work.

    python 06_continuous_distributions.py

What it does, in order:

    1. Density is not probability   heights above 1, areas that are probabilities
    2. Normal probabilities          the call centre, and the five interval shapes
    3. z is not always normal        standardizing skewed data keeps the skew
    4. Working backwards             percentiles, a service goal, reorder points
    5. The empirical rule            computed, with Chebyshev and Cantelli
    6. Is it normal enough?          Nevada's GDP growth against the bell curve
    7. Expected value                RAD Construction with normal probabilities
    8. Normal for binomial           the continuity correction, every wording
    9. The exponential               rate versus mean, and the Poisson link
   10. The uniform                   the start-up checklist and a goal check
   11. Samples and the mechanism     precision as the sample grows

Data file it reads:

    nevada_economy.csv

Everything else is a parameter supplied in the problem, or is simulated with
a fixed seed.
"""

import sys
from math import exp, log, sqrt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                # noqa: E402
from _stats import empirical_rule                                    # noqa: E402
from _dist import binomial_table, poisson_table, emv_table          # noqa: E402
from _cont import (normal_cdf, normal_inv, normal_area,              # noqa: E402
                   normal_interval_table, normal_percentile_table,
                   empirical_rule_normal, one_sided_bounds,
                   exponential_parameters, exponential_pdf,
                   exponential_area, exponential_percentile,
                   uniform_pdf, uniform_area, uniform_mean, uniform_sd,
                   uniform_percentile, approximation_conditions,
                   continuity_guide, normal_approximation, simulate,
                   sample_vs_theory, z_tail_check, arrivals_per_interval,
                   normal_fit_check)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 70)


# ===========================================================================
banner("1. DENSITY IS NOT PROBABILITY: read the area, not the height")
# ===========================================================================

print(f"""Coffee dispensed is uniform from 7.4 to 8.2 ounces.
    height of the density, 1/(8.2 - 7.4)  = {uniform_pdf(7.8, 7.4, 8.2):.4f}
    P(7.5 < x < 8.0), the area            = {uniform_area(7.4, 8.2, 7.5, 8.0):.4f}

An exponential waiting time with lambda = 3 per hour.
    height of the density at x = 0        = {exponential_pdf(0, 3):.4f}
    P(x < 0.25 hours), the area           = {exponential_area(3, None, 0.25):.4f}

  Both heights are above 1, and neither distribution is broken. A height is a
  density. Only an AREA under the curve is a probability, and a single point
  has no area: P(x = 7.8) = 0.
""")


# ===========================================================================
banner("2. NORMAL PROBABILITIES: a difference of two cumulative probabilities")
# ===========================================================================

MU_C, SD_C = 12, 3
print("Customer service calls: normal, mean 12 minutes, sd 3 minutes.\n")
calls = normal_interval_table(MU_C, SD_C, [(None, 14), (14, None),
                                           (None, 8.5), (8.5, None)])
print(calls[["event", "shape", "computed as", "probability",
             "Excel"]].round(4).to_string())
print("""
  Rows 1 and 2 are complements and sum to 1; so do rows 3 and 4.
""")

print("Minneapolis snowfall: normal, mean 46 inches, sd 18.5 inches.\n")
snow = normal_interval_table(46, 18.5, [(None, 20), (70, None), (30, 70),
                                        (60, 75), (12, 35)])
print(snow[["event", "shape", "P(X <= upper)", "P(X <= lower)",
            "probability"]].round(4).to_string())
print("""
  Five shapes, one method: P(X <= upper) - P(X <= lower). Name the shape and
  sketch it before computing. When the interval lies entirely on one side of
  the mean, you subtract two cumulative probabilities. You never add 0.5.
""")


# ===========================================================================
banner("3. Z IS STANDARD NORMAL ONLY IF X IS NORMAL")
# ===========================================================================

waits = simulate("exponential", draws=100_000, lam=0.25)
zt = z_tail_check(waits)
print("100,000 exponential waiting times (right-skewed), standardized.\n")
print(f"    mean of z = {zt.attrs['mean of z']:.4f},  "
      f"sd of z = {zt.attrs['sd of z']:.4f}\n")
print(zt.round(4).to_string())
print("""
  Standardizing always gives mean 0 and sd 1. It does not change the shape.
  These z-scores are as skewed as the waits, so standard normal probabilities
  attached to them are wrong, even though every z was computed correctly.
""")


# ===========================================================================
banner("4. WORKING BACKWARDS: from a probability to a value")
# ===========================================================================

goal = normal_percentile_table(MU_C, SD_C, [0.95])
x95 = float(goal["x = mu + z sigma"].iloc[0])
print("Goal: 95% of calls finished in under 18 minutes.\n")
print(goal.round(4).to_string())
print(f"""
  95th percentile = {x95:.2f} minutes, inside the 18-minute target.
  Equivalently, P(x < 18) = {normal_area(MU_C, SD_C, None, 18):.4f}, above the required 0.95. The two checks always agree. The goal is met.
""")

MU_G, SD_G = 930, 140
reorder = normal_percentile_table(MU_G, SD_G, [0.90, 0.95, 0.99])
reorder["safety stock"] = reorder["x = mu + z sigma"] - MU_G
print("Gasoline demand over the 4-day lead time: normal, mean 930, sd 140.\n")
print(reorder.round(2).to_string())
print(f"""
  Each extra point of service level costs more safety stock than the last.
  The current reorder point of 1,200 gallons gives a service level of
  P(demand < 1,200) = {normal_area(MU_G, SD_G, None, 1200):.4f}.
""")


# ===========================================================================
banner("5. THE EMPIRICAL RULE, computed rather than remembered")
# ===========================================================================

print(empirical_rule_normal().round(4).to_string())
print()
print(one_sided_bounds(2).round(4).to_string())
print("""
  The 68-95-99.7 rule is the NORMAL distribution's rule. Chebyshev holds for
  any shape and bounds both tails together. For a bound on ONE tail that holds
  for any shape, use Cantelli, 1/(1 + k^2). Halving Chebyshev is not a bound,
  because a skewed distribution can put all of its outside probability in one
  tail.
""")


# ===========================================================================
banner("6. IS IT NORMAL ENOUGH? Nevada's real GDP growth")
# ===========================================================================

nv = pd.read_csv(find_data("nevada_economy.csv"), parse_dates=["date"])
n_missing = int(nv["gdp_growth"].isna().sum())
growth = nv["gdp_growth"].dropna()
print(f"{len(growth)} quarters of year-over-year growth. {n_missing} quarters have no")
print("growth rate (the first year has no year-earlier quarter) and are left out.")
print(f"mean = {growth.mean():.4f},  sd = {growth.std(ddof=1):.4f}\n")
print(empirical_rule(growth, name="Nevada growth").to_string())
print()
fit = normal_fit_check(growth, [-0.08, -0.05, 0.0, 0.05])
print(fit.round(4).to_string())
worst = nv.dropna(subset=["gdp_growth"]).nsmallest(8, "gdp_growth")
years = ", ".join(str(y) for y in sorted(set(worst["date"].dt.year)))
print(f"""
  Too many quarters near the middle, and too many in the tails.
  Below -8%: the normal model expects {fit.loc[-0.08, 'normal model says']:.1%} of quarters,
             Nevada had {fit.loc[-0.08, 'share observed']:.1%}.
  The eight worst quarters all fall in {years}: two episodes, not eight
  independent draws. A plan sized to the normal curve would understate the
  chance of a deep downturn.
""")


# ===========================================================================
banner("7. EXPECTED VALUE WITH NORMAL PROBABILITIES: RAD Construction")
# ===========================================================================

p_early = normal_area(92, 9, None, 84)
p_late = normal_area(92, 9, 106, None)
states = {"early (< 84 days)": p_early,
          "on time (84 to 106)": 1 - p_early - p_late,
          "late (> 106 days)": p_late}
print("Completion time is normal, mean 92 days, sd 9. Profit $125,000, a")
print("$20,000 bonus if under 84 days, a $15,000 penalty if over 106 days.\n")
for s, p in states.items():
    print(f"    P({s:<22}) = {p:.4f}")
rad = emv_table({"bridge contract": {"early (< 84 days)": 145_000,
                                     "on time (84 to 106)": 125_000,
                                     "late (> 106 days)": 110_000}}, states)
print()
print(rad.round(4).to_string())
print("""
  The normal model turns a belief about completion time into the three
  probabilities: a left tail, a right tail, and the interval between them.
""")


# ===========================================================================
banner("8. THE NORMAL APPROXIMATION TO THE BINOMIAL")
# ===========================================================================

print("Carl's Jr.: 6% of drive-through orders are wrong. The next 120 orders.\n")
print(approximation_conditions(120, 0.06).to_string())
print()
print(continuity_guide(3).to_string(index=False))
print()
cj = pd.DataFrame([normal_approximation(120, 0.06, "exactly", 7),
                   normal_approximation(120, 0.06, "at most", 4),
                   normal_approximation(120, 0.06, "more than", 8),
                   normal_approximation(120, 0.06, "between", 7, 9)]
                  ).set_index("question")
print(cj.round(4).to_string())
print("""
  First turn the words into a list of whole numbers, then widen the list by
  half a unit at each end. The correction is what keeps the error small; the
  uncorrected column shows what happens without it.
""")


# ===========================================================================
banner("9. THE EXPONENTIAL: get lambda and mu straight first")
# ===========================================================================

print("The same deli, described two ways, and a hockey team's shots:\n")
print(exponential_parameters(mean=4, unit="minute", event="customer")
      .round(4).to_string(), "\n")
print(exponential_parameters(rate=15, per=60, to=1, unit="minute",
                             event="customer").round(4).to_string(), "\n")
print(exponential_parameters(rate=32.2, per=60, to=1, unit="minute",
                             event="shot").round(4).to_string(), "\n")

LAM = 0.25
deli = pd.DataFrame([exponential_area(LAM, None, 2, explain=True),
                     exponential_area(LAM, 4, 8, explain=True),
                     exponential_area(LAM, 7, None, explain=True)],
                    index=["within 2 min", "4 to 8 min", "over 7 min"])
print(deli[["event", "computed as", "probability", "Excel"]]
      .round(4).to_string())
print(f"""
  mean wait   = 1/lambda             = {1 / LAM:.2f} minutes
  median wait = ln(2)/lambda         = {exponential_percentile(0.5, LAM):.2f} minutes
  95% of gaps are shorter than       = {exponential_percentile(0.95, LAM):.2f} minutes

  EXPON.DIST takes the RATE. Hand it the mean and the answer is wrong but
  looks reasonable.

  No memory: P(x > 9 | x > 5) = e^(-0.25 x 4) = {exp(-0.25 * 4):.4f} = P(x > 4).
""")

gaps = simulate("exponential", draws=20_000, lam=4)
hourly = arrivals_per_interval(gaps, interval=1)
counts = hourly.value_counts().sort_index()
pois = poisson_table(4).loc[:counts.index.max()]
link = pd.DataFrame({"share of hours observed":
                     (counts / counts.sum()).reindex(pois.index, fill_value=0),
                     "Poisson P(X = x), lambda = 4": pois["P(X = x)"]})
print("Exponential gaps (4 per hour), counted hour by hour:\n")
print(link.round(4).to_string())
print(f"""
  mean count = {hourly.mean():.3f}, variance = {hourly.var(ddof=0):.3f}, both close to lambda = 4.
  Time the gaps and you get an exponential; count the arrivals and you get a Poisson.
  One process, seen two ways.
""")


# ===========================================================================
banner("10. THE UNIFORM: width over width")
# ===========================================================================

A, B = 15, 35
print("Start-up checklist time is uniform from 15 to 35 minutes.\n")
print(f"    height f(x) = 1/(35 - 15)          = {uniform_pdf(25, A, B):.4f}")
print(f"    mean (a + b)/2                     = {uniform_mean(A, B):.2f}")
print(f"    sd (b - a)/sqrt(12)                = {uniform_sd(A, B):.4f}")
print(f"    P(20 < x < 25) = (25 - 20)/20      = {uniform_area(A, B, 20, 25):.4f}")
print(f"    80th percentile = 15 + 0.80 (20)   = {uniform_percentile(0.8, A, B):.2f}")
print(f"    P(x < 30) = (30 - 15)/20           = {uniform_area(A, B, None, 30):.4f}")
print("""
  Goal: 80% of start-ups within 30 minutes. The 80th percentile is 31 minutes
  and only 75% finish within 30, so the goal is not met.
""")


# ===========================================================================
banner("11. SAMPLES AND THE MECHANISM: precision as n grows")
# ===========================================================================

prec = pd.concat({
    "normal(10, 4)": sample_vs_theory("normal", mu=10, sigma=4),
    "exponential(0.5)": sample_vs_theory("exponential", lam=0.5),
    "uniform(10, 35)": sample_vs_theory("uniform", a=10, b=35),
})
print(prec.round(4).to_string())
print("""
  Each row is 2,000 samples of the given size. The average distance between a
  sample statistic and the truth falls by about a factor of 10 for every
  factor of 100 in the sample size. For the exponential, the truth is
  1/lambda = 2 for both the mean and the standard deviation.

  QUESTION TO ANSWER IN WRITING (one paragraph, your own words):
  Section 6 found that a normal model fitted to Nevada's growth rate expects
  fewer severe downturns than Nevada has had. You are advising a Las Vegas
  resort on how large a cash reserve to hold against a bad year. Explain to
  its CFO why "the probability of a quarter worse than -8% is about 2%" is not
  a number to plan around, and say what you would look at instead.
""")
