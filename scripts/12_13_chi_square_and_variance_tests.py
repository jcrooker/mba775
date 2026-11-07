"""MBA 775 - Chapters 12 and 13: Chi-Square Tests and Tests for a Population Variance

Run this and read the output. Nothing here needs editing to work.

    python 12_13_chi_square_and_variance_tests.py

Every Chapter 12 test uses one statistic,

    chi-square = sum over cells of (observed - expected)^2 / expected,

and the three tests differ only in where the EXPECTED counts come from.
Chapter 13 asks whether a SPREAD is what was claimed.

What the script does, in order:

    1. The chi-square distribution   simulated under H0, checked against the formula
    2. k proportions                 S&P 500 up days by weekday (ten years, FRED)
    3. 2 x 2 is last week's test     chi-square = z^2
    4. Goodness of fit, stated model bank drive-through demand
    5. Goodness of fit, Poisson      lambda estimated: df = k - m - 1
    6. Independence                  Best Buy brand by age, residuals, Cramer's V
    7. Small expected counts         faculty rank by sex, then combining ranks
    8. One variance                  checkout times (lower tail)
    9. Interval for sigma            Amana refrigerator temperatures
   10. Two variances                 Taco Bell (one tail), BCBS (two tails)
   11. The normality catch           how often a 5% variance test is wrong

Data: sp500_daily.csv, big9_faculty_1999.csv and nevada_agi_2016_sample.csv
(in this folder or in data/). Other data are typed in from the textbook
(Donnelly, Chapters 12 and 13).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import banner, find_data                                    # noqa: E402
from _two import two_proportion_test                                     # noqa: E402
from _chisq import (chi2_inv, k_proportions_test, gof_test, poisson_gof,  # noqa: E402
                    independence_test, variance_test_from_data,
                    sigma_interval, f_test_from_data, f_test_variances,
                    simulate_chi2_null, variance_test_error_rate)

np.seterr(all="ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 70)


def show(series):
    """Print a result, one line per item."""
    for k, v in series.items():
        if isinstance(v, (float, np.floating)):
            v = (f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 1
                 else f"{v:.4f}")
        print(f"    {k:<42} {v}")


CHECKOUT = [4.4, 3.9, 4.2, 4.5, 2.3, 2.7, 4.2, 6.0, 6.2, 6.0, 5.0, 5.1,
            1.9, 1.8, 5.6, 4.3, 3.6, 5.3, 4.0, 6.1, 6.1, 3.4, 1.3, 5.0]
AMANA = [38.2, 37.5, 29.7, 42.5, 43.4, 41.5, 41.0, 41.2, 40.0, 41.4,
         39.5, 39.8, 44.6, 42.2, 34.9, 28.5, 38.5, 34.7, 36.6, 37.0,
         32.0, 34.9, 36.9, 36.6, 34.9, 35.6, 45.6, 34.1, 45.2, 33.5]
ON_SITE = [105, 125, 60, 184, 189, 126, 211, 176, 149, 79, 122, 175]
OUTSOURCED = [163, 158, 153, 174, 138, 140, 195, 138, 189, 142, 182, 138, 175, 91]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ===========================================================================
banner("1. THE CHI-SQUARE DISTRIBUTION: simulate the statistic when H0 is true")
# ===========================================================================

drive_p = [0.20, 0.10, 0.10, 0.15, 0.25, 0.20]
sim = simulate_chi2_null(drive_p, 240, reps=20_000)
print(f"  20,000 samples of 240 from a 6-category model that is TRUE.")
print(f"  Average chi-square: {sim.mean():.3f}   (theory: the df, 5)")
print(f"  Share above the 5% critical value {chi2_inv(0.95, 5):.3f}: "
      f"{(sim > chi2_inv(0.95, 5)).mean():.4f}   (theory: 0.05)")


# ===========================================================================
banner("2. k PROPORTIONS: is the share of S&P 500 up days the same every weekday?")
# ===========================================================================

sp = pd.read_csv(find_data("sp500_daily.csv"), parse_dates=["date"]).dropna()
sp = sp.sort_values("date")
sp["up"] = sp["sp500_close"].diff() > 0
sp = sp.iloc[1:]
sp["weekday"] = pd.Categorical(sp["date"].dt.day_name(), DAYS, ordered=True)
counts = sp.groupby("weekday", observed=True)["up"].agg(["sum", "size"])
counts.columns = ["up days", "trading days"]
counts["share up"] = (counts["up days"] / counts["trading days"]).round(3)
print(f"  FRED daily closes, {sp['date'].min():%Y-%m-%d} to {sp['date'].max():%Y-%m-%d}\n")
print(counts.to_string(), "\n")
_, res = k_proportions_test(counts["up days"], counts["trading days"], DAYS)
show(res)


# ===========================================================================
banner("3. THE 2 x 2 CASE IS LAST WEEK'S TWO-PROPORTION z-TEST")
# ===========================================================================

_, chi = k_proportions_test([216, 168], [2400, 2400], ["email B", "email A"])
z = two_proportion_test(216, 2400, 168, 2400)
print(f"  z = {z['test statistic z']:.4f},  z^2 = {z['test statistic z'] ** 2:.4f},"
      f"  chi-square = {chi['chi-square statistic']:.4f}")
print(f"  p-values: z test {z['p-value']:.4f}, chi-square {chi['p-value']:.4f}")


# ===========================================================================
banner("4. GOODNESS OF FIT, model fully stated: bank drive-through (alpha = 0.05)")
# ===========================================================================

tab, res = gof_test([58, 21, 27, 20, 71, 43], drive_p,
                    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], m=0, alpha=0.05)
print(tab.round(3).to_string(na_rep=""), "\n")
show(res.drop(labels=["CHISQ.TEST"]))


# ===========================================================================
banner("5. GOODNESS OF FIT, Poisson with lambda ESTIMATED: absences per class")
# ===========================================================================

tab, res = poisson_gof(list(range(7)), [22, 38, 39, 29, 15, 5, 2], top=5)
print(tab.round(3).to_string(na_rep=""), "\n")
show(res.drop(labels=["CHISQ.TEST"]))
print("""
  df = k - m - 1 = 6 - 1 - 1 = 4: estimating lambda costs a degree of freedom.
  Excel's CHISQ.TEST would use 5 df here. A large p-value means the data are
  CONSISTENT with a Poisson model, not that the model is proven.""")


# ===========================================================================
banner("6. INDEPENDENCE: does camera brand depend on age? (Best Buy)")
# ===========================================================================

bestbuy = pd.DataFrame({"Canon": [30, 22, 8], "Nikon": [16, 25, 9],
                        "Sony": [8, 19, 13]}, index=["18-34", "35-51", "52+"])
bb = independence_test(bestbuy)
print("  Observed:\n" + bestbuy.to_string())
print("\n  Expected = row total x column total / n:\n" + bb["expected"].round(2).to_string())
print("\n  Standardized residuals (f_o - f_e)/sqrt(f_e), WHERE the dependence is:\n"
      + bb["residuals"].round(2).to_string(), "\n")
show(bb["result"])


# ===========================================================================
banner("7. SMALL EXPECTED COUNTS: faculty rank by sex, then combining ranks")
# ===========================================================================

faculty = pd.read_csv(find_data("big9_faculty_1999.csv"))
t1 = pd.crosstab(faculty["rank"], faculty["sex"])
r1 = independence_test(t1)
print("  Observed:\n" + t1.to_string())
print("\n  Expected:\n" + r1["expected"].round(2).to_string())
print(f"\n  Smallest expected count {r1['result']['smallest expected count']:.2f}"
      " is below 5: do not trust this p-value. Combine the two junior ranks.\n")
rank2 = faculty["rank"].replace({"Assistant": "Asst or Assoc", "Associate": "Asst or Assoc"})
t2 = pd.crosstab(rank2, faculty["sex"])
r2 = independence_test(t2)
print("  Observed:\n" + t2.to_string())
print("\n  Expected:\n" + r2["expected"].round(2).to_string(), "\n")
show(r2["result"])
print("""
  Observational data: association, not causation. Timing (when women entered
  the field) and promotion practices would both produce this table.""")


# ===========================================================================
banner("8. ONE VARIANCE: is the SD of self-serve checkout times below 2 minutes?")
# ===========================================================================

show(variance_test_from_data(CHECKOUT, 4.0, alternative="less"))


# ===========================================================================
banner("9. TWO TAILS AND AN INTERVAL FOR SIGMA: Amana refrigerator (target 4.0 F)")
# ===========================================================================

am = variance_test_from_data(AMANA, 16.0, alternative="two-sided")
show(am)
print()
ci = sigma_interval(am["sample sd s"], am["n"], 0.95)
show(ci)
print(f"""
  Not rejecting is not the same as showing sigma = 4. The 95% interval runs
  from {ci['lower limit for sigma']:.2f} to {ci['upper limit for sigma']:.2f} degrees F.""")


# ===========================================================================
banner("10. TWO VARIANCES: Taco Bell (one tail) and BCBS hospital stays (two tails)")
# ===========================================================================

print("  Taco Bell. H1: on-site variance > outsourced variance (set by the question)\n")
show(f_test_from_data(ON_SITE, OUTSOURCED, alternative="greater",
                      labels=("on-site", "outsourced")))
print("\n  BCBS. H1: the variances differ (no direction)\n")
show(f_test_variances(0.92, 20, 0.71, 20, alternative="two-sided",
                      labels=("Christiana", "Bayhealth")))


# ===========================================================================
banner("11. THE NORMALITY CATCH: how often does a 5% test reject a TRUE null?")
# ===========================================================================

agi = pd.read_csv(find_data("nevada_agi_2016_sample.csv"))["AGI"].dropna().to_numpy()
rows = []
for n in [10, 30, 100, 400]:
    a = variance_test_error_rate(agi, n, reps=20_000)
    z = variance_test_error_rate(None, n, reps=20_000)
    rows.append({"n": n,
                 "variance test, AGI": a["variance test rejects a true H0"],
                 "t-test, AGI": a["t-test rejects a true H0"],
                 "variance test, normal": z["variance test rejects a true H0"],
                 "t-test, normal": z["t-test rejects a true H0"]})
print(pd.DataFrame(rows).set_index("n").round(3).to_string())
print("""
  On normal data every test rejects about 5% of the time. On skewed income
  data the t-test settles toward 5% as n grows (the Central Limit Theorem);
  the variance test stays near 21% at every n. Use variance tests only on
  data you have reason to think are normal, and never as a pretest to choose
  between the pooled and Welch t-tests: just use Welch.

  QUESTION TO ANSWER IN WRITING (one paragraph, your own words):
  A casino operations manager tests whether the standard deviation of slot
  revenue per machine-day exceeds $400, using 200 machine-days, and gets
  p = 0.01. Revenue per machine-day has a long right tail. Explain why you
  would not yet trust this p-value, why collecting 2,000 machine-days would
  not fix the problem, and what you would look at first.
""")
