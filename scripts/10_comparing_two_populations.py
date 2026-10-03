"""MBA 775 - Chapter 10: Comparing Two Populations (with a slice of Chapter 11)

Run this and read the output. Nothing here needs editing to work.

    python 10_comparing_two_populations.py

Every test this week is

    (difference in estimates - difference claimed in H0) / SE of the difference

and the DESIGN of the study decides which standard error.

What the script does, in order:

    1. The SE of a difference      simulated, checked against the formula
    2. Welch vs pooled             Maxima fuel economy and the rat study
    3. Interval for a difference   federal vs private salaries
    4. A nonzero difference        Ruby Tuesday's $3.00 threshold
    5. Paired samples              Rockstar end-aisle vs middle-aisle
    6. Why pairing matters         the same data analysed wrongly
    7. Two proportions             homeownership; good-student discounts
    8. A/B tests                   visitors per arm, and power
    9. A permutation test          shuffle the labels, no formula
   10. Many pairs, many errors     familywise false positives
   11. One-way ANOVA               airfares by airline; F = t^2
   12. Blocking                    the same fares with route as a block

Every data set is typed in from the textbook (Donnelly, Chapters 10 and 11)
or simulated with a fixed seed. No data files are needed.
"""

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import banner                                            # noqa: E402
from _two import (two_sample_t, two_sample_t_from_data, two_sample_z,  # noqa: E402
                  diff_interval, paired_t_from_data,
                  paired_vs_independent, pairing_variance,
                  two_proportion_test, two_proportion_interval,
                  ab_sample_size, ab_power, permutation_test,
                  familywise_error, simulate_pairwise_false_positives,
                  one_way_anova, block_anova, two_sample_catalog)

np.seterr(all="ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 70)


def show(series):
    """Print a result: numbers first, then the words."""
    words = series.get("conclusion")
    nums = series.drop(labels=[k for k in ("conclusion",) if k in series])
    for k, v in nums.items():
        if isinstance(v, (float, np.floating)):
            v = (f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 1000
                 else f"{v:.4f}")
        print(f"    {k:<40} {v}")
    if isinstance(words, str):
        print(f"\n    {words}")


MAXIMA_AC = [24.2, 26.1, 28.8, 30.1, 24.6, 29.1, 28.2, 27.4, 28.3, 27.9]
MAXIMA_WINDOWS = [23.0, 21.7, 32.7, 20.7, 31.1, 19.6, 26.5, 21.1, 20.6, 27.2, 26.8]
ROCKSTAR_END = [64, 54, 126, 97, 37, 74, 117, 90, 81]
ROCKSTAR_MIDDLE = [72, 41, 100, 62, 40, 60, 122, 62, 78]
airfare = pd.DataFrame({
    "fare": [1063, 1299, 995, 1299, 1406, 1314, 1493, 1443, 1577, 1314,
             1442, 1413, 1063, 1442, 1575, 1015, 856, 802, 850, 935,
             1374, 977, 1043, 1070, 1428, 1603, 1531, 1079, 807, 870],
    "airline": (["United"] * 5 + ["Air France"] * 5 + ["British Airways"] * 5) * 2,
    "route": ["Chicago-London"] * 15 + ["London-Chicago"] * 15,
})


# ===========================================================================
banner("1. THE STANDARD ERROR OF A DIFFERENCE")
# ===========================================================================

rng = np.random.default_rng(20261022)
d = (rng.normal(8_025, 225, (100_000, 100)).mean(axis=1)
     - rng.normal(8_000, 225, (100_000, 100)).mean(axis=1))
print("Two bulb lines, true means 8,025 and 8,000, sigma 225, n = 100 each.")
print(f"  simulated SD of x-bar1 - x-bar2: {d.std():.2f}")
print(f"  formula sqrt(s1^2/n1 + s2^2/n2): {sqrt(2 * 225 ** 2 / 100):.2f}")
print(f"  share of samples where line 2 LOOKS better: {(d < 0).mean():.3f}")
print("""
  Variances add even though means subtract. A 25-hour edge is less than one
  standard error, so a sample of 100 per line often gets the order wrong.
""")


# ===========================================================================
banner("2. TWO INDEPENDENT MEANS: Welch (default) and pooled")
# ===========================================================================

print("Maxima: AC on (pop 1) vs windows down (pop 2), H0: mu1 - mu2 = 0\n")
show(two_sample_t_from_data(MAXIMA_AC, MAXIMA_WINDOWS))
pooled = two_sample_t_from_data(MAXIMA_AC, MAXIMA_WINDOWS, pooled=True)
print(f"\n  Pooled test on the same data: t = {pooled['test statistic t']:.4f}, "
      f"p = {pooled['p-value']:.4f}")
print("\nRats (an experiment): enriched vs barren, H0: mu1 - mu2 <= 0\n")
rp = two_sample_t(2.6, 0.6, 20, 2.1, 0.8, 25, 0, "greater", pooled=True)
rw = two_sample_t(2.6, 0.6, 20, 2.1, 0.8, 25, 0, "greater")
print(f"    pooled: t = {rp['test statistic t']:.4f}, df = {rp['df']}, p = {rp['p-value']:.4f}")
print(f"    Welch:  t = {rw['test statistic t']:.4f}, df = {rw['df']:.2f}, p = {rw['p-value']:.4f}")
print("""
  Use Welch unless you have a reason to assume equal variances. When they
  are equal, the two tests agree closely; when they are not, pooled can mislead.
""")


# ===========================================================================
banner("3. DOES THE INTERVAL FOR THE DIFFERENCE CONTAIN 0?")
# ===========================================================================

print("Federal ($66,700, n = 35) vs private ($60,400, n = 32); sigmas known.\n")
show(two_sample_z(66_700, 12_000, 35, 60_400, 11_000, 32))
ci = diff_interval(66_700, 12_000, 35, 60_400, 11_000, 32, 0.95, sigma_known=True)
print(f"\n  95% interval for mu1 - mu2: ${ci['lower']:,.0f} to ${ci['upper']:,.0f}"
      f"  (contains 0? {ci['contains 0?']})")
print("""
  The interval excludes 0, so the two-tailed test rejects -- and the interval
  also shows how imprecisely the size of the gap is known. These are
  observational groups: the test says the averages differ, not why.
""")


# ===========================================================================
banner("4. A DIFFERENCE OTHER THAN ZERO: Ruby Tuesday, H0: mu1 - mu2 <= 3")
# ===========================================================================

show(two_sample_t(45.90, 5.60, 23, 39.25, 5.20, 19, 3.00, "greater", pooled=True))


# ===========================================================================
banner("5. PAIRED SAMPLES: Rockstar, end aisle (1) vs middle aisle (2)")
# ===========================================================================

print(pd.DataFrame({"end": ROCKSTAR_END, "middle": ROCKSTAR_MIDDLE,
                    "d": np.subtract(ROCKSTAR_END, ROCKSTAR_MIDDLE)},
                   index=pd.RangeIndex(1, 10, name="store")).T.to_string(), "\n")
show(paired_t_from_data(ROCKSTAR_END, ROCKSTAR_MIDDLE, 0, "greater", 0.05, level=0.90))
print("""
  A one-tailed test at 0.05 matches a two-sided 90% interval. Design note:
  every store ran end-aisle FIRST, so a trend in demand is a rival explanation.
""")


# ===========================================================================
banner("6. WHY PAIRING MATTERS: the same 18 numbers, analysed two ways")
# ===========================================================================

print(paired_vs_independent(ROCKSTAR_END, ROCKSTAR_MIDDLE, "greater").round(4).to_string(), "\n")
print(pairing_variance(ROCKSTAR_END, ROCKSTAR_MIDDLE).round(2).to_string())
print("""
  Stores differ hugely in size, and that spread is the SAME in both weeks
  (r is high). Pairing subtracts it out; the independent test counts it as
  noise and misses the effect.
""")


# ===========================================================================
banner("7. TWO PROPORTIONS")
# ===========================================================================

print("Homeownership: Southeast 105/150 (pop 1) vs Northeast 80/125\n")
show(two_proportion_test(105, 150, 80, 125))
hci = two_proportion_interval(105, 150, 80, 125)
print(f"\n  95% interval (unpooled SE): {hci['lower']:.4f} to {hci['upper']:.4f}")
print("\nGood students 167/1,207 (pop 1) vs others 799/4,963 had accidents;"
      " H0: p1 - p2 >= 0\n")
show(two_proportion_test(167, 1207, 799, 4963, "less"))
print("""
  The TEST pools (H0 says one common p); the INTERVAL does not. Good students
  have fewer accidents -- which justifies the discount as pricing, not as
  proof that grades cause safe driving.
""")


# ===========================================================================
banner("8. A/B TESTS: how many visitors?")
# ===========================================================================

for p2 in (0.11, 0.12, 0.13):
    r = ab_sample_size(0.10, p2)
    print(f"  10% -> {p2:.0%}: {r['n per arm (round UP)']:>6,} per arm "
          f"({r['total visitors']:,} total) for 80% power at alpha = 0.05")
print("\n  Power to detect 10% -> 12% by visitors per arm:")
for n in (500, 1000, 2000, 4000, 8000):
    print(f"    {n:>5,} per arm: {ab_power(0.10, 0.12, n):.3f}")
print("\nEmail test: B 216/2,400 (pop 1) vs A 168/2,400\n")
show(two_proportion_test(216, 2400, 168, 2400))
eci = two_proportion_interval(216, 2400, 168, 2400)
print(f"\n  95% interval for B's lift: {100 * eci['lower']:.2f} to "
      f"{100 * eci['upper']:.2f} percentage points")
print("""
  Fix n in advance, don't stop at the first p < 0.05, report the interval for
  the lift, and pick one primary metric. Random assignment is what lets you
  say the email CAUSED the difference.
""")


# ===========================================================================
banner("9. A PERMUTATION TEST: shuffle the labels")
# ===========================================================================

perm = permutation_test(MAXIMA_AC, MAXIMA_WINDOWS, reps=20_000)
w = two_sample_t_from_data(MAXIMA_AC, MAXIMA_WINDOWS)
print(f"  observed difference: {perm['observed difference']:.4f} mpg")
print(f"  share of 20,000 shuffles at least that extreme: {perm['permutation p-value']:.4f}")
print(f"  Welch p-value, for comparison:                  {w['p-value']:.4f}")
print("""
  If the labels made no difference, any shuffle is as likely as the real
  split. The share of shuffles as extreme as the data IS a p-value.
""")


# ===========================================================================
banner("10. MANY PAIRS, MANY FALSE POSITIVES (Chapter 11)")
# ===========================================================================

print(familywise_error().round(4).to_string(), "\n")
print(simulate_pairwise_false_positives(k=4, n=20, reps=5_000).to_string())
print("""
  Four groups from the SAME population: running all six t-tests finds a
  'significant' pair about one time in five; one ANOVA keeps the rate at 5%.
""")


# ===========================================================================
banner("11. ONE-WAY ANOVA: do average fares differ by airline?")
# ===========================================================================

summ, aov = one_way_anova({a: g["fare"].to_numpy()
                           for a, g in airfare.groupby("airline", sort=False)})
print(summ.round(1).to_string(), "\n")
print(aov.round(4).to_string(na_rep=""), "\n")
routes = {r: g["fare"].to_numpy() for r, g in airfare.groupby("route")}
_, aov2 = one_way_anova(routes)
t2 = two_sample_t_from_data(*routes.values(), pooled=True)
print(f"  Two groups (route): ANOVA F = {aov2.loc['Between Groups', 'F']:.4f};"
      f"  pooled t^2 = {t2['test statistic t'] ** 2:.4f}.  Same test.")


# ===========================================================================
banner("12. BLOCKING: the same fares, with route as a block")
# ===========================================================================

print(block_anova(airfare, "fare", "airline", "route").astype(float)
      .round(4).to_string(na_rep=""))
print("""
  Removing the route-to-route spread from the error term shrinks the noise,
  and the airline effect becomes significant. Blocking is pairing for more
  than two groups.
""")
print(two_sample_catalog().to_string())
print("""
  QUESTION TO ANSWER IN WRITING (one paragraph, your own words):
  A marketing team ran an A/B test of a new checkout page for two weeks,
  checked the results every morning, and stopped on day 6 when p first fell
  below 0.05. They also report that the new page "significantly" improved
  conversions for mobile users in the West region, one of 12 segments they
  examined. Explain why you would not yet trust either claim, and what you
  would ask them to do instead.
""")
