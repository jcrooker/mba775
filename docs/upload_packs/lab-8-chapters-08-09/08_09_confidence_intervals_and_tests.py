"""MBA 775 - Chapters 8 and 9: Confidence Intervals and Hypothesis Testing

Run this and read the output. Nothing here needs editing to work.

    python 08_09_confidence_intervals_and_tests.py

One engine does all the work this week:

    estimate  +/-  (critical value) x (standard error)

What the script does, in order:

    1. The t-distribution        critical values for t and z
    2. A confidence interval     one sample of 30 faculty, checked against
                                 the true mean
    3. What 95% means            100 intervals, then 100,000
    4. The margin of error       confidence level and sample size
    5. When to trust t           simulated error rates, normal vs skewed
    6. A proportion interval     116 of 175 at 99%
    7. Sample size               for a mean and for a proportion
    8. Errors                    false positives and false negatives
    9. A one-tailed t-test       kiosk checkout times
   10. A two-tailed t-test       Nielsen TV hours, and the matching interval
   11. A proportion test         women on French boards
   12. Power                     the Nissan Leaf: beta, power, and n
   13. Practical significance    a tiny effect, a huge sample

Data files it reads:

    big9_faculty_1999.csv        (salaries of 215 economics faculty)
    nevada_agi_2016_sample.csv   (simulated Nevada incomes; see data/README)

Everything else is a parameter from the textbook or this week's note, or is
simulated with a fixed seed.
"""

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                 # noqa: E402
from _samp import simple_random_sample                                # noqa: E402
from _infer import (critical_value_table, mean_interval,              # noqa: E402
                    mean_interval_from_data, proportion_interval,
                    interval_levels_table, interval_sizes_table,
                    sample_size_mean, sample_size_proportion,
                    sample_size_table, t_test, proportion_test,
                    interval_test_agreement, power_mean, power_table,
                    power_by_n, n_for_power, alpha_beta_table,
                    simulate_intervals, simulate_t_statistics, t_tail_rates,
                    simulate_rejections, error_table)

np.seterr(all="ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 70)


def money(x):
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def show(series):
    """Print a test or interval result: numbers first, then the words."""
    words = series.get("conclusion")
    nums = series.drop(labels=[k for k in ("conclusion",) if k in series])
    for k, v in nums.items():
        if isinstance(v, (float, np.floating)):
            v = (f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 1000
                 else f"{v:.4f}")
        print(f"    {k:<38} {v}")
    if words:
        print(f"\n    {words}")


faculty = pd.read_csv(find_data("big9_faculty_1999.csv"))
agi = pd.read_csv(find_data("nevada_agi_2016_sample.csv"))["AGI"].dropna()
MU = float(faculty["salary"].mean())


# ===========================================================================
banner("1. THE t-DISTRIBUTION: critical values, two-tailed")
# ===========================================================================

print(critical_value_table().round(3).to_string())
print("""
  t is always larger than z for the same confidence, because s varies from
  sample to sample. Past about 100 degrees of freedom the difference is
  cosmetic. Excel: =T.INV.2T(alpha, df) and =NORM.S.INV(1 - alpha/2).
""")


# ===========================================================================
banner("2. A CONFIDENCE INTERVAL for the mean faculty salary")
# ===========================================================================

srs = simple_random_sample(faculty, 30, seed=21)["salary"]
ci = mean_interval_from_data(srs, 0.95)
show(ci)
inside = ci["lower"] <= MU <= ci["upper"]
print(f"\n  True mean of all 215: {money(MU)} -> "
      f"{'INSIDE' if inside else 'OUTSIDE'} this interval.")
print("""
  We know the truth only because we have the whole population. In practice
  you never learn whether your interval caught it.
""")


# ===========================================================================
banner("3. WHAT '95% CONFIDENT' MEANS")
# ===========================================================================

hundred = simulate_intervals(faculty["salary"].to_numpy(), 30, MU,
                             reps=100, seed=8)
print(f"100 fresh samples of 30: {int(hundred['captured mu?'].sum())} intervals "
      f"captured the true mean, {int((~hundred['captured mu?']).sum())} missed.")
print("\nThe ones that missed:\n")
print(hundred.loc[~hundred["captured mu?"], ["x-bar", "lower", "upper"]]
      .round(0).to_string())
cov_norm = simulate_intervals(lambda rng, shape: rng.normal(100, 16, shape),
                              30, 100, reps=100_000)["captured mu?"].mean()
cov_fac = simulate_intervals(faculty["salary"].to_numpy(), 30, MU,
                             reps=100_000)["captured mu?"].mean()
print(f"\n100,000 intervals:  normal population {cov_norm:.4f};  "
      f"faculty salaries {cov_fac:.4f}")
print("""
  The 95% describes the METHOD. Any one interval either contains mu or it
  does not; it is wrong to say "there is a 95% probability mu is in THIS
  interval." Skewed salaries fall a little short of 95% at n = 30.
""")


# ===========================================================================
banner("4. THE MARGIN OF ERROR: three levers")
# ===========================================================================

print("QVC, sigma known = $40.60, n = 32, x-bar = $129.20:\n")
print(interval_levels_table(129.2, 40.60, 32, (0.90, 0.95, 0.99),
                            sigma_known=True).round(3).to_string())
print("\nMargin of error at 95%, s = $29,500, as n grows:\n")
print(interval_sizes_table(29_500, [25, 100, 400, 1_600])
      .round({"standard error": 1, "critical value": 3, "margin of error": 1})
      .to_string())
print("""
  More confidence costs width. Four times the data buys half the margin.
""")


# ===========================================================================
banner("5. WHEN TO TRUST t: simulated error rates of 5% one-tailed tests")
# ===========================================================================

rows = {}
normal = lambda rng, shape: rng.normal(0, 1, shape)   # noqa: E731
for n in (10, 30):
    rows[f"normal, n = {n}"] = t_tail_rates(simulate_t_statistics(normal, n, 0.0), n)
for n in (10, 30, 100, 400):
    rows[f"Nevada AGI, n = {n}"] = t_tail_rates(
        simulate_t_statistics(agi.to_numpy(), n, float(agi.mean())), n)
print(pd.DataFrame(rows).T.round(3).to_string())
print("""
  Each row is 20,000 samples from a population whose mean we know, so every
  rejection is a Type I error. A normal population gives 5% as promised.
  Skewed income data put too many errors in the lower tail and too few in
  the upper one, even at n = 100. "n >= 30" is a starting point, not a
  guarantee.
""")


# ===========================================================================
banner("6. A CONFIDENCE INTERVAL FOR A PROPORTION: 116 of 175, 99%")
# ===========================================================================

show(proportion_interval(116, 175, 0.99))
print("""
  The interval's standard error uses p-hat. A TEST of a claimed p uses the
  claimed value p0 instead (section 11).
""")


# ===========================================================================
banner("7. SAMPLE SIZE")
# ===========================================================================

print("AT&T: sigma = 400, margin $75, 95%:")
show(sample_size_mean(400, 75, 0.95))
print("\nProportion, margin 0.05, 98%, planning p = 0.12:")
show(sample_size_proportion(0.05, 0.98, 0.12))
print("\nSame, with no prior information (p = 0.5):")
show(sample_size_proportion(0.05, 0.98, 0.5))
print("\nHalving the margin quadruples the sample (proportion, 95%, p = 0.5):\n")
print(sample_size_table([0.08, 0.04, 0.02, 0.01]).to_string())
print("""
  Always round UP. The textbook's 230 and 543 use z = 2.33 from its table;
  Excel's 2.326 gives 229 and 542.
""")


# ===========================================================================
banner("8. TWO WAYS TO BE WRONG")
# ===========================================================================

print(error_table().to_string())
print("""
  "Positive" means the test REJECTS H0: it reports finding something.
  Type I error  = FALSE POSITIVE = rejecting a true H0      probability alpha
  Type II error = FALSE NEGATIVE = missing a false H0       probability beta
  Power = 1 - beta = probability of a TRUE positive.

  The textbook's FDA box (end of Section 9.5) swaps the two labels. With
  H0 "the drug has no benefit", approving a useless drug is a Type I error,
  a false positive; refusing a beneficial drug is a Type II error, a false
  negative.
""")


# ===========================================================================
banner("9. A ONE-TAILED t-TEST: are kiosks faster than 5 minutes?")
# ===========================================================================

print("H0: mu >= 5   H1: mu < 5   (n = 12, x-bar = 4.2, s = 1.42)\n")
show(t_test(4.2, 1.42, 12, 5, "less", 0.05))
print("""
  The p-value is the probability, IF H0 were true, of a sample mean this low
  or lower. It is NOT the probability that H0 is true.
""")


# ===========================================================================
banner("10. A TWO-TAILED t-TEST: Nielsen's 34.5 hours, alpha = 0.02")
# ===========================================================================

print("H0: mu = 34.5   H1: mu != 34.5   (n = 10, x-bar = 39.6, s = 16.4)\n")
show(t_test(39.6, 16.4, 10, 34.5, "two-sided", 0.02))
ci98 = mean_interval(39.6, 16.4, 10, 0.98)
print(f"\n98% confidence interval: {ci98['lower']:.2f} to {ci98['upper']:.2f}\n")
print(interval_test_agreement(39.6, 16.4, 10, [20, 25, 34.5, 50, 55],
                              alpha=0.02).round(4).to_string())
print("""
  A two-tailed test at alpha rejects exactly the claimed values that fall
  outside the (1 - alpha) interval. Failing to reject 34.5 is weak news: the
  same data fail to reject 25 and 50 as well.
""")


# ===========================================================================
banner("11. A PROPORTION TEST: women on large French boards")
# ===========================================================================

print("H0: p <= 0.34   H1: p > 0.34   (96 women of 240 sampled)\n")
show(proportion_test(96, 240, 0.34, "greater", 0.05))
print(f"\n  SE from p0 = 0.34:    {sqrt(0.34 * 0.66 / 240):.4f}   (correct for a test)")
print(f"  SE from p-hat = 0.40: {sqrt(0.40 * 0.60 / 240):.4f}   (that is the interval's)")


# ===========================================================================
banner("12. POWER: the Nissan Leaf, H0: mu <= 151, sigma = 12, n = 50")
# ===========================================================================

show(power_mean(151, 12, 50, 156, 0.05, "greater"))
sim = simulate_rejections(lambda rng, shape: rng.normal(156, 12, shape),
                          50, 151, "greater", 0.05, sigma=12)
print(f"\n  Simulated: 20,000 studies with a true mean of 156 failed to reject "
      f"{sim['failed to reject (rate)']:.4f} of the time.\n")
print("Power against several true means:\n")
print(power_table(151, 12, 50, range(151, 159)).round(4).to_string())
print("\nLowering alpha raises beta (true mean 157):\n")
print(alpha_beta_table(151, 12, 50, 157, (0.01, 0.05, 0.10)).round(4).to_string())
print("\nPower to detect a 3-mile improvement (true mean 154) as n grows:\n")
print(power_by_n(151, 12, [25, 50, 100, 200], 154).round(4).to_string())
need = n_for_power(151, 12, 154, 0.80)
print(f"\n  Cars needed for 80% power: {need['n (round UP)']}")
print("""
  The textbook's beta = 0.1190 at alpha = 0.01 comes from rounding z to 2.33
  and the critical mean to 155. Unrounded, beta = 0.1133.
""")


# ===========================================================================
banner("13. STATISTICALLY SIGNIFICANT IS NOT THE SAME AS IMPORTANT")
# ===========================================================================

big = t_test(4.95, 1.42, 20_000, 5, "less", 0.05)
big_ci = mean_interval(4.95, 1.42, 20_000, 0.95)
print(f"20,000 checkouts averaging 4.95 minutes: p-value {big['p-value']:.7f}")
print(f"95% interval: {big_ci['lower']:.3f} to {big_ci['upper']:.3f} minutes")
print("""
  Overwhelming evidence that kiosks are faster than 5 minutes -- by about
  three seconds. Large samples find trivial effects; small samples miss
  important ones. Report the interval, not just the p-value.

  QUESTION TO ANSWER IN WRITING (one paragraph, your own words):
  A vendor tells you its fraud-screening model "flags only 5% of honest
  transactions." Is that number alpha, beta, or power? Is a flagged honest
  transaction a false positive or a false negative? And why does the 5%
  NOT tell you what share of flagged transactions are actually honest?
""")
