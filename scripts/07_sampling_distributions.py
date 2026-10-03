"""MBA 775 - Chapter 7: Sampling and Sampling Distributions

Run this and read the output. Nothing here needs editing to work.

    python 07_sampling_distributions.py

What it does, in order:

    1. Sampling designs          simple random, systematic, stratified, cluster
    2. Weighting a stratified    why the plain average of a stratified sample
       sample                    can be badly wrong, and how to fix it
    3. Sampling error            every possible sample from a small population
    4. Nonsampling error         self-selected reviews: more data, same bias
    5. Every possible sample     the entree prices: mu, and sigma / sqrt(n)
    6. A sample mean             text messages, and sigma versus the SE
    7. When is n large enough?   the CLT and Nevada income
    8. Judging a claim           the commute: interval and tail probability
    9. Finite populations        the correction, and why it goes to zero
   10. Proportions               the Super Bowl and the continuity correction
   11. The bootstrap             a standard error from one sample

Data files it reads:

    big9_faculty_1999.csv        (salaries of 215 economics faculty)
    nevada_agi_2016_sample.csv   (simulated Nevada incomes; see data/README)

Everything else is a parameter supplied in the problem, or is simulated with
a fixed seed.
"""

import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                 # noqa: E402
from _cont import normal_area                                          # noqa: E402
from _samp import (simple_random_sample, systematic_sample,           # noqa: E402
                   proportional_allocation, stratified_sample,
                   stratified_estimate, cluster_sample, design_comparison,
                   all_sample_errors, enumerate_sample_means, fpc,
                   se_mean, se_proportion, fpc_table, xbar_probability,
                   xbar_interval, interval_table, proportion_question,
                   simulate_sample_means, tail_check, bootstrap,
                   bootstrap_interval, selection_bias_demo)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)


def money(x):
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


faculty = pd.read_csv(find_data("big9_faculty_1999.csv"))
agi = pd.read_csv(find_data("nevada_agi_2016_sample.csv"))["AGI"].dropna()
RANKS = faculty["rank"].value_counts().to_dict()
MU = float(faculty["salary"].mean())


# ===========================================================================
banner("1. SAMPLING DESIGNS: the 215 Big Ten economists of 1999")
# ===========================================================================

print(f"Population: {len(faculty)} faculty, true mean salary {money(MU)}.")
print(f"Missing salaries: {faculty['salary'].isna().sum()}\n")
print(faculty.groupby("rank")["salary"].agg(["size", "mean"]).round(0)
      .to_string(), "\n")

srs = simple_random_sample(faculty, 30, seed=7)
print(f"Simple random sample of 30 (without replacement): mean "
      f"{money(srs['salary'].mean())}, sampling error "
      f"{money(srs['salary'].mean() - MU)}")

alloc = proportional_allocation(RANKS, 30)
print("\nProportional allocation of a stratified sample of 30:\n")
print(alloc.round(3).to_string())

cl = cluster_sample(faculty, "institution", 3, seed=5)
print(f"\nCluster sample: campuses {', '.join(cl.attrs['clusters chosen'])}, "
      f"everyone there: {len(cl)} faculty, mean {money(cl['salary'].mean())}")

# Periodicity: a restaurant's daily covers with a weekly cycle (illustrative).
rng = np.random.default_rng(20261003)
days = pd.date_range("2025-01-06", periods=364, freq="D")
level = {0: 210, 1: 200, 2: 215, 3: 240, 4: 340, 5: 380, 6: 300}
covers = pd.Series([level[d.dayofweek] + rng.normal(0, 25) for d in days],
                   index=days)
print(f"\nPeriodicity. Illustrative daily covers, true mean {covers.mean():.1f}.")
for start in range(7):
    s = systematic_sample(covers, 52, start=start)
    print(f"    every 7th day starting on a {s.index[0].day_name():<9}: "
          f"mean {s.mean():6.1f}")
print("""
  With k = 7 every sampled day is the same weekday, so the answer depends on
  where you start. Before sampling systematically, ask what order the list
  is in.
""")


# ===========================================================================
banner("2. WEIGHTING A STRATIFIED SAMPLE")
# ===========================================================================

EQUAL = {"Assistant": 10, "Associate": 10, "Professor": 10}
UNEQUAL = {"Assistant": 6, "Associate": 9, "Professor": 15}
PROP = alloc["sample size"].to_dict()
st = stratified_estimate(stratified_sample(faculty, "rank", EQUAL, seed=11),
                         "rank", "salary", RANKS)
print("One stratified sample, 10 from each rank:\n")
print(st.round(3).to_string())
print(f"\n    plain average          {money(st.attrs['unweighted sample mean'])}")
print(f"    stratified estimate    {money(st.attrs['stratified estimate'])}"
      f"   (sum of W_h x stratum mean)")
print(f"    true mean              {money(MU)}\n")

designs = {
    "simple random, n = 30":
        lambda fr, s: simple_random_sample(fr, 30, s)["salary"].mean(),
    "stratified, proportional":
        lambda fr, s: stratified_sample(fr, "rank", PROP, s)["salary"].mean(),
    "stratified 10/10/10, plain average":
        lambda fr, s: stratified_sample(fr, "rank", EQUAL, s)["salary"].mean(),
    "stratified 10/10/10, weighted":
        lambda fr, s: stratified_estimate(stratified_sample(fr, "rank", EQUAL, s),
                                          "rank", "salary", RANKS
                                          ).attrs["stratified estimate"],
    "stratified 6/9/15, W_h per obs (wrong)":
        lambda fr, s: stratified_estimate(stratified_sample(fr, "rank", UNEQUAL, s),
                                          "rank", "salary", RANKS
                                          ).attrs["W_h per observation (wrong)"],
    "cluster, 3 of 9 campuses":
        lambda fr, s: cluster_sample(fr, "institution", 3, s)["salary"].mean(),
    "simple random, n = 72":
        lambda fr, s: simple_random_sample(fr, 72, s)["salary"].mean(),
}
print("Each design repeated 500 times:\n")
print(design_comparison(faculty, "salary", designs, reps=500).round(0).to_string())
print("""
  Bias is the average miss; the standard error is the scatter. Proportional
  stratification beats a simple random sample of the same size. The plain
  average of a 10/10/10 sample is biased low, because it over-represents the
  lower ranks; weighting each STRATUM MEAN by its population share fixes it.
  Weighting each OBSERVATION by W_h does not. The cluster sample is cheaper
  (3 trips, not 9) but less precise than a simple random sample of its size.
""")


# ===========================================================================
banner("3. SAMPLING ERROR: every possible sample from ten employees")
# ===========================================================================

AGES = [24, 33, 28, 41, 38, 29, 26, 33, 37, 31]
print(f"Ages {AGES},  mu = {np.mean(AGES):.1f}\n")
for n in (2, 5):
    e = all_sample_errors(AGES, n)
    print(f"  n = {n}: {len(e):>3} samples ({comb(10, n)} = 10C{n}); "
          f"mean of sample means = {e['sample mean'].mean():.1f}; "
          f"within 2 years: {(e['sampling error'].abs() <= 2).mean():.0%}; "
          f"errors from {e['sampling error'].min():+.1f} to "
          f"{e['sampling error'].max():+.1f}")
print("""
  Errors vary from sample to sample and average to zero. Larger samples make
  small errors more likely, but never guarantee one.
""")


# ===========================================================================
banner("4. NONSAMPLING ERROR: self-selected reviews")
# ===========================================================================

pop = np.repeat([5, 4, 3, 2, 1], [35, 30, 15, 10, 10])
probs = {5: 0.25, 4: 0.05, 3: 0.03, 2: 0.05, 1: 0.15}
print("Illustrative: happy and angry customers are likelier to post a review.\n")
print(selection_bias_demo(pop, probs, [100, 1_000, 10_000]).round(3).to_string())
print("""
  The standard error shrinks as reviews pile up. The bias does not. More
  self-selected data is a more precise estimate of the wrong number.
""")


# ===========================================================================
banner("5. EVERY POSSIBLE SAMPLE: entree prices $12, $14, $16, $18")
# ===========================================================================

for n in (1, 2, 3):
    t = enumerate_sample_means([12, 14, 16, 18], n)
    a = t.attrs
    print(f"  n = {n}: {a['samples']:>2} samples;  mean of x-bar = "
          f"{a['mean of x-bar']:.2f};  sd of x-bar = {a['sd of x-bar']:.4f};  "
          f"sigma / sqrt(n) = {a['sigma / sqrt(n)']:.4f}")
print("\nThe sampling distribution of x-bar for n = 2:\n")
print(enumerate_sample_means([12, 14, 16, 18], 2).round(4).to_string())
print("""
  mu_xbar = mu and sigma_xbar = sigma / sqrt(n), exactly. A flat population
  already gives a peaked distribution of averages at n = 2.
""")


# ===========================================================================
banner("6. A SAMPLE MEAN: divide by the standard error, not sigma")
# ===========================================================================

print("Texts per month: mu = 551, sigma = 87, a sample of 72 people.\n")
print(xbar_probability(551, 87, 72, 565, None, explain=True).to_string())
print(f"""
  Dividing by sigma instead would give P = {normal_area(551, 87, 565, None):.4f},
  which is about ONE person (and only if individuals were normal).
""")


# ===========================================================================
banner("7. WHEN IS n LARGE ENOUGH? Nevada AGI (right-skewed)")
# ===========================================================================

mu_a, sd_a = float(agi.mean()), float(agi.std(ddof=0))
print(f"AGI (tens of thousands of dollars): mean {mu_a:.3f}, sd {sd_a:.3f}, "
      f"median {agi.median():.3f}\n")
rows = {}
for n in (30, 100, 400, 1_600):
    rows[f"n = {n}"] = tail_check(simulate_sample_means(agi, n), mu_a,
                                  sd_a / sqrt(n))
print(pd.DataFrame(rows).T.round(4).to_string())
print("""
  If x-bar were normal, each tail would hold 2.5%. At n = 30 the upper tail
  holds too much and the lower too little: the sampling distribution is
  still skewed. n >= 30 is a rule of thumb, not a guarantee.
""")


# ===========================================================================
banner("8. JUDGING A CLAIM: a 50-minute commute, 42 trips averaging 45")
# ===========================================================================

print(xbar_interval(50, 14, 42).round(4).to_string(), "\n")
print(interval_table(50, 14, [42, 100, 168, 400]).round(2).to_string(), "\n")
p45 = xbar_probability(50, 14, 42, None, 45)
print(f"  P(x-bar <= 45 | mu = 50) = {p45:.4f}")
print("""
  45 lies outside the 95% interval, and a result this low would happen about
  once in a hundred samples if the claim were true: evidence against it. A
  sample mean INSIDE the interval would be "consistent with" the claim, not
  proof of it. Four times the sample halves the interval.
""")


# ===========================================================================
banner("9. FINITE POPULATIONS: the correction, and a census")
# ===========================================================================

print("100 pool-service customers, ratings sigma = 0.7:\n")
print(fpc_table(0.7, 100, [10, 40, 60, 80, 100]).round(4).to_string(), "\n")
print(xbar_probability(7.2, 0.7, 40, 7.5, None, N=100, explain=True).to_string())
print(f"""
  At n = N the corrected standard error is zero: a census has no sampling
  error. For n = 30 from N = 10,000 the correction is {fpc(30, 10_000):.4f}.
""")


# ===========================================================================
banner("10. PROPORTIONS: the Super Bowl claim, and the continuity correction")
# ===========================================================================

print("Claim: 45% of households watched. A sample of 200 finds 84.\n")
print(f"  np = 90, n(1-p) = 110;  standard error = {se_proportion(0.45, 200):.4f}\n")
print(proportion_question(0.45, 200, "at most", 84).to_string(), "\n")
print("770 graduates, claim 70% placed; 97 of 120 sampled were.\n")
print(proportion_question(0.70, 120, "at least", 97, N=770).to_string())
print("""
  For a COUNT, keep the continuity correction (0.5/n on the proportion
  scale); dropping it costs two points in the Super Bowl example. A result
  that turns up one sample in five FAILS TO CONTRADICT the claim; it does not
  support it.
""")


# ===========================================================================
banner("11. THE BOOTSTRAP: a standard error from the one sample you have")
# ===========================================================================

shoppers = np.array([1] * 58 + [0] * 42)
bi = bootstrap_interval(bootstrap(shoppers))
print(f"58 of 100 transactions by women.  bootstrap SE = "
      f"{bi['bootstrap standard error']:.4f};  formula = {sqrt(0.58 * 0.42 / 100):.4f};"
      f"  middle 95%: {bi['lower']:.3f} to {bi['upper']:.3f}\n")
one = simple_random_sample(faculty, 30, seed=21)["salary"].to_numpy()
bm = bootstrap_interval(bootstrap(one))
print(f"One sample of 30 faculty: mean {money(one.mean())};  bootstrap SE "
      f"{money(bm['bootstrap standard error'])};  true SE "
      f"{money(se_mean(float(faculty['salary'].std(ddof=0)), 30, len(faculty)))}")
print(f"  middle 95%: {money(bm['lower'])} to {money(bm['upper'])};  true mean {money(MU)}")
print("""
  The bootstrap resamples the SAMPLE, never the population. It is a way of
  analysing data, not a way of collecting it.

  QUESTION TO ANSWER IN WRITING (one paragraph, your own words):
  Section 4 shows 10,000 self-selected reviews with a tiny standard error and
  a large bias. A colleague says, "With ten thousand reviews the margin of
  error is basically zero, so the average rating is accurate." Explain what
  the standard error does and does not measure, and what you would need to
  know about how the reviews were collected before trusting the average.
""")
