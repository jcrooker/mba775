"""MBA 775 - Chapter 5: Discrete Probability Distributions

Run this and read the output. Nothing here needs editing to work.

    python 05_discrete_distributions.py

What it does, in order:

    1. A discrete distribution     the rules, the mean, the variance two ways
    2. The long run                what an expected value is an average OF
    3. Expected monetary value     RAD Construction, and a stocking decision
    4. The binomial                built from Chapter 4, then Corporate HQ
    5. Translating the question    "fewer than" is not "at most"
    6. Does the binomial fit?      Nevada's low-growth quarters, checked
    7. The Poisson                 arrivals, and changing the interval
    8. Poisson for binomial        when the shortcut works and when it fails
    9. The hypergeometric          sampling without replacement
   10. Judging a claim             the supplier, and the casino audit

Data file it reads:

    nevada_economy.csv

Everything else is a probability supplied in the problem, or is simulated with
a fixed seed.
"""

import sys
from math import comb, sqrt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                # noqa: E402
from _stats import percentile                                        # noqa: E402
from _dist import (check_distribution, moments_table, dist_mean,     # noqa: E402
                   dist_variance, dist_sd, simulate_distribution,
                   emv_table, stocking_payoffs,
                   binomial_pmf, binomial_table, binomial_mean, binomial_sd,
                   binomial_arrangements, poisson_pmf, poisson_table,
                   rescale_rate, hypergeometric_table, hypergeometric_mean,
                   hypergeometric_sd, probability_of, phrase_guide,
                   poisson_vs_binomial, binomial_vs_hypergeometric,
                   window_counts, dispersion_check)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def one_in(p):
    n = 1 / p
    return f"{float(f'{n:.2g}'):,.0f}" if n >= 100 else f"{n:,.0f}"


# ===========================================================================
banner("1. A DISCRETE DISTRIBUTION: the rules, the mean, the variance")
# ===========================================================================

DEFECTS = [0, 1, 2, 3, 4, 5]
P_DEFECTS = [0.02, 0.24, 0.36, 0.15, 0.13, 0.10]

print("Defective units in a random sample of 1,000 from the line.\n")
print(check_distribution(P_DEFECTS).to_string())
print()
table = moments_table(DEFECTS, P_DEFECTS, name="defective units")
print(table.round(4).fillna("").to_string())

mu = dist_mean(DEFECTS, P_DEFECTS)
var = dist_variance(DEFECTS, P_DEFECTS)
sd = dist_sd(DEFECTS, P_DEFECTS)
sum_x2p = float(table.loc["TOTAL", "x^2 P(x)"])
print(f"""
  Read the TOTAL row.
    mean      mu      = sum of x P(x)            = {mu:.4f}
    variance  sigma^2 = sum of (x - mu)^2 P(x)   = {var:.4f}
    shortcut  sigma^2 = sum of x^2 P(x) - mu^2   = {sum_x2p:.4f} - {mu**2:.4f} = {sum_x2p - mu**2:.4f}
    std dev   sigma   = sqrt(variance)           = {sd:.4f} defective units

  The two variance routes agree, which is the check. Report the standard
  deviation, not the variance: it is in the units of the data.
""")


# ===========================================================================
banner("2. THE LONG RUN: what an expected value is an average of")
# ===========================================================================

draws = simulate_distribution(DEFECTS, P_DEFECTS, draws=100_000)
print("Running average of defective units per sample, as samples accumulate:\n")
for k in (10, 100, 1_000, 10_000, 100_000):
    print(f"    after {k:>7,} samples: {draws.iloc[:k].mean():.4f}")
print(f"""
  No single sample can contain {mu:.2f} defective units. The expected value is
  what the average of many samples settles on. It describes the mechanism
  that generates samples, not any one sample.
""")


# ===========================================================================
banner("3. EXPECTED MONETARY VALUE: the average, and the risk around it")
# ===========================================================================

RAD_PROBS = {"early": 0.25, "on time": 0.60, "late": 0.15}
contracts = emv_table(
    {"original contract": {"early": 120_000, "on time": 100_000, "late": 90_000},
     "modified contract": {"early": 123_000, "on time": 100_000, "late": 85_000}},
    RAD_PROBS)
print("RAD Construction: two versions of the bridge contract.\n")
print(contracts.round(4).to_string())
print("""
  Same EMV. The modified contract has the larger standard deviation and the
  larger coefficient of variation, so it carries more risk for the same
  expected money. Prefer the original.
""")

DEMAND_PROBS = {"demand 40": 0.20, "demand 60": 0.40,
                "demand 80": 0.30, "demand 100": 0.10}
stock = emv_table(
    stocking_payoffs([40, 60, 80, 100], [40, 60, 80, 100],
                     price=4.00, cost=1.50, salvage=0.50),
    DEMAND_PROBS)
print("The coffee cart: how many croissants to buy each morning.\n")
print(stock.round(3).to_string())
best = stock["EMV"].idxmax()
print(f"""
  Highest EMV: {best}, at ${stock.loc[best, 'EMV']:.2f} a day. It is also far
  from the safest row. For a decision repeated daily, the swings average out
  and EMV is the right guide. For a one-time decision it is not the whole
  story.
""")


# ===========================================================================
banner("4. THE BINOMIAL: arrangements, times the probability of each")
# ===========================================================================

P_SPEND = 0.4
ways = binomial_arrangements(n=3, x=1, p=P_SPEND)
print("Three customers, each with p = 0.4 of spending $100 or more.")
print("Every way exactly ONE of them can be the big spender:\n")
print(ways.to_string())
print(f"""
  {len(ways)} mutually exclusive arrangements, each with probability
  {ways['probability'].iloc[0]:.3f}, so P(exactly 1) = {len(ways)} x {ways['probability'].iloc[0]:.3f} = {binomial_pmf(1, 3, P_SPEND):.3f}.
  That {len(ways)} is 3C1. The binomial formula is nCx arrangements times
  p^x q^(n-x) for each.
""")

N_HQ, X_HQ = 50, 5
hq = binomial_table(N_HQ, P_SPEND)
print("Corporate Headquarters: 50 customers, p = 0.4. We observed 5.\n")
print(hq.loc[:8].to_string(float_format=lambda v: f"{v:.10f}"))
p_le5 = probability_of(hq, "at most", X_HQ)
mu_hq, sd_hq = binomial_mean(N_HQ, P_SPEND), binomial_sd(N_HQ, P_SPEND)
print(f"""
  50C5 = {comb(50, 5):,}
  P(exactly 5)  = {probability_of(hq, 'exactly', X_HQ):.10f}
  P(5 or fewer) = {p_le5:.10f}   about 1 in {one_in(p_le5)}

  If p really is 0.4 we expect np = {mu_hq:.0f} big spenders, give or take
  sqrt(npq) = {sd_hq:.2f}. Five is {abs((X_HQ - mu_hq) / sd_hq):.1f} standard deviations below that.

  Either yesterday was a 1-in-{one_in(p_le5)} day, or p is not 0.4 at this
  store. That reasoning is a hypothesis test, and Chapter 9 formalises it.
  Note that we used the TAIL, "5 or fewer", not "exactly 5". With 50 trials
  even the most likely single outcome, exactly 20, has probability only
  {probability_of(hq, 'exactly', 20):.4f}.
""")


# ===========================================================================
banner("5. TRANSLATING THE QUESTION: most errors happen before the arithmetic")
# ===========================================================================

print(phrase_guide(4).to_string(index=False))
print("\nTwelve customers, p = 0.4:\n")
twelve = binomial_table(12, P_SPEND)
for phrase in ["exactly", "at most", "fewer than", "at least", "more than"]:
    print("    " + probability_of(twelve, phrase, 4, explain=True))
print(f"""
  "At least 4" is 1 minus the cumulative probability at 3, not at 4.
  Check: "at least 4" + "fewer than 4" must equal 1:
  {probability_of(twelve, 'at least', 4):.6f} + {probability_of(twelve, 'fewer than', 4):.6f} = {probability_of(twelve, 'at least', 4) + probability_of(twelve, 'fewer than', 4):.6f}
""")


# ===========================================================================
banner("6. DOES THE BINOMIAL FIT? Nevada's low-growth quarters")
# ===========================================================================

nv = pd.read_csv(find_data("nevada_economy.csv"), parse_dates=["date"])
print(f"  {len(nv)} quarters loaded, {nv['date'].min():%Y-%m} to "
      f"{nv['date'].max():%Y-%m}")
missing = int(nv["gdp_growth"].isna().sum())
print(f"  {missing} have no four-quarter growth (the first four cannot); "
      f"they are set aside, not filled.\n")
nv_ok = nv.dropna(subset=["gdp_growth"]).copy()

low_cut = percentile(nv_ok["gdp_growth"], 0.25)
nv_ok["low"] = nv_ok["gdp_growth"] < low_cut
p_low = float(nv_ok["low"].mean())
per_year = window_counts(nv_ok["low"], window=4, labels=nv_ok["date"].dt.year)

print(f"  'Low' = growth below the 25th percentile ({100 * low_cut:.2f}%), "
      f"so p = {p_low:.2f} by construction.")
print(f"  Low-growth quarters in each of {len(per_year)} complete years "
      f"({per_year.attrs['observations dropped']} quarters left over):\n")
print(per_year.to_frame().T.to_string())
print()
check = dispersion_check(per_year, n=4, p=p_low)
print(check.round(3).to_string())
a = check.attrs
print(f"""
  mean:     observed {a['observed mean']:.2f}   binomial np  = {a['binomial mean']:.2f}
  variance: observed {a['observed variance']:.2f}   binomial npq = {a['binomial variance']:.2f}   ratio {a['variance ratio']:.1f}

  The averages match and nothing else does. The binomial expects
  {check.loc[4, 'windows expected']:.2f} years with all four quarters Low; Nevada had {int(check.loc[4, 'windows observed'])}.
  Bad quarters arrive in clumps, so the trials are not independent and the
  binomial understates how often the extreme years occur.

  Part of the clumping is built into the data: four-quarter growth rates of
  neighbouring quarters share three quarters of their data. The rest is the
  economy. Either way, characteristic 4 fails.
""")


# ===========================================================================
banner("7. THE POISSON: a count with no trials")
# ===========================================================================

LAM = 4
extra = poisson_table(LAM)
print("Corporate HQ expects 4 customers in an extra hour of opening.\n")
print(extra.loc[:12].round(5).to_string())
print(f"""
  {probability_of(extra, 'more than', 8, explain=True)}

  A Poisson count has no upper limit, so "more than 8" has to be done with
  the complement rule. The sum being subtracted starts at x = 0.
  Mean = variance = {LAM}, so the standard deviation is {sqrt(LAM):.0f} customers.
""")

print("Changing the interval. The store averages 4 customers an hour.\n")
for minutes, x in ((60, 4), (30, 2), (15, 0)):
    lam = rescale_rate(4, per=60, to=minutes)
    print(f"    next {minutes:>2} minutes: lambda = {lam:<4g} "
          f"P(exactly {x}) = {poisson_pmf(x, lam):.4f}")
print("""
  "Exactly 4 in an hour" and "exactly 2 in half an hour" are different
  probabilities. You rescale the MEAN to the new interval. You cannot rescale
  the probability.
""")


# ===========================================================================
banner("8. POISSON FOR BINOMIAL: n >= 20 and p <= 0.05")
# ===========================================================================

good = poisson_vs_binomial(n=40, p=0.03)
bad = poisson_vs_binomial(n=40, p=0.40, max_x=30)
print(f"40 orders, 3% mis-packed. lambda = np = {good.attrs['lambda']:.2f}. "
      f"Rule satisfied: {good.attrs['rule satisfied']}\n")
print(good.round(4).to_string())
print(f"""
  Largest gap with p = 0.03: {good['difference'].abs().max():.4f}
  Largest gap with p = 0.40: {bad['difference'].abs().max():.4f}   (rule satisfied: {bad.attrs['rule satisfied']})

  Software computes the binomial exactly, so the shortcut no longer saves
  work. What it still does is explain why counts of rare events are Poisson:
  a huge unknown n and a tiny unknown p, of which only the product np matters.
""")


# ===========================================================================
banner("9. THE HYPERGEOMETRIC: sampling without replacement")
# ===========================================================================

N_INV, R_BAD, n_pull = 32, 9, 5
print("The Chapter 4 auditor pulls 5 of 32 invoices; 9 contain an error.\n")
print(binomial_vs_hypergeometric(n_pull, R_BAD, N_INV).round(4).to_string())
fpc = sqrt((N_INV - n_pull) / (N_INV - 1))
print(f"""
  mean, both columns:        nR/N = np      = {hypergeometric_mean(n_pull, R_BAD, N_INV):.3f}
  sd, hypergeometric:                         {hypergeometric_sd(n_pull, R_BAD, N_INV):.3f}
  sd, binomial sqrt(npq):                     {binomial_sd(n_pull, R_BAD / N_INV):.3f}
  ratio = sqrt((N - n)/(N - 1)):              {fpc:.3f}

  Sampling without replacement is less variable. That ratio is the finite
  population correction of Chapter 7, and it goes to 1 as N grows.
""")

print("Store managers: choose 3 of 12, of whom 2 are inexperienced.\n")
managers = hypergeometric_table(n=3, R=2, N=12)
print(managers.round(4).to_string())
print(f"\n  P(none inexperienced) = 10C3 x 2C0 / 12C3 = "
      f"{comb(10, 3)} x {comb(2, 0)} / {comb(12, 3)} = "
      f"{managers.loc[0, 'P(X = x)']:.4f}\n")


# ===========================================================================
banner("10. JUDGING A CLAIM: how surprising is what we saw?")
# ===========================================================================

supplier = binomial_table(20, 0.05)
p_ge4 = probability_of(supplier, "at least", 4)
print("A supplier claims at most 5% defective. We test 20 and find 4.\n")
print(supplier.loc[:6].round(5).to_string())
print(f"""
  expected if the claim holds: np = {binomial_mean(20, 0.05):.2f}, sd = {binomial_sd(20, 0.05):.3f}
  {probability_of(supplier, 'at least', 4, explain=True)}

  A sample this bad would come from an honest supplier about once in
  {one_in(p_ge4)} shipments. Strong evidence against the claim; not proof.
""")

casino = hypergeometric_table(n=6, R=8, N=25)
p_ge5 = probability_of(casino, "at least", 5)
print("A casino audits 6 of 25 cabinets 'at random'. 8 belong to one vendor,")
print("and 5 of the 6 audited were that vendor's.\n")
print(casino.round(5).to_string())
print(f"""
  expected under random selection: nR/N = {hypergeometric_mean(6, 8, 25):.2f}, sd = {hypergeometric_sd(6, 8, 25):.2f}
  {probability_of(casino, 'at least', 5, explain=True)}

  About 1 in {one_in(p_ge5)}. Either an unusual draw, or the selection was not
  random.

  QUESTION TO ANSWER IN WRITING (one paragraph, your own words):
  In both cases we computed the probability of a result "at least as extreme"
  as the one observed, not the probability of exactly that result. Using the
  Corporate HQ numbers from section 4, explain to a colleague why "exactly"
  would have been the wrong probability to report.
""")
