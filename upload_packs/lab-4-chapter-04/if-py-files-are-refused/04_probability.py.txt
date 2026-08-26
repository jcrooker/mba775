"""MBA 775 - Chapter 4: Introducing Probabilities

Run this and read the output. Nothing here needs editing to work.

    python 04_probability.py

What it does, in order:

    1. Classical probability      the two-dice sample space
    2. Empirical probability      states of the Nevada economy
    3. The law of large numbers   a million simulated rolls
    4. Two events at once         contingency and joint probability tables
    5. Conditional probability    and why P(A|B) is not P(B|A)
    6. Independence               dice memory, and Nevada's
    7. Bayes' Theorem             the four-column table, checked two ways
    8. Counting                   permutations and combinations
    9. Applications               the drug trial, the promotion, Madison

Data file it reads:

    nevada_economy.csv

Everything else in this script is either simulated with a fixed seed or is a
probability supplied in the problem, which is how business probability
problems usually arrive.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _course import find_data, banner                                # noqa: E402
from _stats import percentile                                        # noqa: E402
from _prob import (two_dice_sample_space, roll_two_dice,             # noqa: E402
                   empirical_probability, joint_probability_table,
                   marginal, joint, conditional, addition_rule,
                   multiplication_rule, independence_report,
                   bayes_table, expected_value, fundamental_counting,
                   permutations, combinations, counting_comparison)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 20)

LEAD = 2                 # quarters ahead: "six months"
TODAY = "Low,Middle"     # low growth, middle unemployment, right now
FUTURE = "Low,High"      # low growth, high unemployment, in six months


# ===========================================================================
banner("1. CLASSICAL PROBABILITY: get the sample space right")
# ===========================================================================

space = two_dice_sample_space()
print(space[["ways", "fraction", "probability"]].round(4).to_string())

p7 = float(space.loc[7, "probability"])
print(f"""
  The tempting error is to count the 11 possible totals and say P(7) = 1/11
  = {1/11:.4f}. That is wrong: those 11 totals are not equally likely, and
  classical probability requires that they be.

  The equally likely outcomes are the 36 ordered pairs of the two dice.
  {int(space.loc[7,'ways'])} of them sum to 7, so P(7) = {int(space.loc[7,'ways'])}/36 = {p7:.4f}.

  When a probability question feels hard, it is almost always because the
  sample space is not what it first appeared to be.
""")


# ===========================================================================
banner("2. EMPIRICAL PROBABILITY: the state of the Nevada economy")
# ===========================================================================

nv = pd.read_csv(find_data("nevada_economy.csv"), parse_dates=["date"])
nv_ok = nv.dropna(subset=["gdp_growth"]).copy()


def quarter(d):
    d = pd.Timestamp(d)
    return f"{d.year} Q{(d.month - 1) // 3 + 1}"


print(f"  {quarter(nv['date'].min())} to {quarter(nv['date'].max())}: "
      f"{len(nv):,} quarters, {len(nv_ok):,} with four-quarter growth")


def classify(series):
    lo, hi = percentile(series, 0.25), percentile(series, 0.75)
    return np.where(series < lo, "Low",
                    np.where(series > hi, "High", "Middle")), lo, hi


growth_state, g_lo, g_hi = classify(nv_ok["gdp_growth"])
unemp_state, u_lo, u_hi = classify(nv_ok["unemployment_rate"])

print(f"\n  Growth:       Low below {100*g_lo:6.2f}%   High above {100*g_hi:6.2f}%")
print(f"  Unemployment: Low below {u_lo:6.2f}%    High above {u_hi:6.2f}%")

states = pd.DataFrame({"date": nv_ok["date"].to_numpy(),
                       "growth": growth_state,
                       "unemployment": unemp_state})
states["today"] = states["growth"] + "," + states["unemployment"]

freq = states["today"].value_counts()
print("\n  Empirical probability of each combined state:\n")
print(pd.DataFrame({"quarters": freq,
                    "probability": (freq / len(states)).round(4)}).to_string())

print("""
  Careful here. "Low growth" is a definition we chose -- the bottom quartile
  -- not a fact about Nevada. A different cut point gives different
  probabilities from identical data. Say so when you report one.
""")


# ===========================================================================
banner("3. THE LAW OF LARGE NUMBERS")
# ===========================================================================

rolls = roll_two_dice(1_000_000)
observed = rolls.value_counts().sort_index()

lln = space[["probability"]].copy()
lln.columns = ["classical"]
lln["empirical"] = observed / len(rolls)
lln["difference"] = lln["empirical"] - lln["classical"]
print(lln.round(5).to_string())

running = np.cumsum(rolls == 7) / np.arange(1, len(rolls) + 1)
print("\n  Running empirical probability of a 7:\n")
for k in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
    print(f"    after {k:>9,} rolls:  {running[k-1]:.5f}")
print(f"    classical value:      {p7:.5f}")
print(f"""
  Convergence is real and it is slow. At ten rolls the running probability is
  nowhere near 1/6; at a hundred it is still visibly wrong. "We ran a pilot
  with 40 customers" is the business equivalent of the left end of that list.
""")


# ===========================================================================
banner("4. TWO EVENTS AT ONCE: the contingency table")
# ===========================================================================

states["forward"] = states["today"].shift(-LEAD)
paired = states.dropna(subset=["forward"]).copy()
n_pairs = len(paired)

counts = pd.crosstab(paired["today"], paired["forward"],
                     margins=True, margins_name="Total")
print(f"  Counts ({n_pairs} quarters have a six-month-forward state):\n")
print(counts.to_string())

probs = joint_probability_table(paired["today"], paired["forward"],
                                x_name="today", y_name="in six months")
print("\n  The same table as probabilities:\n")
print(probs.round(4).to_string())
print(f"\n  Bottom-right corner: {probs.loc['Total','Total']:.4f} "
      "-- it must be 1.00, and that is your check.")

p_today = marginal(probs, row=TODAY)
p_future = marginal(probs, col=FUTURE)
p_both = joint(probs, TODAY, FUTURE)
n_today = int(counts.loc[TODAY, "Total"])
n_future = int(counts.loc["Total", FUTURE])
n_both = int(counts.loc[TODAY, FUTURE])

banner("4b. The addition rule")

p_or = addition_rule(p_today, p_future, p_both)
print(f"  A = today is {TODAY:<12} {n_today:>3} of {n_pairs}  P(A) = {p_today:.4f}")
print(f"  B = in 6mo is {FUTURE:<11} {n_future:>3} of {n_pairs}  P(B) = {p_future:.4f}")
print(f"  Both                       {n_both:>3} of {n_pairs}  P(A and B) = {p_both:.4f}")
print(f"\n  P(A or B) = {p_today:.4f} + {p_future:.4f} - {p_both:.4f} = {p_or:.4f}")
print(f"  Forget to subtract the overlap and you report {p_today + p_future:.4f} instead.")


# ===========================================================================
banner("5. CONDITIONAL PROBABILITY: the denominator is the whole story")
# ===========================================================================

p_future_given_today = conditional(probs, TODAY, FUTURE, given="row")
p_today_given_future = conditional(probs, TODAY, FUTURE, given="col")

print(f"  P(B|A) = {n_both}/{n_today} = {p_future_given_today:.4f}")
print(f"           'Nevada looks like {TODAY} right now. How worried should")
print(f"            I be about six months from now?'   -- the FORECASTING question")
print()
print(f"  P(A|B) = {n_both}/{n_future} = {p_today_given_future:.4f}")
print(f"           'Nevada is in a bad state. How often did it look like this")
print(f"            six months earlier?'               -- the DIAGNOSTIC question")
print("""
  Same numerator. Different denominator. Different question. Confusing these
  two is the most consequential error in this chapter -- it is the error
  behind most misread medical tests and most overstated marketing results.
""")


# ===========================================================================
banner("6. INDEPENDENCE: compare the prior to the posterior")
# ===========================================================================

sevens = (rolls == 7).to_numpy()
after_seven = sevens[1:][sevens[:-1]]
print("  Do dice have memory?")
print(f"    rolls that were a 7:              {sevens.sum():>9,}")
print(f"    of those, followed by another 7:  {after_seven.sum():>9,}")
print("   ", independence_report(sevens.mean(), after_seven.mean(),
                                 "roll a 7", "last roll was a 7"))

print("\n  Does Nevada's economy have memory?")
print("   ", independence_report(p_future, p_future_given_today,
                                 f"'{FUTURE}' in six months", f"'{TODAY}' today"))
print("""
  Independence is assumed silently and constantly. "Each component is 99%
  reliable, so the system is 0.99^10" assumes the failures are independent,
  and components sharing a power supply are not. When a risk model surprises
  everyone, a false independence assumption is the usual culprit.
""")

banner("6b. The multiplication rule")

print(pd.DataFrame({
    "method": ["Multiplication rule: P(A) x P(B|A)",
               "Read directly from the table",
               "If they were independent: P(A) x P(B)"],
    "P(A and B)": [multiplication_rule(p_today, p_future_given_today),
                   p_both, p_today * p_future],
}).round(6).to_string(index=False))
print("\n  The first two agree by construction. The gap to the third measures")
print("  how dependent these events actually are.")


# ===========================================================================
banner("7. BAYES' THEOREM")
# ===========================================================================

p_today_given_not_future = ((p_today - p_both) / (1 - p_future))

bayes = bayes_table(
    priors={FUTURE: p_future, f"not {FUTURE}": 1 - p_future},
    likelihoods={FUTURE: p_today_given_future,
                 f"not {FUTURE}": p_today_given_not_future},
    evidence_name=f"today={TODAY}")
print(bayes.round(4).to_string())

post_col = [c for c in bayes.columns if c.startswith("posterior")][0]
joint_col = [c for c in bayes.columns if c.startswith("joint")][0]
bayes_answer = float(bayes.loc[FUTURE, post_col])

print(f"""
  Read straight off the contingency table:  P(B|A) = {p_future_given_today:.4f}
  Computed by Bayes' Theorem:               P(B|A) = {bayes_answer:.4f}

  They agree. That is the point of the exercise. Bayes is not a different
  answer -- it is a route to the same answer when you do NOT have the table,
  which is the situation in every applied problem below.

  The joint column total, {float(bayes.loc['TOTAL', joint_col]):.4f}, is P(A) -- the denominator of the
  formula. It matches the marginal {p_today:.4f} from the original table.
""")


# ===========================================================================
banner("8. COUNTING")
# ===========================================================================

print(f"  Four regions send one manager each, from 12, 6, 8 and 7 managers:")
print(f"    12 x 6 x 8 x 7 = {fundamental_counting(12, 6, 8, 7):,}")
print(f"\n  Twelve managers elect a Senior and an Associate (order MATTERS):")
print(f"    12P2 = 12!/10! = {permutations(12, 2)}")
print(f"\n  Seven managers send two to Hawaii (order does NOT matter):")
print(f"    7C2 = 7!/(5!2!) = {combinations(7, 2)}")
print()
print(counting_comparison(12, 2).to_string(index=False))
print("""
  One question decides it: would swapping two of the chosen items give a
  different outcome? Senior and Associate, yes -- permutation. Two people on
  the same flight to Hawaii, no -- combination. The permutation is always
  larger, by exactly x!.
""")


# ===========================================================================
banner("9. APPLICATIONS: when there is no data set")
# ===========================================================================

print("9a. The drug trial\n")
drug = bayes_table(priors={"Effective": 0.8, "Not effective": 0.2},
                   likelihoods={"Effective": 0.9, "Not effective": 0.02},
                   evidence_name="Pass")
print(drug.round(4).to_string())
d_post = [c for c in drug.columns if c.startswith("posterior")][0]
d_joint = [c for c in drug.columns if c.startswith("joint")][0]
print(f"""
  P(Pass) = {float(drug.loc['TOTAL', d_joint]):.4f}   (the joint column total)
  P(Effective | Pass) = {float(drug.loc['Effective', d_post]):.4f}

  The belief moved from a prior of 80% to a posterior of {100*float(drug.loc['Effective', d_post]):.2f}%. Notice
  which input did that work: not the 0.9, but the 0.02. A test is informative
  when it is unlikely to fire under the WRONG hypothesis, not merely when it
  is likely to fire under the right one.
""")

print("9b. The sportsbook promotion\n")
promo = bayes_table(priors={"Promotion": 0.1, "No promotion": 0.9},
                    likelihoods={"Promotion": 0.08, "No promotion": 0.05},
                    evidence_name="Adoption")
print(promo.round(4).to_string())
p_post = [c for c in promo.columns if c.startswith("posterior")][0]
p_joint = [c for c in promo.columns if c.startswith("joint")][0]
p_adopt = float(promo.loc["TOTAL", p_joint])
p_promo_given_adopt = float(promo.loc["Promotion", p_post])

REVENUE = 505.44
e_no = expected_value({REVENUE: 0.05, 0.0: 0.95})
e_yes = expected_value({REVENUE: 0.08, 0.0: 0.92})
print(f"""
  P(Adoption) = {p_adopt:.4f}

  Expected revenue per user, no promotion:  ${e_no:>7.2f}
  Expected revenue per user, promotion:     ${e_yes:>7.2f}
  Difference:                               ${e_yes - e_no:>7.2f}

  That difference is a CEILING on what the promotion is worth, not a verdict.
  Nothing here says what it cost, and nothing here establishes that the
  promotion caused the higher adoption rather than being placed where
  adoption was going to happen anyway.

  P(Promotion | Adoption) = {p_promo_given_adopt:.4f}

  Sit with that. Promoted regions have the higher adoption RATE, yet {100*(1-p_promo_given_adopt):.1f}% of
  adopters came from regions with no promotion at all -- because only 10% of
  regions were promoted. A high conditional probability in a small group can
  still account for very little of the total.
""")

print("9c. Madison's breakfast menu\n")
PRIORS = {"HS": 0.10, "MS": 0.20, "BE": 0.40, "MU": 0.20, "HU": 0.10}
LIKELIHOODS = {"HS": 0.95, "MS": 0.60, "BE": 0.30, "MU": 0.20, "HU": 0.10}
madison = bayes_table(PRIORS, LIKELIHOODS, evidence_name="FR")
print(madison.round(4).to_string())
m_post = [c for c in madison.columns if c.startswith("posterior")][0]
m_joint = [c for c in madison.columns if c.startswith("joint")][0]
p_hs_given_fr = float(madison.loc["HS", m_post])
print(f"""
  P(Favorable Review) = {float(madison.loc['TOTAL', m_joint]):.4f}
""")
print("   ", independence_report(PRIORS["HS"], p_hs_given_fr,
                                 "Highly Successful", "Favorable Review"))
post = madison[m_post].drop("TOTAL")
leaders = post[np.isclose(post, post.max())].index.tolist()
rank = int((post > post["HS"]).sum()) + 1
print(f"""
  But read the whole posterior column, not just the top row. Even WITH a
  favorable review, Highly Successful is only the {rank}rd most likely outcome;
  {" and ".join(leaders)} {"remains" if len(leaders) == 1 else "remain"} more probable.

  Good news raises a probability. It rarely settles the question.
""")

banner("Done")
print("Every number above came from nevada_economy.csv, from a fixed-seed")
print("simulation, or from a probability stated in the problem. If a figure")
print("in your write-up is not in this output, say where it came from.")
