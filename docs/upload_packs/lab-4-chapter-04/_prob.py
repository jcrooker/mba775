"""Probability helpers for MBA 775, Chapter 4.

Each function here matches a definition in the chapter, and each one shows
its work. Probability is the part of this course where a right answer arrived
at by the wrong route is most likely to go unnoticed, because every answer is
a number between 0 and 1 and they all look equally plausible. So the functions
return tables with the intermediate quantities in them, not bare numbers.

The most important one is `bayes_table`. Nearly every "revise a belief in
light of evidence" problem in business is that table, and once you can read
it you do not have to remember the formula.

You do not need to read this file to do the coursework.
"""

from __future__ import annotations

from math import comb, factorial, perm

import numpy as np
import pandas as pd

__all__ = [
    "classical_probability", "empirical_probability",
    "two_dice_sample_space", "roll_two_dice",
    "joint_probability_table", "marginal", "joint", "conditional",
    "addition_rule", "multiplication_rule",
    "independence_report", "bayes_table", "natural_frequency_table",
    "expected_value",
    "sequential_draws", "replacement_comparison",
    "fundamental_counting", "permutations", "combinations",
    "counting_comparison",
]


# ---------------------------------------------------------------------------
# The three kinds of probability
# ---------------------------------------------------------------------------

def classical_probability(favourable: int, total: int) -> float:
    """Outcomes that make the event happen, over outcomes in the sample space.

    Only valid when every outcome in the sample space is equally likely. That
    condition is the whole reason dice and cards appear in every textbook:
    they are the rare cases where you can be sure of it.
    """
    if total <= 0:
        raise ValueError("The sample space cannot be empty.")
    if not 0 <= favourable <= total:
        raise ValueError(f"{favourable} favourable outcomes out of {total} "
                         "is not possible.")
    return favourable / total


def empirical_probability(occurrences: int, observations: int) -> float:
    """How often the event actually happened, over how many chances it had."""
    if observations <= 0:
        raise ValueError("There must be at least one observation.")
    return occurrences / observations


# ---------------------------------------------------------------------------
# Dice: the worked example that makes classical probability concrete
# ---------------------------------------------------------------------------

def two_dice_sample_space() -> pd.DataFrame:
    """Every way two six-sided dice can sum to each total, with the classical
    probability of each.

    The table is built by enumerating all 36 ordered pairs rather than by
    typing in the counts, so the "number of ways" column is derived rather
    than remembered.
    """
    pairs = [(a, b) for a in range(1, 7) for b in range(1, 7)]
    sums = pd.Series([a + b for a, b in pairs])
    ways = sums.value_counts().sort_index()
    return pd.DataFrame({
        "outcome": ways.index,
        "ways": ways.to_numpy(),
        "probability": (ways / len(pairs)).to_numpy(),
        "fraction": [f"{w}/{len(pairs)}" for w in ways.to_numpy()],
    }).set_index("outcome")


def roll_two_dice(trials: int = 1_000_000, seed: int = 20220918) -> pd.Series:
    """Simulate `trials` rolls of two dice and return the sums.

    The seed is fixed so the note, the student script, and your classmate all
    produce the same numbers. A simulation whose result changes every run is
    not something two people can discuss.
    """
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(1, 7, trials) + rng.integers(1, 7, trials))


# ---------------------------------------------------------------------------
# Two events at once
# ---------------------------------------------------------------------------

def joint_probability_table(x, y, x_name="x", y_name="y",
                            as_probability=True) -> pd.DataFrame:
    """A contingency table with row and column totals.

    With `as_probability=True` every cell is a joint probability, the margins
    are marginal probabilities, and the bottom-right corner is 1.00 — which is
    the fastest check that you built the table correctly.
    """
    table = pd.crosstab(pd.Series(list(x), name=x_name),
                        pd.Series(list(y), name=y_name),
                        margins=True, margins_name="Total")
    if as_probability:
        total = table.loc["Total", "Total"]
        table = table / total
    return table


def marginal(table: pd.DataFrame, *, row=None, col=None) -> float:
    """P(A) — read from the margin of a probability table."""
    if (row is None) == (col is None):
        raise ValueError("Give exactly one of row= or col=.")
    return float(table.loc[row, "Total"] if row is not None
                 else table.loc["Total", col])


def joint(table: pd.DataFrame, row, col) -> float:
    """P(A and B) — read from the body of a probability table."""
    return float(table.loc[row, col])


def conditional(table: pd.DataFrame, row, col, *, given="col") -> float:
    """P(row | col) when given="col", or P(col | row) when given="row".

        P(A | B) = P(A and B) / P(B)

    The conditioning event goes in the denominator. Which of the two events
    that is happens to be the single most common mistake in this chapter, so
    the argument is named rather than positional.
    """
    both = joint(table, row, col)
    if given == "col":
        base = marginal(table, col=col)
    elif given == "row":
        base = marginal(table, row=row)
    else:
        raise ValueError('given must be "row" or "col".')
    if base == 0:
        return float("nan")
    return both / base


def addition_rule(p_a: float, p_b: float, p_a_and_b: float = 0.0) -> float:
    """P(A or B) = P(A) + P(B) - P(A and B).

    Leave `p_a_and_b` at zero only when the events are mutually exclusive.
    Forgetting to subtract it is how people end up reporting probabilities
    above 1.
    """
    result = p_a + p_b - p_a_and_b
    if result > 1 + 1e-9:
        raise ValueError(
            f"P(A or B) came out to {result:.4f}, which is above 1. "
            "Either the inputs are wrong, or the events overlap and "
            "p_a_and_b was left at zero.")
    return result


def multiplication_rule(p_b: float, p_a_given_b: float) -> float:
    """P(A and B) = P(B) * P(A|B).

    When A and B are independent, P(A|B) is just P(A) and this collapses to
    the familiar P(A)P(B).
    """
    return p_b * p_a_given_b


def sequential_draws(successes: int, total: int, draws: int = 2,
                     replace: bool = False) -> pd.DataFrame:
    """The probability that every one of `draws` items drawn from a group of
    `total`, of which `successes` have the property, has the property.

    Shown one draw at a time, because that is where the with/without
    replacement distinction lives. Without replacement, each draw shrinks the
    group by one and (if the last draw was a success) the successes by one, so
    the conditional probability changes at every step. With replacement, the
    group is restored each time, so every draw is the same independent event.

    Returns one row per draw with the pool it was drawn from, the conditional
    probability of a success at that draw, and the running product — which is
    the multiplication rule applied step by step.
    """
    if not 0 <= successes <= total:
        raise ValueError(f"{successes} successes out of {total} is not possible.")
    if draws < 1:
        raise ValueError("Draw at least once.")
    if not replace and draws > successes:
        # The probability is zero, but the table still shows why.
        pass
    rows, running = [], 1.0
    s, n = successes, total
    for k in range(1, draws + 1):
        p = s / n if n > 0 else 0.0
        running *= p
        rows.append({"draw": k, "pool": n, "successes in pool": s,
                     "P(success | earlier draws)": p,
                     "running product": running})
        if not replace:
            s, n = max(s - 1, 0), n - 1
    return pd.DataFrame(rows).set_index("draw")


def replacement_comparison(successes: int, total: int,
                           draws: int = 2) -> pd.DataFrame:
    """P(all draws are successes), with and without replacement, side by side.

    With replacement is the independent case, P(A)^draws. Without replacement
    is the dependent case. The gap between them shrinks as `total` grows
    relative to `draws`, which is why sampling a few hundred customers from
    a population of millions can be treated as independent draws — the
    idea behind the "5% rule" in Chapter 7.
    """
    with_r = sequential_draws(successes, total, draws, replace=True)
    without = sequential_draws(successes, total, draws, replace=False)
    return pd.DataFrame({
        "sampling": ["With replacement (independent draws)",
                     "Without replacement (dependent draws)"],
        f"P(all {draws} have the property)": [
            float(with_r["running product"].iloc[-1]),
            float(without["running product"].iloc[-1])],
    })


def independence_report(prior: float, posterior: float,
                        label_event="A", label_given="B") -> str:
    """Compare P(A) against P(A|B) and say what the comparison implies.

    The chapter's test for independence is exactly this comparison. Equal
    means the evidence told you nothing; different means it did.
    """
    gap = posterior - prior
    if prior == 0:
        return "P(A) is zero; the comparison is not informative."
    relative = 100 * gap / prior
    if abs(relative) < 1:
        verdict = ("indistinguishable — knowing "
                   f"{label_given} tells you essentially nothing about "
                   f"{label_event}")
    elif gap > 0:
        verdict = (f"higher — {label_given} makes {label_event} more likely")
    else:
        verdict = (f"lower — {label_given} makes {label_event} less likely")
    return (f"P({label_event}) = {prior:.6g};  "
            f"P({label_event}|{label_given}) = {posterior:.6g};  "
            f"change {gap:+.6g} ({relative:+.2f}%) — {verdict}.")


# ---------------------------------------------------------------------------
# Bayes
# ---------------------------------------------------------------------------

def bayes_table(priors: dict, likelihoods: dict,
                evidence_name="B") -> pd.DataFrame:
    """Revise a set of prior beliefs in light of one piece of evidence.

    `priors`      P(A_i) for every state of the world, summing to 1.
    `likelihoods` P(B|A_i) — how likely the evidence is under each state.

    Returns a table with one row per state and these columns:

        prior          P(A_i)
        likelihood     P(B|A_i)
        joint          P(A_i and B)  =  P(A_i) * P(B|A_i)
        posterior      P(A_i|B)      =  joint / sum of joints

    The sum of the `joint` column is P(B), the denominator of Bayes' Theorem.
    Reading that column total is usually easier than remembering the formula,
    and it is the same number either way.
    """
    states = list(priors)
    missing = [s for s in states if s not in likelihoods]
    if missing:
        raise ValueError(f"No likelihood given for: {', '.join(map(str, missing))}")

    prior = np.array([priors[s] for s in states], dtype="float64")
    if not np.isclose(prior.sum(), 1.0):
        raise ValueError(
            f"The priors sum to {prior.sum():.6g}, not 1. Every state of the "
            "world has to be listed, and they cannot overlap.")

    like = np.array([likelihoods[s] for s in states], dtype="float64")
    joint_p = prior * like
    evidence = joint_p.sum()
    if evidence == 0:
        raise ValueError("The evidence has probability zero under every "
                         "state, so there is nothing to revise.")

    table = pd.DataFrame({
        "prior": prior,
        f"likelihood P({evidence_name}|state)": like,
        f"joint P(state and {evidence_name})": joint_p,
        f"posterior P(state|{evidence_name})": joint_p / evidence,
    }, index=pd.Index(states, name="state"))

    total = pd.DataFrame({
        "prior": [prior.sum()],
        f"likelihood P({evidence_name}|state)": [np.nan],
        f"joint P(state and {evidence_name})": [evidence],
        f"posterior P(state|{evidence_name})": [1.0],
    }, index=pd.Index(["TOTAL"], name="state"))

    return pd.concat([table, total])


def natural_frequency_table(prior: float, sensitivity: float,
                            false_positive_rate: float, n: int = 10_000,
                            condition="Condition", evidence="Positive") -> pd.DataFrame:
    """Bayes for a two-state problem, as a head count instead of a formula.

    Start with `n` people (or emails, or loans). `prior` of them have the
    condition. The test fires on `sensitivity` of those and, wrongly, on
    `false_positive_rate` of the rest. The posterior P(condition | positive)
    is then just: positives with the condition, over all positives.

    Counts are left unrounded so the table agrees exactly with `bayes_table`;
    round them for display if you want whole people.
    """
    for name, v in [("prior", prior), ("sensitivity", sensitivity),
                    ("false_positive_rate", false_positive_rate)]:
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    with_c = n * prior
    without = n - with_c
    pos_with, pos_without = with_c * sensitivity, without * false_positive_rate
    table = pd.DataFrame({
        evidence: [pos_with, pos_without],
        f"Not {evidence.lower()}": [with_c - pos_with, without - pos_without],
    }, index=pd.Index([condition, f"No {condition.lower()}"], name=f"of {n:,}"))
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum()
    return table


def expected_value(values: dict) -> float:
    """Expected value from a {outcome_value: probability} mapping.

    Included here because the payoff questions in this chapter are one
    multiplication away from the probabilities, and because it is the bridge
    to Chapter 5.
    """
    total_p = sum(values.values())
    if not np.isclose(total_p, 1.0):
        raise ValueError(f"The probabilities sum to {total_p:.6g}, not 1.")
    return float(sum(v * p for v, p in values.items()))


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def fundamental_counting(*choices: int) -> int:
    """Multiply the number of choices at each stage.

    fundamental_counting(12, 6, 8, 7) is 12 * 6 * 8 * 7.
    """
    if not choices:
        raise ValueError("Give at least one stage.")
    total = 1
    for k in choices:
        if k < 1:
            raise ValueError("Every stage needs at least one choice.")
        total *= int(k)
    return total


def permutations(n: int, x: int) -> int:
    """nPx = n! / (n-x)! — arrangements of x objects from n, order matters."""
    if not 0 <= x <= n:
        raise ValueError(f"Cannot select {x} from {n}.")
    return perm(int(n), int(x))


def combinations(n: int, x: int) -> int:
    """nCx = n! / ((n-x)! x!) — selections of x from n, order does not."""
    if not 0 <= x <= n:
        raise ValueError(f"Cannot select {x} from {n}.")
    return comb(int(n), int(x))


def counting_comparison(n: int, x: int) -> pd.DataFrame:
    """Permutations and combinations side by side, with the ratio.

    The ratio is x!, which is the number of orderings of the chosen group —
    exactly what the combination throws away and the permutation keeps.
    """
    p, c = permutations(n, x), combinations(n, x)
    return pd.DataFrame({
        "measure": [f"Permutations  {n}P{x}  (order matters)",
                    f"Combinations  {n}C{x}  (order does not)",
                    f"Ratio = {x}!"],
        "value": [p, c, factorial(int(x))],
    })
