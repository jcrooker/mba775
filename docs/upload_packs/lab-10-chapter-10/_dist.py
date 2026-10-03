"""Discrete probability distribution helpers for MBA 775, Chapter 5.

Chapter 4 worked with the probability of one event at a time. This chapter
works with the whole list at once: every value a count can take, and the
probability of each. That list is a probability distribution, and it is the
first thing in this course that deserves to be called a model.

Each function here matches a definition in the chapter, and each one shows its
work. The tables come back with the intermediate columns in them, because in
this chapter the most common mistake is not the arithmetic. It is answering
"fewer than 4" with the number for "4 or fewer", and a bare number cannot show
you that you did it.

Nothing here needs scipy. The formulas are short enough to write out, and
writing them out is the point.

You do not need to read this file to do the coursework.
"""

from __future__ import annotations

from itertools import combinations as _subsets
from math import comb, exp, factorial, sqrt

import numpy as np
import pandas as pd

__all__ = [
    "check_distribution", "distribution_from_counts", "moments_table",
    "dist_mean", "dist_variance", "dist_sd", "simulate_distribution",
    "emv_table", "stocking_payoffs",
    "binomial_pmf", "binomial_table", "binomial_mean", "binomial_sd",
    "binomial_arrangements",
    "poisson_pmf", "poisson_table", "rescale_rate",
    "hypergeometric_pmf", "hypergeometric_table",
    "hypergeometric_mean", "hypergeometric_sd",
    "probability_of", "phrase_guide",
    "poisson_vs_binomial", "binomial_vs_hypergeometric",
    "window_counts", "dispersion_check",
]


# ---------------------------------------------------------------------------
# Any discrete distribution: the rules, the mean, the variance
# ---------------------------------------------------------------------------

def check_distribution(probabilities, tolerance: float = 0.0015) -> pd.Series:
    """Test a column of numbers against the rules for a discrete probability
    distribution, and say which rule fails.

    The outcomes of a distribution are mutually exclusive by construction (a
    count cannot be both 2 and 3), so the two rules left to check are that
    every probability lies in [0, 1] and that they sum to 1. The tolerance
    allows for a column that was rounded to three decimals before it reached
    you, which is how they usually arrive.
    """
    p = np.asarray(list(probabilities), dtype=float)
    in_range = bool(((p >= 0) & (p <= 1)).all())
    total = float(p.sum())
    sums_to_one = bool(abs(total - 1.0) <= tolerance)
    if in_range and sums_to_one:
        verdict = "valid"
    elif not in_range:
        bad = ", ".join(f"{v:g}" for v in p[(p < 0) | (p > 1)])
        verdict = f"NOT valid: {bad} is outside [0, 1]"
        if not sums_to_one:
            verdict += f", and the column sums to {total:g}"
    else:
        verdict = f"NOT valid: the column sums to {total:g}, not 1"
    return pd.Series({"every P(x) in [0, 1]": in_range,
                      "sum of P(x)": round(total, 6),
                      "sums to 1": sums_to_one,
                      "verdict": verdict})


def distribution_from_counts(values, counts, name: str = "x") -> pd.DataFrame:
    """Turn a frequency table into a probability distribution.

    This is the relative frequency column from Chapter 2, read as a set of
    empirical probabilities from Chapter 4.
    """
    counts = np.asarray(list(counts), dtype=float)
    if (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Counts must be non-negative and not all zero.")
    table = pd.DataFrame({"frequency": counts.astype(int),
                          "P(x)": counts / counts.sum()},
                         index=pd.Index(list(values), name=name))
    return table


def _validated(values, probabilities):
    x = np.asarray(list(values), dtype=float)
    p = np.asarray(list(probabilities), dtype=float)
    if len(x) != len(p):
        raise ValueError(f"{len(x)} values but {len(p)} probabilities.")
    report = check_distribution(p)
    if report["verdict"] != "valid":
        raise ValueError(f"This is not a probability distribution. "
                         f"{report['verdict']}.")
    return x, p


def dist_mean(values, probabilities) -> float:
    """mu = sum of x P(x). Also called the expected value, E(x)."""
    x, p = _validated(values, probabilities)
    return float((x * p).sum())


def dist_variance(values, probabilities) -> float:
    """sigma^2 = sum of (x - mu)^2 P(x)."""
    x, p = _validated(values, probabilities)
    mu = (x * p).sum()
    return float((((x - mu) ** 2) * p).sum())


def dist_sd(values, probabilities) -> float:
    """sigma, the square root of the variance, in the units of x."""
    return sqrt(dist_variance(values, probabilities))


def moments_table(values, probabilities, name: str = "x") -> pd.DataFrame:
    """The mean and the variance of a discrete distribution, worked out in
    columns the way you would on paper.

    The TOTAL row carries the answers: the x P(x) column sums to the mean, the
    (x - mu)^2 P(x) column sums to the variance, and the x^2 P(x) column is the
    first term of the shortcut formula, sigma^2 = sum of x^2 P(x), minus mu^2.
    Both routes to the variance are shown so that one checks the other.
    """
    x, p = _validated(values, probabilities)
    mu = float((x * p).sum())
    table = pd.DataFrame({
        "P(x)": p,
        "x P(x)": x * p,
        "x - mu": x - mu,
        "(x - mu)^2": (x - mu) ** 2,
        "(x - mu)^2 P(x)": ((x - mu) ** 2) * p,
        "x^2 P(x)": (x ** 2) * p,
    }, index=pd.Index([int(v) if float(v).is_integer() else v for v in x],
                      name=name))
    total = table.sum()
    total["x - mu"] = np.nan
    total["(x - mu)^2"] = np.nan
    table.loc["TOTAL"] = total
    return table


def simulate_distribution(values, probabilities, draws: int = 100_000,
                          seed: int = 20221005) -> pd.Series:
    """Draw a sample from a discrete distribution. The seed is fixed so that
    everyone running this sees the same sample."""
    x, p = _validated(values, probabilities)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(x, size=draws, p=p / p.sum()), name="draw")


# ---------------------------------------------------------------------------
# Expected monetary value
# ---------------------------------------------------------------------------

def emv_table(payoffs: dict, probabilities: dict) -> pd.DataFrame:
    """Expected monetary value, and the risk around it, for each alternative.

    `payoffs` maps each alternative to a {state: dollars} dictionary and
    `probabilities` maps each state to its probability. The result has one row
    per alternative: the payoff in every state, then EMV, the standard
    deviation of the payoff, and the coefficient of variation (SD / EMV).

    EMV ranks the alternatives by what they return on average. The last two
    columns are there because two alternatives with the same EMV are not the
    same decision.
    """
    states = list(probabilities)
    p = np.array([probabilities[s] for s in states], dtype=float)
    report = check_distribution(p)
    if report["verdict"] != "valid":
        raise ValueError(f"State probabilities: {report['verdict']}.")
    rows = {}
    for alt, by_state in payoffs.items():
        missing = [s for s in states if s not in by_state]
        if missing:
            raise ValueError(f"'{alt}' has no payoff for: {missing}")
        v = np.array([by_state[s] for s in states], dtype=float)
        emv = float((v * p).sum())
        sd = sqrt(float((((v - emv) ** 2) * p).sum()))
        row = {f"payoff if {s}": by_state[s] for s in states}
        row.update({"EMV": emv, "SD": sd,
                    "CV": sd / emv if emv > 0 else np.nan})
        rows[alt] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "alternative"
    return table


def stocking_payoffs(order_quantities, demand_levels, price: float,
                     cost: float, salvage: float = 0.0) -> dict:
    """Profit for every combination of how many you stocked and how many
    customers wanted.

    You pay `cost` for every unit stocked, sell min(stocked, demanded) at
    `price`, and recover `salvage` on each unit left over. Unmet demand earns
    nothing; it is simply a sale you did not make. The result is shaped for
    `emv_table`.
    """
    out = {}
    for q in order_quantities:
        out[f"stock {q}"] = {
            f"demand {d}": (price * min(q, d) + salvage * max(q - d, 0)
                            - cost * q)
            for d in demand_levels}
    return out


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------

def _check_np(n: int, p: float):
    if n < 1 or int(n) != n:
        raise ValueError("n must be a whole number of trials, at least 1.")
    if not 0 <= p <= 1:
        raise ValueError(f"p = {p} is not a probability.")


def binomial_pmf(x: int, n: int, p: float) -> float:
    """P(exactly x successes in n trials) = nCx p^x q^(n-x)."""
    _check_np(n, p)
    if x < 0 or x > n:
        return 0.0
    return comb(n, x) * p ** x * (1 - p) ** (n - x)


def binomial_mean(n: int, p: float) -> float:
    return n * p


def binomial_sd(n: int, p: float) -> float:
    return sqrt(n * p * (1 - p))


def _with_cumulative(table: pd.DataFrame) -> pd.DataFrame:
    exact = table["P(X = x)"]
    table["P(X <= x)"] = exact.cumsum().clip(upper=1.0)
    table["P(X >= x)"] = (1.0 - table["P(X <= x)"] + exact).clip(lower=0.0)
    return table


def binomial_table(n: int, p: float) -> pd.DataFrame:
    """Every value the count can take, 0 through n, with the probability of
    exactly x, of x or fewer, and of x or more.

    The first column is Excel's BINOM.DIST(x, n, p, FALSE). The second is
    BINOM.DIST(x, n, p, TRUE). Every binomial question in this chapter is
    answered by reading the right cell of this table.
    """
    _check_np(n, p)
    table = pd.DataFrame(
        {"P(X = x)": [binomial_pmf(k, n, p) for k in range(n + 1)]},
        index=pd.RangeIndex(n + 1, name="x"))
    return _with_cumulative(table)


def binomial_arrangements(n: int, x: int, p: float,
                          labels=("S", "F")) -> pd.DataFrame:
    """List every ordering of x successes among n trials, with the probability
    of each ordering.

    This is the binomial formula taken apart. Every row has the same
    probability, p^x q^(n-x), by the multiplication rule for independent
    events. The rows are mutually exclusive, so by the addition rule the
    probability of "x successes in any order" is that probability times the
    number of rows. The number of rows is nCx.
    """
    _check_np(n, p)
    if comb(n, x) > 5_000:
        raise ValueError(f"{comb(n, x):,} arrangements is too many to list. "
                         f"That is what the formula is for.")
    s, f = labels
    each = p ** x * (1 - p) ** (n - x)
    rows = []
    for where in _subsets(range(n), x):
        seq = "".join(s if i in where else f for i in range(n))
        rows.append({"arrangement": seq, "probability": each})
    table = pd.DataFrame(rows)
    table.index = pd.RangeIndex(1, len(table) + 1, name="way")
    return table


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

def poisson_pmf(x: int, lam: float) -> float:
    """P(exactly x occurrences) = lambda^x e^(-lambda) / x!"""
    if lam <= 0:
        raise ValueError("lambda, the mean number of occurrences, must be "
                         "positive.")
    if x < 0:
        return 0.0
    return lam ** x * exp(-lam) / factorial(x)


def poisson_table(lam: float, max_x: int | None = None) -> pd.DataFrame:
    """Poisson probabilities for x = 0, 1, 2, ... with the same three columns
    as `binomial_table`.

    A Poisson count has no upper limit, so the table has to stop somewhere.
    By default it stops once the probability left in the tail is below one in
    a million, and says how much that is in `attrs["tail beyond table"]`.
    """
    if max_x is None:
        max_x, running = 0, poisson_pmf(0, lam)
        while 1 - running > 1e-6 and max_x < 170:
            max_x += 1
            running += poisson_pmf(max_x, lam)
    table = pd.DataFrame(
        {"P(X = x)": [poisson_pmf(k, lam) for k in range(max_x + 1)]},
        index=pd.RangeIndex(max_x + 1, name="x"))
    table = _with_cumulative(table)
    table.attrs["tail beyond table"] = float(1 - table["P(X = x)"].sum())
    return table


def rescale_rate(mean: float, per: float, to: float) -> float:
    """Convert a Poisson mean from one interval length to another.

    `rescale_rate(12, per=60, to=30)` turns 12 arrivals per 60 minutes into
    6 per 30 minutes. `per` and `to` must be in the same units. The mean
    scales with the interval because a Poisson process has the same rate in
    every equal stretch of it.
    """
    if per <= 0 or to <= 0:
        raise ValueError("Interval lengths must be positive.")
    return mean * to / per


# ---------------------------------------------------------------------------
# Hypergeometric
# ---------------------------------------------------------------------------

def _check_hyper(n: int, R: int, N: int):
    if not (0 <= R <= N and 0 < n <= N):
        raise ValueError(f"Cannot draw n={n} from N={N} with R={R} successes.")


def hypergeometric_pmf(x: int, n: int, R: int, N: int) -> float:
    """P(exactly x successes in a sample of n, drawn without replacement from
    a population of N that contains R successes).

    The argument order matches Excel's HYPGEOM.DIST(x, n, R, N, cumulative).
    """
    _check_hyper(n, R, N)
    if x < max(0, n - (N - R)) or x > min(n, R):
        return 0.0
    return comb(N - R, n - x) * comb(R, x) / comb(N, n)


def hypergeometric_mean(n: int, R: int, N: int) -> float:
    return n * R / N


def hypergeometric_sd(n: int, R: int, N: int) -> float:
    return sqrt(n * R * (N - R) / N ** 2) * sqrt((N - n) / (N - 1))


def hypergeometric_table(n: int, R: int, N: int) -> pd.DataFrame:
    """Every possible number of successes in the sample, with exact and
    cumulative probabilities."""
    _check_hyper(n, R, N)
    table = pd.DataFrame(
        {"P(X = x)": [hypergeometric_pmf(k, n, R, N) for k in range(n + 1)]},
        index=pd.RangeIndex(n + 1, name="x"))
    return _with_cumulative(table)


# ---------------------------------------------------------------------------
# Translating the English
# ---------------------------------------------------------------------------

_PHRASES = {
    "exactly":      ("X = {x}",  "exact"),
    "at most":      ("X <= {x}", "le"),
    "no more than": ("X <= {x}", "le"),
    "or fewer":     ("X <= {x}", "le"),
    "fewer than":   ("X < {x}",  "lt"),
    "less than":    ("X < {x}",  "lt"),
    "at least":     ("X >= {x}", "ge"),
    "no fewer than": ("X >= {x}", "ge"),
    "or more":      ("X >= {x}", "ge"),
    "more than":    ("X > {x}",  "gt"),
}


def probability_of(table: pd.DataFrame, phrase: str, x: int,
                   explain: bool = False):
    """Answer a question asked in English from a distribution table.

    `probability_of(tbl, "fewer than", 4)` returns P(X < 4) = P(X <= 3).
    With `explain=True` it returns a sentence showing the translation it made,
    which is the part worth checking.
    """
    key = phrase.strip().lower()
    if key not in _PHRASES:
        raise ValueError(f"I do not know the phrase '{phrase}'. Use one of: "
                         + ", ".join(_PHRASES))
    symbol, kind = _PHRASES[key]
    exact = table["P(X = x)"]
    top = int(exact.index.max())

    def cdf(k):
        if k < 0:
            return 0.0
        return float(exact.loc[:min(k, top)].sum())

    if kind == "exact":
        value, route = (float(exact.get(x, 0.0)), f"P(X = {x})")
    elif kind == "le":
        value, route = cdf(x), f"P(X <= {x})"
    elif kind == "lt":
        value, route = cdf(x - 1), f"P(X <= {x - 1})"
    elif kind == "ge":
        value, route = 1 - cdf(x - 1), f"1 - P(X <= {x - 1})"
    else:
        value, route = 1 - cdf(x), f"1 - P(X <= {x})"
    value = min(max(value, 0.0), 1.0)
    if explain:
        return (f"'{key} {x}' means {symbol.format(x=x)}, computed as "
                f"{route} = {value:.6f}")
    return value


def phrase_guide(x: int = 4) -> pd.DataFrame:
    """The translation table: the words in the question, the inequality they
    mean, and how to get it from a cumulative probability."""
    rows = [
        ("exactly {x}", "X = {x}", "P(X = {x})", "cumulative = FALSE"),
        ("at most {x}; no more than {x}; {x} or fewer", "X <= {x}",
         "P(X <= {x})", "cumulative = TRUE"),
        ("fewer than {x}; less than {x}", "X < {x}", "P(X <= {xm})",
         "cumulative = TRUE at {xm}"),
        ("at least {x}; {x} or more", "X >= {x}", "1 - P(X <= {xm})",
         "1 minus cumulative at {xm}"),
        ("more than {x}", "X > {x}", "1 - P(X <= {x})",
         "1 minus cumulative at {x}"),
    ]
    fmt = dict(x=x, xm=x - 1)
    return pd.DataFrame(
        [[c.format(**fmt) for c in r] for r in rows],
        columns=["the question says", "which means", "compute", "in Excel"])


# ---------------------------------------------------------------------------
# One distribution standing in for another
# ---------------------------------------------------------------------------

def poisson_vs_binomial(n: int, p: float, max_x: int | None = None
                        ) -> pd.DataFrame:
    """Binomial(n, p) beside Poisson(lambda = np).

    The textbook's rule is that the two are close enough when n >= 20 and
    p <= 0.05. The `difference` column lets you see how close.
    """
    _check_np(n, p)
    lam = n * p
    if max_x is None:
        max_x = min(n, max(6, int(lam + 5 * sqrt(lam))))
    table = pd.DataFrame({
        "binomial": [binomial_pmf(k, n, p) for k in range(max_x + 1)],
        "Poisson": [poisson_pmf(k, lam) for k in range(max_x + 1)],
    }, index=pd.RangeIndex(max_x + 1, name="x"))
    table["difference"] = table["Poisson"] - table["binomial"]
    table.attrs["lambda"] = lam
    table.attrs["rule satisfied"] = bool(n >= 20 and p <= 0.05)
    return table


def binomial_vs_hypergeometric(n: int, R: int, N: int) -> pd.DataFrame:
    """Hypergeometric(n, R, N) beside Binomial(n, p = R/N).

    The binomial pretends each draw is put back. The gap between the columns
    is what that pretence costs, and it shrinks as N grows relative to n.
    """
    _check_hyper(n, R, N)
    p = R / N
    table = pd.DataFrame({
        "hypergeometric (without replacement)":
            [hypergeometric_pmf(k, n, R, N) for k in range(n + 1)],
        "binomial (as if replaced)":
            [binomial_pmf(k, n, p) for k in range(n + 1)],
    }, index=pd.RangeIndex(n + 1, name="x"))
    table["difference"] = (table.iloc[:, 1] - table.iloc[:, 0])
    table.attrs["sample share of population"] = n / N
    return table


# ---------------------------------------------------------------------------
# Does the binomial describe the data?
# ---------------------------------------------------------------------------

def window_counts(flags, window: int, labels=None) -> pd.Series:
    """Count the successes in consecutive, non-overlapping windows.

    `flags` is a True/False sequence in time order. With `window=4` and
    quarterly data, each count is "how many quarters of that year were a
    success". A trailing partial window is dropped and reported, never padded.
    """
    f = pd.Series(list(flags)).astype(bool).to_numpy()
    full = len(f) // window
    dropped = len(f) - full * window
    counts = f[:full * window].reshape(full, window).sum(axis=1)
    if labels is not None:
        lab = list(labels)[:full * window:window]
    else:
        lab = range(1, full + 1)
    out = pd.Series(counts, index=pd.Index(lab, name="window"), name="successes")
    out.attrs["observations dropped"] = dropped
    return out


def dispersion_check(counts, n: int, p: float) -> pd.DataFrame:
    """Observed window counts against what Binomial(n, p) says they should be.

    Returns one row per possible count with the number of windows observed and
    the number the binomial expects. `attrs` carries the means and variances.
    A binomial count has variance npq. If the observed variance is far above
    that, successes are arriving in clumps, which means the trials are not
    independent and the binomial is the wrong model.
    """
    c = pd.Series(list(counts)).astype(int)
    windows = len(c)
    table = pd.DataFrame({
        "windows observed": [int((c == k).sum()) for k in range(n + 1)],
        "binomial P(X = x)": [binomial_pmf(k, n, p) for k in range(n + 1)],
    }, index=pd.RangeIndex(n + 1, name="x"))
    table["windows expected"] = windows * table["binomial P(X = x)"]
    table.attrs.update({
        "windows": windows,
        "observed mean": float(c.mean()),
        "binomial mean": float(n * p),
        "observed variance": float(c.var(ddof=0)),
        "binomial variance": float(n * p * (1 - p)),
        "variance ratio": float(c.var(ddof=0) / (n * p * (1 - p))),
    })
    return table
