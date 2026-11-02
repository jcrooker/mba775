"""Continuous probability distribution helpers for MBA 775, Chapter 6.

Chapter 5 put a probability on every value a count can take. A measurement,
such as a waiting time, a weight, or a growth rate, has too many possible values
for that. Between any two of them there are infinitely many more, so the
probability of any single exact value is zero, and probability has to be
measured as AREA under a curve over an interval.

That one idea organises this file. Every question in the chapter reduces to
an area, and every area here is computed as a difference of two cumulative
probabilities, P(X <= upper) - P(X <= lower). The functions show that
subtraction rather than hiding it, because the most common mistakes in this
chapter are made in setting it up: the wrong tail, the wrong rate, the wrong
side of a continuity correction.

Each function that answers a question can also say, in one line, what it
computed and which Excel formula reproduces it. Excel is what you will be
handed at work, so the formula is worth reading.

Nothing here needs scipy. The normal cumulative probability is built from
math.erf, and its inverse is found by bisection, which is slow by computer
standards and instant by human ones.

You do not need to read this file to do the coursework.
"""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt

import numpy as np
import pandas as pd

from _dist import binomial_pmf, binomial_table, probability_of

__all__ = [
    "normal_pdf", "normal_cdf", "normal_inv", "z_value", "x_value",
    "normal_area", "interval_shape", "normal_interval_table",
    "normal_percentile_table", "empirical_rule_normal", "one_sided_bounds",
    "exponential_parameters", "exponential_pdf", "exponential_cdf",
    "exponential_area", "exponential_percentile",
    "uniform_pdf", "uniform_cdf", "uniform_area", "uniform_mean",
    "uniform_sd", "uniform_percentile",
    "approximation_conditions", "continuity_guide", "normal_approximation",
    "simulate", "sample_vs_theory", "standardize", "z_tail_check",
    "arrivals_per_interval", "normal_fit_check", "six_sigma_table",
]


def _fmt(v: float) -> str:
    """A number for a formula string: no trailing zeros, no thousands commas
    (Excel will not accept them inside a formula)."""
    return f"{v:.6g}" if abs(v) < 1e6 else f"{v:.0f}"


# ---------------------------------------------------------------------------
# The normal distribution
# ---------------------------------------------------------------------------

def _check_sigma(sigma: float):
    if sigma <= 0:
        raise ValueError("The standard deviation must be positive.")


def normal_pdf(x, mu: float = 0.0, sigma: float = 1.0):
    """The HEIGHT of the normal curve at x. This is a density, not a
    probability. Excel's NORM.DIST(x, mu, sigma, FALSE)."""
    _check_sigma(sigma)
    x = np.asarray(x, dtype=float)
    out = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt(2 * pi))
    return float(out) if out.ndim == 0 else out


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """P(X <= x), the area under the normal curve to the left of x.
    Excel's NORM.DIST(x, mu, sigma, TRUE). With mu = 0 and sigma = 1 it is
    NORM.S.DIST(z, TRUE)."""
    _check_sigma(sigma)
    if x == float("inf"):
        return 1.0
    if x == float("-inf"):
        return 0.0
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def normal_inv(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """The value with probability p to its LEFT. Excel's NORM.INV(p, mu,
    sigma), or NORM.S.INV(p) for the standard normal.

    This is the cumulative probability run backwards: you supply the area and
    get the cut-off. Found by bisection on the standard normal, then rescaled
    with x = mu + z sigma.
    """
    _check_sigma(sigma)
    if not 0 < p < 1:
        raise ValueError(f"p = {p} must be strictly between 0 and 1. The "
                         f"normal curve never reaches 0 or 1.")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return mu + sigma * (lo + hi) / 2


def z_value(x: float, mu: float, sigma: float) -> float:
    """z = (x - mu) / sigma: how many standard deviations x is from the mean."""
    _check_sigma(sigma)
    return (x - mu) / sigma


def x_value(z: float, mu: float, sigma: float) -> float:
    """x = mu + z sigma: the z-score formula solved for x."""
    _check_sigma(sigma)
    return mu + z * sigma


def interval_shape(mu: float, lower=None, upper=None) -> str:
    """Name the kind of interval being asked about, relative to the mean.

    The textbook works through every one of these, and a student who can name
    the shape before computing anything rarely gets the subtraction wrong.
    """
    if lower is None and upper is None:
        return "the whole distribution"
    if lower is None:
        return "left tail"
    if upper is None:
        return "right tail"
    if lower < mu < upper:
        return "straddles the mean"
    if lower >= mu:
        return "entirely above the mean"
    return "entirely below the mean"


def normal_area(mu: float, sigma: float, lower=None, upper=None,
                explain: bool = False):
    """P(lower < X < upper) for a normal distribution. Leave `lower` as None
    for a left tail and `upper` as None for a right tail.

    With `explain=True` the result is a Series that shows the z-scores, the
    two cumulative probabilities, their difference, and the Excel formula
    that gives the same answer. That is the whole method, laid out.
    """
    _check_sigma(sigma)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower must not exceed upper.")
    F_up = normal_cdf(upper, mu, sigma) if upper is not None else 1.0
    F_lo = normal_cdf(lower, mu, sigma) if lower is not None else 0.0
    prob = F_up - F_lo
    if not explain:
        return prob

    m, s = _fmt(mu), _fmt(sigma)
    if lower is None:
        event = f"X < {_fmt(upper)}"
        route = f"P(X <= {_fmt(upper)})"
        excel = f"=NORM.DIST({_fmt(upper)}, {m}, {s}, TRUE)"
    elif upper is None:
        event = f"X > {_fmt(lower)}"
        route = f"1 - P(X <= {_fmt(lower)})"
        excel = f"=1 - NORM.DIST({_fmt(lower)}, {m}, {s}, TRUE)"
    else:
        event = f"{_fmt(lower)} < X < {_fmt(upper)}"
        route = f"P(X <= {_fmt(upper)}) - P(X <= {_fmt(lower)})"
        excel = (f"=NORM.DIST({_fmt(upper)}, {m}, {s}, TRUE) - "
                 f"NORM.DIST({_fmt(lower)}, {m}, {s}, TRUE)")
    return pd.Series({
        "event": event,
        "shape": interval_shape(mu, lower, upper),
        "z lower": z_value(lower, mu, sigma) if lower is not None else "-inf",
        "z upper": z_value(upper, mu, sigma) if upper is not None else "+inf",
        "P(X <= upper)": F_up,
        "P(X <= lower)": F_lo,
        "computed as": route,
        "probability": prob,
        "Excel": excel,
    })


def normal_interval_table(mu: float, sigma: float, intervals) -> pd.DataFrame:
    """Several interval questions about one normal distribution, one row
    each. `intervals` is a list of (lower, upper) pairs, using None for an
    open end."""
    rows = []
    for lower, upper in intervals:
        r = normal_area(mu, sigma, lower, upper, explain=True)
        rows.append(r)
    table = pd.DataFrame(rows)
    table.index = pd.RangeIndex(1, len(table) + 1, name="question")
    return table


def normal_percentile_table(mu: float, sigma: float, percents) -> pd.DataFrame:
    """The inverse problem: for each percent, the value with that share of
    the distribution below it.

    `percents` are shares (0.95, not 95). Each row shows z from NORM.S.INV,
    then x = mu + z sigma, then the one-step Excel formula NORM.INV.
    """
    rows = []
    for p in percents:
        z = normal_inv(p)
        rows.append({
            "share below": p,
            "z = NORM.S.INV(share)": z,
            "x = mu + z sigma": x_value(z, mu, sigma),
            "Excel": f"=NORM.INV({_fmt(p)}, {_fmt(mu)}, {_fmt(sigma)})",
        })
    return pd.DataFrame(rows).set_index("share below")


def empirical_rule_normal() -> pd.DataFrame:
    """The 68-95-99.7 rule computed from the normal curve, instead of
    remembered, with Chebyshev's floor beside it for comparison."""
    rows = []
    for k in (1, 2, 3):
        inside = normal_cdf(k) - normal_cdf(-k)
        rows.append({
            "within": f"mu +/- {k} sigma",
            "normal: inside": inside,
            "normal: each tail": (1 - inside) / 2,
            "Chebyshev: at least inside": max(0.0, 1 - 1 / k ** 2),
        })
    return pd.DataFrame(rows).set_index("within")


def one_sided_bounds(k: float) -> pd.Series:
    """How much probability can lie MORE than k standard deviations ABOVE the
    mean? Three answers, which are not interchangeable.

    - If the distribution is normal: exactly 1 - Phi(k).
    - For ANY distribution, Cantelli's inequality: at most 1 / (1 + k^2).
    - Chebyshev bounds both tails together: at most 1 / k^2 outside, which is
      also a valid (looser) bound on one tail.

    Halving Chebyshev's two-tail figure is NOT a bound for a skewed
    distribution, because nothing stops all of the outside probability from
    sitting in one tail. It is listed so you can see how far off it can be.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    return pd.Series({
        "k (standard deviations above the mean)": k,
        "normal: exact upper tail": 1 - normal_cdf(k),
        "any distribution (Cantelli): at most": 1 / (1 + k ** 2),
        "any distribution (Chebyshev, both tails): at most":
            min(1.0, 1 / k ** 2),
        "half of Chebyshev (NOT a valid bound)": min(1.0, 1 / k ** 2) / 2,
    })


# ---------------------------------------------------------------------------
# The exponential distribution
# ---------------------------------------------------------------------------

def _check_rate(lam: float):
    if lam <= 0:
        raise ValueError("lambda is a rate, and must be positive. It does "
                         "not have to be a whole number.")


def exponential_parameters(rate=None, mean=None, per: float = 1.0,
                           to: float = 1.0, unit: str = "minute",
                           event: str = "arrival") -> pd.Series:
    """Get lambda and mu straight before computing anything.

    Supply EITHER `rate` (events per `per` units of time) OR `mean` (time
    units between events, already in the unit you want). `to` rescales a rate
    to a different interval, exactly like rescale_rate in Chapter 5: 32.2
    shots per 60-minute game, per=60, to=1, becomes 0.537 shots per minute.

    lambda is a COUNTABLE RATE (events per unit of time). mu = 1/lambda is a
    MEASURABLE INTERVAL (time per event). They must be in the same time unit,
    which is the step most often skipped.
    """
    if (rate is None) == (mean is None):
        raise ValueError("Give exactly one of rate= or mean=.")
    if rate is not None:
        if per <= 0 or to <= 0:
            raise ValueError("Interval lengths must be positive.")
        lam = rate * to / per
    else:
        if mean <= 0:
            raise ValueError("The mean time between events must be positive.")
        lam = 1.0 / mean
    _check_rate(lam)
    return pd.Series({
        f"lambda ({event}s per {unit})": lam,
        f"mu = 1/lambda ({unit}s per {event})": 1.0 / lam,
        f"sigma = mu ({unit}s)": 1.0 / lam,
    })


def exponential_pdf(x, lam: float):
    """The HEIGHT of the exponential curve, lambda e^(-lambda x), for x >= 0.
    A density, not a probability. At x = 0 it equals lambda, which can be any
    positive number, including numbers above 1."""
    _check_rate(lam)
    x = np.asarray(x, dtype=float)
    out = np.where(x >= 0, lam * np.exp(-lam * np.clip(x, 0, None)), 0.0)
    return float(out) if out.ndim == 0 else out


def exponential_cdf(a: float, lam: float) -> float:
    """P(X <= a) = 1 - e^(-lambda a). Excel's EXPON.DIST(a, lambda, TRUE)."""
    _check_rate(lam)
    return 0.0 if a <= 0 else 1.0 - exp(-lam * a)


def exponential_area(lam: float, lower=None, upper=None,
                     explain: bool = False):
    """P(lower < X < upper) for an exponential distribution with rate lam.

    The right tail has its own short form, P(X > a) = e^(-lambda a), which
    the explanation shows when it applies.
    """
    _check_rate(lam)
    F_up = exponential_cdf(upper, lam) if upper is not None else 1.0
    F_lo = exponential_cdf(lower, lam) if lower is not None else 0.0
    prob = F_up - F_lo
    if not explain:
        return prob
    L = _fmt(lam)
    if lower is None:
        event = f"X < {_fmt(upper)}"
        route = f"1 - e^(-{L} x {_fmt(upper)})"
        excel = f"=EXPON.DIST({_fmt(upper)}, {L}, TRUE)"
    elif upper is None:
        event = f"X > {_fmt(lower)}"
        route = f"e^(-{L} x {_fmt(lower)})"
        excel = f"=1 - EXPON.DIST({_fmt(lower)}, {L}, TRUE)"
    else:
        event = f"{_fmt(lower)} < X < {_fmt(upper)}"
        route = f"P(X <= {_fmt(upper)}) - P(X <= {_fmt(lower)})"
        excel = (f"=EXPON.DIST({_fmt(upper)}, {L}, TRUE) - "
                 f"EXPON.DIST({_fmt(lower)}, {L}, TRUE)")
    return pd.Series({
        "event": event,
        "P(X <= upper)": F_up,
        "P(X <= lower)": F_lo,
        "computed as": route,
        "probability": prob,
        "Excel": excel,
    })


def exponential_percentile(p: float, lam: float) -> float:
    """The waiting time with probability p of being shorter:
    x = -ln(1 - p) / lambda. With p = 0.5 this is the median, which is only
    69% of the mean."""
    _check_rate(lam)
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1.")
    return -log(1 - p) / lam


# ---------------------------------------------------------------------------
# The continuous uniform distribution
# ---------------------------------------------------------------------------

def _check_ab(a: float, b: float):
    if not b > a:
        raise ValueError("The largest value b must exceed the smallest a.")


def uniform_pdf(x, a: float, b: float):
    """The HEIGHT of the uniform density: 1/(b - a) between a and b, zero
    outside. If b - a is less than 1, the height is more than 1."""
    _check_ab(a, b)
    x = np.asarray(x, dtype=float)
    out = np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)
    return float(out) if out.ndim == 0 else out


def uniform_cdf(c: float, a: float, b: float) -> float:
    """P(X <= c) = (c - a)/(b - a), clipped to [0, 1] outside the range."""
    _check_ab(a, b)
    return min(1.0, max(0.0, (c - a) / (b - a)))


def uniform_area(a: float, b: float, lower=None, upper=None,
                 explain: bool = False):
    """P(lower < X < upper) for Uniform(a, b): the width of the interval,
    as a share of the width of the whole distribution. Excel has no uniform
    function; the formula is the arithmetic."""
    _check_ab(a, b)
    lo = a if lower is None else max(a, min(b, lower))
    hi = b if upper is None else max(a, min(b, upper))
    prob = max(0.0, hi - lo) / (b - a)
    if not explain:
        return prob
    return pd.Series({
        "event": (f"{_fmt(lower) if lower is not None else _fmt(a)} < X < "
                  f"{_fmt(upper) if upper is not None else _fmt(b)}"),
        "width of interval inside [a, b]": hi - lo,
        "width of [a, b]": b - a,
        "computed as": f"({_fmt(hi)} - {_fmt(lo)}) / ({_fmt(b)} - {_fmt(a)})",
        "probability": prob,
        "Excel": f"=({_fmt(hi)} - {_fmt(lo)}) / ({_fmt(b)} - {_fmt(a)})",
    })


def uniform_mean(a: float, b: float) -> float:
    """mu = (a + b)/2, the midpoint."""
    _check_ab(a, b)
    return (a + b) / 2


def uniform_sd(a: float, b: float) -> float:
    """sigma = (b - a)/sqrt(12)."""
    _check_ab(a, b)
    return (b - a) / sqrt(12)


def uniform_percentile(p: float, a: float, b: float) -> float:
    """The value with share p below it: x = a + p (b - a)."""
    _check_ab(a, b)
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")
    return a + p * (b - a)


# ---------------------------------------------------------------------------
# The normal approximation to the binomial
# ---------------------------------------------------------------------------

def approximation_conditions(n: int, p: float) -> pd.Series:
    """The textbook's rule: approximate Binomial(n, p) with a normal curve
    when np >= 5 and nq >= 5. The curve has mean np and sd sqrt(npq)."""
    return pd.Series({
        "np": n * p,
        "nq": n * (1 - p),
        "np >= 5 and nq >= 5": bool(n * p >= 5 and n * (1 - p) >= 5),
        "mu = np": n * p,
        "sigma = sqrt(npq)": sqrt(n * p * (1 - p)),
    })


# phrase -> (which whole numbers, corrected normal interval as (lo, hi))
def _corrected(phrase: str, x: int, x2: int | None = None):
    key = phrase.strip().lower()
    if key == "exactly":
        return f"{x}", (x - 0.5, x + 0.5)
    if key in ("at most", "or fewer", "no more than"):
        return f"0, 1, ..., {x}", (None, x + 0.5)
    if key in ("fewer than", "less than"):
        return f"0, 1, ..., {x - 1}", (None, x - 0.5)
    if key in ("at least", "or more", "no fewer than"):
        return f"{x}, {x + 1}, ..., n", (x - 0.5, None)
    if key == "more than":
        return f"{x + 1}, {x + 2}, ..., n", (x + 0.5, None)
    if key == "between":
        if x2 is None:
            raise ValueError("'between' needs x2, the upper whole number.")
        return f"{x}, ..., {x2} (inclusive)", (x - 0.5, x2 + 0.5)
    raise ValueError(f"Unknown phrase '{phrase}'.")


def continuity_guide(x: int = 3) -> pd.DataFrame:
    """The translation table for the normal approximation. It is Chapter 5's
    phrase guide with one more column: where the edge of the normal interval
    goes once the half-unit correction is applied.

    The rule underneath every row: each whole number k is represented by the
    slice of the curve from k - 0.5 to k + 0.5. Include a number and you
    include its whole slice; exclude it and you exclude the whole slice.
    """
    rows = []
    for phrase in ["fewer than", "at most", "exactly", "at least",
                   "more than"]:
        values, (lo, hi) = _corrected(phrase, x)
        if lo is None:
            normal = f"X < {_fmt(hi)}"
        elif hi is None:
            normal = f"X > {_fmt(lo)}"
        else:
            normal = f"{_fmt(lo)} < X < {_fmt(hi)}"
        rows.append({"the question says": f"{phrase} {x}",
                     "whole numbers included": values,
                     "normal interval, corrected": normal})
    values, (lo, hi) = _corrected("between", x, x + 2)
    rows.append({"the question says": f"{x}, {x + 1}, or {x + 2}",
                 "whole numbers included": values,
                 "normal interval, corrected": f"{_fmt(lo)} < X < {_fmt(hi)}"})
    return pd.DataFrame(rows)


def normal_approximation(n: int, p: float, phrase: str, x: int,
                         x2: int | None = None) -> pd.Series:
    """One binomial question answered three ways: exactly (Chapter 5), by
    the normal curve WITH the continuity correction, and by the normal curve
    WITHOUT it. The last column is there so you can see what the correction
    buys."""
    mu, sigma = n * p, sqrt(n * p * (1 - p))
    tbl = binomial_table(n, p)
    if phrase == "between":
        cum = tbl["P(X <= x)"]
        exact = float(cum.loc[x2] - (cum.loc[x - 1] if x > 0 else 0.0))
    else:
        exact = probability_of(tbl, phrase, x)
    values, (lo, hi) = _corrected(phrase, x, x2)
    corrected = normal_area(mu, sigma, lo, hi)
    # Without the correction: use the stated numbers as the edges.
    key = phrase.strip().lower()
    naive_edges = {
        "exactly": (x, x), "at most": (None, x), "or fewer": (None, x),
        "no more than": (None, x), "fewer than": (None, x),
        "less than": (None, x), "at least": (x, None), "or more": (x, None),
        "no fewer than": (x, None), "more than": (x, None),
        "between": (x, x2),
    }[key]
    naive = normal_area(mu, sigma, *naive_edges)
    label = (f"{x} to {x2} inclusive" if key == "between" else f"{key} {x}")
    return pd.Series({
        "question": label,
        "whole numbers": values,
        "normal interval": (f"({'-inf' if lo is None else _fmt(lo)}, "
                            f"{'+inf' if hi is None else _fmt(hi)})"),
        "exact binomial": exact,
        "normal, corrected": corrected,
        "normal, uncorrected": naive,
        "error, corrected": corrected - exact,
        "error, uncorrected": naive - exact,
    })


# ---------------------------------------------------------------------------
# Simulation: the mechanism, and the samples it generates
# ---------------------------------------------------------------------------

def simulate(kind: str, draws: int = 10_000, seed: int = 20261008,
             **params) -> pd.Series:
    """Random draws from a continuous distribution, with a fixed seed so
    everyone sees the same sample.

    kind="normal", mu=, sigma=;  kind="exponential", lam=;
    kind="uniform", a=, b=.
    """
    rng = np.random.default_rng(seed)
    if kind == "normal":
        x = rng.normal(params["mu"], params["sigma"], draws)
    elif kind == "exponential":
        _check_rate(params["lam"])
        # numpy is parameterised by the MEAN, 1/lambda. Mixing the two up is
        # exactly the bug this helper exists to prevent.
        x = rng.exponential(1.0 / params["lam"], draws)
    elif kind == "uniform":
        x = rng.uniform(params["a"], params["b"], draws)
    else:
        raise ValueError("kind must be 'normal', 'exponential' or 'uniform'.")
    return pd.Series(x, name=kind)


def _theory(kind: str, **params):
    if kind == "normal":
        return params["mu"], params["sigma"]
    if kind == "exponential":
        return 1.0 / params["lam"], 1.0 / params["lam"]
    if kind == "uniform":
        return uniform_mean(params["a"], params["b"]), uniform_sd(
            params["a"], params["b"])
    raise ValueError(kind)


def sample_vs_theory(kind: str, sizes=(10, 100, 1_000, 10_000),
                     trials: int = 2_000, seed: int = 20261008,
                     **params) -> pd.DataFrame:
    """How far, on average, a sample mean and sample standard deviation land
    from the true mu and sigma, at several sample sizes.

    The comparison is always against the distribution's OWN mean and standard
    deviation. For the exponential those are both 1/lambda, not lambda.
    """
    mu, sigma = _theory(kind, **params)
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        err_m, err_s = [], []
        for _ in range(trials):
            x = simulate(kind, n, seed=int(rng.integers(1 << 31)), **params)
            err_m.append(abs(x.mean() - mu))
            err_s.append(abs(x.std(ddof=1) - sigma))
        rows.append({"sample size": n,
                     "true mean": mu,
                     "average |sample mean - mu|": float(np.mean(err_m)),
                     "true sd": sigma,
                     "average |sample sd - sigma|": float(np.mean(err_s))})
    return pd.DataFrame(rows).set_index("sample size")


def standardize(x) -> pd.Series:
    """z = (x - mean)/sd for every observation. The result ALWAYS has mean 0
    and sd 1. It has the same SHAPE as x, whatever that shape is."""
    s = pd.Series(x, dtype=float).dropna()
    return (s - s.mean()) / s.std(ddof=1)


def z_tail_check(x, cutoffs=(-2, -1, 1, 2)) -> pd.DataFrame:
    """Share of standardized values below (for negative cut-offs) or above
    (for positive ones) each z, against what the standard normal says.

    If x is normal the two columns agree. If they do not, then 'z follows the
    standard normal' was never true for these data, and the normal table
    will give wrong answers even though every z was computed correctly.
    """
    z = standardize(x)
    rows = []
    for c in cutoffs:
        if c < 0:
            obs, norm, ev = float((z < c).mean()), normal_cdf(c), f"z < {c}"
        else:
            obs, norm, ev = float((z > c).mean()), 1 - normal_cdf(c), f"z > {c}"
        rows.append({"event": ev, "share observed": obs,
                     "standard normal says": norm})
    out = pd.DataFrame(rows).set_index("event")
    out.attrs["mean of z"] = float(z.mean())
    out.attrs["sd of z"] = float(z.std(ddof=1))
    return out


def arrivals_per_interval(gaps, interval: float) -> pd.Series:
    """Turn a sequence of waiting times between arrivals into counts of
    arrivals per interval. The partial interval at the end is dropped.

    If the waiting times are exponential with rate lambda, the counts are
    Poisson with mean lambda x interval. Chapter 5 counted the arrivals;
    Chapter 6 times the gaps between them. It is one process seen two ways.
    """
    times = np.cumsum(np.asarray(gaps, dtype=float))
    n_full = int(times[-1] // interval)
    edges = interval * np.arange(n_full + 1)
    counts, _ = np.histogram(times[times < n_full * interval], bins=edges)
    return pd.Series(counts, name="arrivals",
                     index=pd.RangeIndex(1, n_full + 1, name="interval"))


# ---------------------------------------------------------------------------
# Is it normal enough?
# ---------------------------------------------------------------------------

def normal_fit_check(x, cutoffs) -> pd.DataFrame:
    """For each cut-off, the share of the data below it, against the share a
    normal distribution with the data's own mean and sd would put below it.

    The ratio column is the one to read. Near 1, the normal model is doing
    its job at that point. Far from 1, a manager relying on the normal model
    would be misinformed exactly there, and the tails are where it usually
    happens.
    """
    s = pd.Series(x, dtype=float).dropna()
    m, sd = float(s.mean()), float(s.std(ddof=1))
    rows = []
    for c in cutoffs:
        obs = float((s < c).mean())
        mod = normal_cdf(c, m, sd)
        rows.append({"below": c, "z": z_value(c, m, sd),
                     "share observed": obs, "normal model says": mod,
                     "observed / model": obs / mod if mod > 0 else np.nan,
                     "observations below": int((s < c).sum())})
    out = pd.DataFrame(rows).set_index("below")
    out.attrs.update({"n": len(s), "mean": m, "sd": sd})
    return out


def six_sigma_table(levels=(3, 4, 5, 6), shift: float = 1.5) -> pd.DataFrame:
    """Defects per million opportunities when the specification limits sit
    k standard deviations either side of the target.

    Two columns, because the famous figure needs both. 'Centred' assumes the
    process mean sits exactly on target. Six Sigma practice assumes the mean
    drifts up to `shift` standard deviations over time, which leaves only
    k - shift standard deviations to the nearer limit. The 3.4 per million
    quoted for Six Sigma is the shifted figure, and it is essentially the
    normal tail beyond 4.5 sigma.
    """
    rows = []
    for k in levels:
        centred = 2 * (1 - normal_cdf(k))
        shifted = (1 - normal_cdf(k - shift)) + normal_cdf(-k - shift)
        rows.append({"sigma level": k,
                     "DPMO, centred": 1e6 * centred,
                     f"DPMO, mean shifted {shift} sigma": 1e6 * shifted,
                     "yield, shifted": 1 - shifted})
    return pd.DataFrame(rows).set_index("sigma level")
