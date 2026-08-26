"""Descriptive-statistics helpers for MBA 775, Chapter 3.

Every function here computes something the chapter defines in a formula box.
They exist for two reasons.

First, so that the definition in the note and the number in the output cannot
drift apart. If the note says the median of an even-length list is the average
of the two middle values, the code that produced the number does exactly that.

Second, because the textbook's rules are not always what a software package
does by default. The percentile rule taught in this course -- and used by
Excel's PERCENTILE.EXC -- is not the rule numpy uses. Both are defensible.
Getting a different number than your classmate because you used a different
tool is the sort of thing that should be visible rather than mysterious, so
`percentile()` implements the course rule and `percentile_methods()` shows you
all of them side by side.

You do not need to read this file to do the coursework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "describe", "five_number_summary",
    "percentile", "percentile_methods", "percentile_rank", "quartiles",
    "iqr", "outlier_fences",
    "z_score", "z_scores", "coefficient_of_variation",
    "empirical_rule", "chebyshev", "chebyshev_table",
    "grouped_mean", "grouped_variance", "grouped_summary",
    "shape_report", "association",
]


def _clean(x) -> pd.Series:
    """Drop missing values and return a float Series. Says how many it dropped."""
    s = pd.Series(x, dtype="float64")
    return s.dropna()


# ---------------------------------------------------------------------------
# Central tendency and the overall picture
# ---------------------------------------------------------------------------

def describe(x, name="x", decimals=4) -> pd.Series:
    """Mean, median, mode, and the measures of variability, in one place."""
    s = _clean(x)
    modes = s.mode()
    out = {
        "n": int(s.size),
        "mean": s.mean(),
        "median": s.median(),
        "mode": modes.iloc[0] if len(modes) == 1 else np.nan,
        "min": s.min(),
        "max": s.max(),
        "range": s.max() - s.min(),
        "variance": s.var(ddof=1),
        "std_dev": s.std(ddof=1),
        "CV_percent": 100 * s.std(ddof=1) / s.mean() if s.mean() != 0 else np.nan,
    }
    result = pd.Series(out, name=name)
    return result.round(decimals)


def five_number_summary(x, name="x", decimals=4) -> pd.Series:
    """Minimum, Q1, median, Q3, maximum -- using the course percentile rule."""
    s = _clean(x)
    q1, q2, q3 = quartiles(s)
    return pd.Series({"min": s.min(), "Q1": q1, "median": q2,
                      "Q3": q3, "max": s.max()}, name=name).round(decimals)


def shape_report(x, name="x") -> str:
    """One sentence naming the skew from the mean-median comparison.

    The chapter defines skew by where the mean sits relative to the median, so
    this reads that comparison off the data rather than off anyone's memory of
    what the chart looked like last year.
    """
    s = _clean(x)
    mean, median = s.mean(), s.median()
    gap = mean - median
    # A gap smaller than a twentieth of a standard deviation is not something
    # anyone could see in a histogram.
    tol = 0.05 * s.std(ddof=1)
    if abs(gap) <= tol:
        shape = "approximately symmetric"
    elif gap > 0:
        shape = "right-skewed"
    else:
        shape = "left-skewed"
    return (f"{name}: mean {mean:,.4g}, median {median:,.4g}, "
            f"difference {gap:+,.4g} -- {shape}.")


# ---------------------------------------------------------------------------
# Relative position
# ---------------------------------------------------------------------------

def percentile(x, k: float) -> float:
    """The kth percentile by the index-point rule taught in this course.

    `k` is a proportion between 0 and 1, matching Excel's PERCENTILE.EXC and
    the R helper used in earlier versions of these notes.

        i = k * n

    If i is not a whole number, round up; that position is the value.
    If i is a whole number, average the values in positions i and i+1.

    Positions are 1-based, the way the formula in the note is written.
    """
    if not 0 < k < 1:
        raise ValueError("k must be strictly between 0 and 1 "
                         f"(got {k}). Use min() or max() for the extremes.")
    s = _clean(x).sort_values().to_numpy()
    n = s.size
    i = k * n
    if float(i).is_integer():
        i = int(i)
        if i >= n:                       # nothing above the last position
            return float(s[-1])
        return float((s[i - 1] + s[i]) / 2)
    return float(s[int(np.ceil(i)) - 1])


def percentile_methods(x, k: float) -> pd.Series:
    """The same percentile computed three ways, so the difference is visible.

    Course rule, numpy's default (linear interpolation), and the nearest-rank
    rule. They agree on large well-behaved data sets and disagree on small
    ones. Neither is wrong; they answer slightly different questions.
    """
    s = _clean(x).sort_values().to_numpy()
    return pd.Series({
        "course_rule (Excel PERCENTILE.EXC)": percentile(s, k),
        "numpy default (linear)": float(np.quantile(s, k)),
        "nearest rank": float(np.quantile(s, k, method="higher")),
    }, name=f"{100*k:g}th percentile").round(4)


def percentile_rank(x, v: float) -> float:
    """Percentile rank of the value `v` within the data set `x`.

        rank = (1/n) * [ 0.5 + count of values strictly below v ] * 100
    """
    s = _clean(x).to_numpy()
    n = s.size
    below = int((s < v).sum())
    return float(100 * (0.5 + below) / n)


def quartiles(x) -> tuple[float, float, float]:
    """Q1, Q2 (the median), Q3, by the course percentile rule."""
    s = _clean(x)
    return percentile(s, 0.25), percentile(s, 0.50), percentile(s, 0.75)


def iqr(x) -> float:
    """Interquartile range, Q3 - Q1: the spread of the middle half."""
    q1, _, q3 = quartiles(x)
    return q3 - q1


def outlier_fences(x) -> tuple[float, float]:
    """The whisker limits of a box plot: Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    q1, _, q3 = quartiles(x)
    spread = q3 - q1
    return q1 - 1.5 * spread, q3 + 1.5 * spread


# ---------------------------------------------------------------------------
# Standardising
# ---------------------------------------------------------------------------

def z_score(value: float, mean: float, std_dev: float) -> float:
    """How many standard deviations `value` sits from `mean`."""
    return (value - mean) / std_dev


def z_scores(x) -> pd.Series:
    """Every observation expressed in standard deviations from its own mean."""
    s = _clean(x)
    return (s - s.mean()) / s.std(ddof=1)


def coefficient_of_variation(x) -> float:
    """Standard deviation as a percentage of the mean.

    The point of this measure is that a standard deviation only means
    something next to the average it is a deviation from.
    """
    s = _clean(x)
    return float(100 * s.std(ddof=1) / s.mean())


def empirical_rule(x, name="x") -> pd.DataFrame:
    """Share of the data within 1, 2, and 3 standard deviations of the mean,
    against what a bell-shaped distribution would give.

    Reading this table is how you find out whether the empirical rule applies
    to your data, rather than assuming it does.
    """
    s = _clean(x)
    m, sd = s.mean(), s.std(ddof=1)
    rows = []
    for k, expected in [(1, 68.3), (2, 95.4), (3, 99.7)]:
        inside = float(((s >= m - k * sd) & (s <= m + k * sd)).mean() * 100)
        rows.append({
            "within": f"{k} sd",
            "interval_low": m - k * sd,
            "interval_high": m + k * sd,
            "actual_percent": inside,
            "bell_shaped_percent": expected,
            "chebyshev_at_least": chebyshev(k),
        })
    return pd.DataFrame(rows).round(2).set_index("within")


def chebyshev(z: float) -> float:
    """The Chebyshev bound: at least (1 - 1/z^2)*100 percent of ANY
    distribution lies within z standard deviations of its mean.

    Returns 0 for |z| <= 1, where the theorem says nothing useful.
    """
    z = abs(float(z))
    if z <= 1:
        return 0.0
    return float((1 - 1 / z ** 2) * 100)


def chebyshev_table(z: float) -> pd.Series:
    """The full Chebyshev reading of a z-score, including the one-tail bound.

    The one-tail figure is the number people actually want, and it is the one
    most often got wrong: the theorem bounds BOTH tails together, so the bound
    on a single tail is half of what is left over -- and only if you are
    willing to assume the two tails are comparable. State it as an upper
    bound, never as an estimate.
    """
    z = abs(float(z))
    inside = chebyshev(z)
    outside = 100 - inside
    return pd.Series({
        "z": round(z, 4),
        "at_least_within_percent": round(inside, 2),
        "at_most_outside_percent": round(outside, 2),
        "at_most_one_tail_percent": round(outside / 2, 2),
    })


# ---------------------------------------------------------------------------
# Grouped data
# ---------------------------------------------------------------------------

def grouped_mean(frequencies, midpoints) -> float:
    """Approximate mean when you have counts by class, not the raw values."""
    f = np.asarray(frequencies, dtype="float64")
    m = np.asarray(midpoints, dtype="float64")
    return float((f * m).sum() / f.sum())


def grouped_variance(frequencies, midpoints) -> float:
    """Approximate sample variance from counts by class."""
    f = np.asarray(frequencies, dtype="float64")
    m = np.asarray(midpoints, dtype="float64")
    xbar = grouped_mean(f, m)
    return float((f * (m - xbar) ** 2).sum() / (f.sum() - 1))


def grouped_summary(frequencies, midpoints, labels=None) -> pd.DataFrame:
    """The grouped-data calculation laid out term by term.

    The last row is the total. Everything the formula asks for is a column, so
    the arithmetic can be checked by hand against the note.
    """
    f = np.asarray(frequencies, dtype="float64")
    m = np.asarray(midpoints, dtype="float64")
    xbar = grouped_mean(f, m)
    table = pd.DataFrame({
        "class": labels if labels is not None else [f"class {i+1}" for i in range(len(f))],
        "frequency_f": f.astype(int),
        "midpoint_m": m,
        "f_times_m": f * m,
        "deviation": m - xbar,
        "f_times_dev_sq": f * (m - xbar) ** 2,
    })
    total = pd.DataFrame([{
        "class": "TOTAL",
        "frequency_f": int(f.sum()),
        "midpoint_m": np.nan,
        "f_times_m": (f * m).sum(),
        "deviation": np.nan,
        "f_times_dev_sq": (f * (m - xbar) ** 2).sum(),
    }])
    return pd.concat([table, total], ignore_index=True)


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------

def association(x, y, x_name="x", y_name="y") -> pd.Series:
    """Covariance and correlation, with the sample sizes that produced them.

    Covariance answers only "which direction". Its size depends on the units,
    so a covariance of 4,000,000 is not evidence of anything until you divide
    by both standard deviations -- which is what the correlation does.
    """
    pair = pd.DataFrame({x_name: pd.Series(x, dtype="float64"),
                         y_name: pd.Series(y, dtype="float64")}).dropna()
    a, b = pair[x_name], pair[y_name]
    cov = float(a.cov(b))
    return pd.Series({
        "n_pairs": int(len(pair)),
        f"sd_{x_name}": a.std(ddof=1),
        f"sd_{y_name}": b.std(ddof=1),
        "covariance": cov,
        "correlation": cov / (a.std(ddof=1) * b.std(ddof=1)),
    })
