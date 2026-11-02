"""Chi-square tests (Chapter 12) and tests for a population variance
(Chapter 13) for MBA 775.

Every chi-square test in Chapter 12 uses one statistic,

    chi-square = sum over cells of (observed - expected)^2 / expected,

and differs only in where the EXPECTED counts come from:

  * k proportions        expected = pooled proportion x each group's total
  * goodness of fit      expected = n x the probabilities the model claims
  * independence         expected = row total x column total / grand total

and in the degrees of freedom. Chapter 13 asks whether the SPREAD of a
process is what was claimed: one variance against a target (chi-square), or
two variances against each other (F).

The chi-square and F distributions are computed here from first principles
(the incomplete gamma and beta functions), so nothing needs scipy.

You do not need to read this file to do the coursework.
"""

from __future__ import annotations

from math import comb, exp, factorial, lgamma, log, sqrt

import numpy as np
import pandas as pd

from _two import f_cdf, f_inv, f_pdf

__all__ = [
    "chi2_pdf", "chi2_cdf", "chi2_sf", "chi2_inv",
    "chi_square_table", "k_proportions_test", "gof_test",
    "poisson_gof", "binomial_gof", "independence_test", "cramers_v",
    "combine_rows",
    "variance_test", "variance_test_from_data", "sigma_interval",
    "f_test_variances", "f_test_from_data",
    "simulate_chi2_null", "variance_test_error_rate",
    "chisq_crosswalk",
]


# ---------------------------------------------------------------------------
# The chi-square distribution, without scipy
# ---------------------------------------------------------------------------

def _gammainc_lower(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) (Numerical Recipes 6.2)."""
    if x <= 0:
        return 0.0
    gln = lgamma(a)
    if x < a + 1.0:                       # series
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return s * exp(-x + a * log(x) - gln)
    # continued fraction for Q, then P = 1 - Q
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - exp(-x + a * log(x) - gln) * h


def chi2_pdf(x, df: float):
    """Density of the chi-square distribution (for drawing)."""
    x = np.asarray(x, dtype=float)
    k = df / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp((k - 1) * np.log(x) - x / 2 - k * log(2) - lgamma(k))
    return np.where(x > 0, out, 0.0)


def chi2_cdf(x: float, df: float) -> float:
    """P(chi-square <= x). Excel: =CHISQ.DIST(x, df, TRUE)."""
    return _gammainc_lower(df / 2.0, x / 2.0)


def chi2_sf(x: float, df: float) -> float:
    """P(chi-square > x), the right tail. Excel: =CHISQ.DIST.RT(x, df)."""
    return 1.0 - chi2_cdf(x, df)


def chi2_inv(p: float, df: float) -> float:
    """The value with area p to its LEFT. The upper critical value at alpha
    is chi2_inv(1 - alpha, df); Excel: =CHISQ.INV.RT(alpha, df)."""
    lo, hi = 0.0, max(1.0, df)
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# The common engine: observed vs expected
# ---------------------------------------------------------------------------

def chi_square_table(observed, expected, labels=None) -> pd.DataFrame:
    """The book's working table: f_o, f_e, f_o - f_e, (f_o - f_e)^2 / f_e,
    with a total row. The total of the last column IS the chi-square
    statistic."""
    fo = np.asarray(observed, dtype=float).ravel()
    fe = np.asarray(expected, dtype=float).ravel()
    out = pd.DataFrame({"observed f_o": fo, "expected f_e": fe,
                        "f_o - f_e": fo - fe,
                        "(f_o - f_e)^2 / f_e": (fo - fe) ** 2 / fe},
                       index=labels)
    out.loc["Total"] = [fo.sum(), fe.sum(), np.nan, out["(f_o - f_e)^2 / f_e"].sum()]
    return out


def _result(stat, df, alpha, extra=None, note=None):
    p = chi2_sf(stat, df)
    out = {"chi-square statistic": stat, "df": df, "p-value": p,
           "critical value": chi2_inv(1 - alpha, df), "alpha": alpha,
           "reject H0?": p <= alpha,
           "Excel p-value": f"=CHISQ.DIST.RT({stat:.4f}, {df:g})",
           "Excel critical value": f"=CHISQ.INV.RT({alpha:g}, {df:g})"}
    if extra:
        out.update(extra)
    if note:
        out["note"] = note
    return pd.Series(out, dtype=object)


def _small_warning(fe):
    m = float(np.min(fe))
    return (None if m >= 5 else
            f"Smallest expected count is {m:.2f} (< 5): combine categories "
            "before trusting this p-value.")


def k_proportions_test(successes, totals, labels=None, alpha: float = 0.05):
    """Chi-square test that k population proportions are equal.

    Returns (working table, result). Expected successes in each group are the
    POOLED proportion times that group's total (the same pooling as the
    two-proportion z-test of Chapter 10). df = k - 1.
    """
    x = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    pbar = x.sum() / n.sum()
    labels = list(labels) if labels is not None else [f"group {i + 1}" for i in range(len(x))]
    obs = np.column_stack([x, n - x]).ravel()
    exp_ = np.column_stack([pbar * n, (1 - pbar) * n]).ravel()
    cell_labels = [f"{g}: {s}" for g in labels for s in ("success", "failure")]
    table = chi_square_table(obs, exp_, cell_labels)
    stat = float(table.loc["Total", "(f_o - f_e)^2 / f_e"])
    res = _result(stat, len(x) - 1, alpha, {"pooled proportion": pbar,
                                            "smallest expected count": exp_.min()},
                  _small_warning(exp_))
    return table, res


def gof_test(observed, probabilities, labels=None, m: int = 0,
             alpha: float = 0.05):
    """Goodness-of-fit: do observed counts follow the stated probabilities?

    df = k - m - 1, where k is the number of categories and m the number of
    parameters ESTIMATED FROM THE DATA to get the probabilities (0 if the
    model was fully stated in H0). Returns (working table, result).
    """
    fo = np.asarray(observed, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if abs(p.sum() - 1) > 1e-6:
        raise ValueError("probabilities must sum to 1")
    fe = fo.sum() * p
    table = chi_square_table(fo, fe, labels)
    stat = float(table.loc["Total", "(f_o - f_e)^2 / f_e"])
    df = len(fo) - m - 1
    excel_test = ("=CHISQ.TEST(observed, expected) gives the SAME p-value"
                  if m == 0 else
                  "=CHISQ.TEST(observed, expected) would use df = k - 1 and give the WRONG p-value here")
    res = _result(stat, df, alpha, {"categories k": len(fo),
                                    "parameters estimated m": m,
                                    "smallest expected count": fe.min(),
                                    "CHISQ.TEST": excel_test},
                  _small_warning(fe))
    return table, res


def combine_rows(values, freqs, probs, top: int):
    """Merge every category from `top` upward into one 'top or more'
    category, so its expected count can reach 5. The probability of the
    merged category is 1 minus everything below it (the whole upper tail)."""
    values = list(values)
    keep = [i for i, v in enumerate(values) if v < top]
    labels = [str(values[i]) for i in keep] + [f"{top} or more"]
    f = [freqs[i] for i in keep] + [sum(freqs[i] for i, v in enumerate(values) if v >= top)]
    p = [probs[i] for i in keep]
    p.append(1.0 - sum(p))
    return labels, np.array(f, dtype=float), np.array(p, dtype=float)


def poisson_gof(values, freqs, top: int, lam: float | None = None,
                alpha: float = 0.05):
    """Does a count follow a Poisson distribution? If `lam` is None it is
    estimated from the data (m = 1); if stated in H0, m = 0. Categories from
    `top` upward are merged into 'top or more'."""
    v = np.asarray(values, dtype=float)
    f = np.asarray(freqs, dtype=float)
    est = lam is None
    lam = float((v * f).sum() / f.sum()) if est else lam
    probs = [exp(-lam) * lam ** int(k) / factorial(int(k)) for k in values]
    labels, fo, p = combine_rows(values, list(f), probs, top)
    table, res = gof_test(fo, p, labels, m=1 if est else 0, alpha=alpha)
    res = pd.concat([pd.Series({"lambda": lam,
                                "lambda estimated from data?": est}, dtype=object), res])
    return table, res


def binomial_gof(values, freqs, n: int, top: int, p: float | None = None,
                 alpha: float = 0.05):
    """Does a count of successes in n trials follow a binomial distribution?
    If `p` is None it is estimated from the data (m = 1)."""
    v = np.asarray(values, dtype=float)
    f = np.asarray(freqs, dtype=float)
    est = p is None
    p = float((v * f).sum() / (n * f.sum())) if est else p
    probs = [comb(n, int(k)) * p ** int(k) * (1 - p) ** (n - int(k)) for k in values]
    labels, fo, pr = combine_rows(values, list(f), probs, top)
    table, res = gof_test(fo, pr, labels, m=1 if est else 0, alpha=alpha)
    res = pd.concat([pd.Series({"trials n": n, "p": p,
                                "p estimated from data?": est}, dtype=object), res])
    return table, res


def cramers_v(stat: float, n: float, r: int, c: int) -> float:
    """Effect size for a contingency table: 0 = no association, 1 = perfect.
    Rough guide for tables with min(r, c) = 2: 0.1 small, 0.3 medium,
    0.5 large."""
    return sqrt(stat / (n * (min(r, c) - 1)))


def independence_test(table, alpha: float = 0.05):
    """Chi-square test of independence for a contingency table (a DataFrame
    of counts, no totals). Returns a dict with the expected counts, the
    standardized residuals (f_o - f_e)/sqrt(f_e) that show WHERE the
    dependence lies, and the result Series, including Cramer's V for HOW
    STRONG it is."""
    obs = table.astype(float)
    n = obs.values.sum()
    expv = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    expected = pd.DataFrame(expv, index=obs.index, columns=obs.columns)
    resid = (obs - expected) / np.sqrt(expected)
    stat = float(((obs - expected) ** 2 / expected).values.sum())
    r, c = obs.shape
    res = _result(stat, (r - 1) * (c - 1), alpha,
                  {"rows r": r, "columns c": c, "n": n,
                   "smallest expected count": expv.min(),
                   "Cramer's V": cramers_v(stat, n, r, c)},
                  _small_warning(expv))
    return {"expected": expected, "residuals": resid, "result": res}


# ---------------------------------------------------------------------------
# Chapter 13: one variance, two variances
# ---------------------------------------------------------------------------

ALTS = ("two-sided", "less", "greater")


def variance_test(s2: float, n: int, sigma2_0: float,
                  alternative: str = "two-sided", alpha: float = 0.05
                  ) -> pd.Series:
    """Chi-square test for one population variance:
    chi-square = (n - 1) s^2 / sigma0^2 with n - 1 df.

    Requires a NORMAL population, and unlike the t-test this requirement
    does not fade as n grows (see variance_test_error_rate)."""
    if alternative not in ALTS:
        raise ValueError(f"alternative must be one of {ALTS}")
    df = n - 1
    stat = df * s2 / sigma2_0
    lower = chi2_cdf(stat, df)
    if alternative == "less":
        p = lower
        crit = f"{chi2_inv(alpha, df):.4f} (reject below)"
        excel = f"=CHISQ.DIST({stat:.4f}, {df}, TRUE)"
    elif alternative == "greater":
        p = 1 - lower
        crit = f"{chi2_inv(1 - alpha, df):.4f} (reject above)"
        excel = f"=CHISQ.DIST.RT({stat:.4f}, {df})"
    else:
        p = min(1.0, 2 * min(lower, 1 - lower))
        crit = (f"{chi2_inv(alpha / 2, df):.4f} and "
                f"{chi2_inv(1 - alpha / 2, df):.4f}")
        excel = f"=2*MIN(CHISQ.DIST({stat:.4f},{df},TRUE), CHISQ.DIST.RT({stat:.4f},{df}))"
    return pd.Series({"H0 variance sigma0^2": sigma2_0,
                      "H0 standard deviation": sqrt(sigma2_0),
                      "alternative": alternative, "n": n,
                      "sample variance s^2": s2, "sample sd s": sqrt(s2),
                      "chi-square statistic (n-1)s^2/sigma0^2": stat,
                      "df": df, "critical value(s)": crit,
                      "p-value": p, "alpha": alpha, "reject H0?": p <= alpha,
                      "Excel p-value": excel}, dtype=object)


def variance_test_from_data(x, sigma2_0: float, alternative: str = "two-sided",
                            alpha: float = 0.05) -> pd.Series:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    return variance_test(float(x.var(ddof=1)), len(x), sigma2_0, alternative,
                         alpha)


def sigma_interval(s: float, n: int, level: float = 0.95) -> pd.Series:
    """Confidence interval for a population standard deviation (normal data):
    sqrt((n-1)s^2 / chi2_upper) to sqrt((n-1)s^2 / chi2_lower). Note it is NOT
    symmetric around s: the chi-square distribution is skewed."""
    df = n - 1
    a = 1 - level
    hi_q, lo_q = chi2_inv(1 - a / 2, df), chi2_inv(a / 2, df)
    lo, hi = sqrt(df * s * s / hi_q), sqrt(df * s * s / lo_q)
    return pd.Series({"s": s, "n": n, "level": level,
                      "chi-square (upper)": hi_q, "chi-square (lower)": lo_q,
                      "lower limit for sigma": lo, "upper limit for sigma": hi,
                      "lower limit for sigma^2": lo * lo,
                      "upper limit for sigma^2": hi * hi})


def f_test_variances(s1: float, n1: int, s2: float, n2: int,
                     alternative: str = "two-sided", alpha: float = 0.05,
                     labels=("population 1", "population 2")) -> pd.Series:
    """F test comparing two population variances: F = s1^2 / s2^2 with
    (n1 - 1, n2 - 1) df. `s1`, `s2` are standard deviations.

    Population 1 and 2 are fixed by the QUESTION, not by which sample came
    out larger. For 'greater' (H1: sigma1^2 > sigma2^2), F < 1 simply means
    no evidence. For 'two-sided' the p-value doubles the smaller tail, which
    is the same as the book's larger-on-top rule with alpha/2."""
    if alternative not in ALTS:
        raise ValueError(f"alternative must be one of {ALTS}")
    d1, d2 = n1 - 1, n2 - 1
    F = (s1 * s1) / (s2 * s2)
    upper = 1 - f_cdf(F, d1, d2)
    if alternative == "greater":
        p = upper
        crit = f"{f_inv(1 - alpha, d1, d2):.4f} (reject above)"
    elif alternative == "less":
        p = 1 - upper
        crit = f"{f_inv(alpha, d1, d2):.4f} (reject below)"
    else:
        p = min(1.0, 2 * min(upper, 1 - upper))
        crit = f"{f_inv(alpha / 2, d1, d2):.4f} and {f_inv(1 - alpha / 2, d1, d2):.4f}"
    return pd.Series({"population 1": labels[0], "population 2": labels[1],
                      "alternative": alternative,
                      "s1^2": s1 * s1, "s2^2": s2 * s2, "F = s1^2/s2^2": F,
                      "df numerator": d1, "df denominator": d2,
                      "critical value(s)": crit, "p-value": p, "alpha": alpha,
                      "reject H0?": p <= alpha,
                      "Excel upper-tail area": f"=F.DIST.RT({F:.4f}, {d1}, {d2})"},
                     dtype=object)


def f_test_from_data(x1, x2, alternative: str = "two-sided",
                     alpha: float = 0.05, labels=("population 1", "population 2")):
    a = np.asarray(pd.Series(x1).dropna(), dtype=float)
    b = np.asarray(pd.Series(x2).dropna(), dtype=float)
    return f_test_variances(a.std(ddof=1), len(a), b.std(ddof=1), len(b),
                            alternative, alpha, labels)


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

def simulate_chi2_null(probs, n: int, reps: int = 20_000,
                       seed: int = 20261105) -> np.ndarray:
    """Chi-square statistics from `reps` samples of size n drawn from the
    model in H0 (multinomial with `probs`). Their histogram should match the
    chi-square density with k - 1 df."""
    p = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, p, size=reps)
    fe = n * p
    return ((counts - fe) ** 2 / fe).sum(axis=1)


def variance_test_error_rate(population, n: int, reps: int = 20_000,
                             alpha: float = 0.05, seed: int = 20261105
                             ) -> pd.Series:
    """How often a two-sided variance test (and, for comparison, a two-sided
    t-test of the mean) rejects a TRUE null when sampling from `population`
    (an array, sampled with replacement) or from a normal (population=None).
    Both nulls use the population's true values, so every rejection is a
    false positive; the nominal rate is alpha."""
    from _infer import critical_value
    rng = np.random.default_rng(seed)
    if population is None:
        x = rng.normal(0.0, 1.0, (reps, n))
        mu, var = 0.0, 1.0
    else:
        pop = np.asarray(population, dtype=float)
        x = pop[rng.integers(0, len(pop), (reps, n))]
        mu, var = pop.mean(), pop.var()
    s2 = x.var(axis=1, ddof=1)
    stat = (n - 1) * s2 / var
    lo, hi = chi2_inv(alpha / 2, n - 1), chi2_inv(1 - alpha / 2, n - 1)
    var_rej = float(((stat < lo) | (stat > hi)).mean())
    t = (x.mean(axis=1) - mu) / np.sqrt(s2 / n)
    tc = critical_value(alpha=alpha, tails=2, df=n - 1)
    t_rej = float((np.abs(t) > tc).mean())
    return pd.Series({"variance test rejects a true H0": var_rej,
                      "t-test rejects a true H0": t_rej,
                      "nominal alpha": alpha})


def chisq_crosswalk() -> pd.DataFrame:
    rows = [
        ("chi-square p-value (right tail)", "=CHISQ.DIST.RT(x, df)",
         "every Chapter 12 test; upper-tail variance test"),
        ("chi-square critical value", "=CHISQ.INV.RT(alpha, df)",
         "upper critical value"),
        ("chi-square left-tail area", "=CHISQ.DIST(x, df, TRUE)",
         "lower-tail variance test"),
        ("p-value from observed and expected ranges", "=CHISQ.TEST(obs, exp)",
         "uses df = k - 1 (one column) or (r-1)(c-1): WRONG if parameters were estimated"),
        ("sample variance", "=VAR.S(range)", "s^2, divisor n - 1"),
        ("F p-value (right tail)", "=F.DIST.RT(F, df1, df2)",
         "double it for a two-tailed test"),
        ("F critical value", "=F.INV.RT(alpha, df1, df2)",
         "use alpha/2 for a two-tailed test"),
        ("F test from raw data", "Data Analysis > F-Test Two-Sample for Variances",
         "reports ONE-tailed results only"),
        ("F test p-value from raw data", "=F.TEST(range1, range2)",
         "TWO-tailed p-value"),
    ]
    return pd.DataFrame(rows, columns=["task", "Excel", "note"]).set_index("task")
