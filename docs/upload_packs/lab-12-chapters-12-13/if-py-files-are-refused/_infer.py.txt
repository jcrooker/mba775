"""Confidence interval and hypothesis testing helpers for MBA 775,
Chapters 8 and 9.

Chapter 7 asked: if we knew the population, how would a sample statistic
behave? Chapters 8 and 9 turn that around. We hold one sample and want to say
something about the population we cannot see. One piece of machinery does all
of the work:

    estimate  +/-  (critical value) x (standard error)

  * Read as a range, it is a CONFIDENCE INTERVAL (Chapter 8).
  * Solved for n, it is a SAMPLE SIZE calculation (Chapter 8).
  * Rearranged as (estimate - claimed value) / standard error, it is the
    TEST STATISTIC of a hypothesis test, and its tail area is the P-VALUE
    (Chapter 9).

The functions here return the pieces of each calculation, not just the
answer, together with the Excel formula that reproduces it, so the numbers
can be checked by hand or in a spreadsheet.

The Student's t-distribution is computed here from first principles (the
regularized incomplete beta function), so nothing needs scipy.

You do not need to read this file to do the coursework.
"""

from __future__ import annotations

from math import ceil, exp, lgamma, log, pi, sqrt

import numpy as np
import pandas as pd

from _cont import normal_cdf, normal_inv

__all__ = [
    "t_pdf", "t_cdf", "t_inv", "critical_value", "critical_value_table",
    "mean_interval", "mean_interval_from_data", "proportion_interval",
    "interval_levels_table", "interval_sizes_table",
    "sample_size_mean", "sample_size_proportion", "sample_size_table",
    "p_value", "t_test", "t_test_from_data", "z_test", "proportion_test",
    "interval_test_agreement",
    "critical_mean", "power_mean", "power_table", "power_by_n",
    "n_for_power", "alpha_beta_table", "power_proportion",
    "simulate_intervals", "simulate_t_statistics", "t_tail_rates",
    "simulate_rejections",
    "error_table", "excel_crosswalk",
]

ALTERNATIVES = ("two-sided", "less", "greater")


def _check_alt(alternative: str) -> str:
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, "
                         f"not {alternative!r}")
    return alternative


# ---------------------------------------------------------------------------
# The Student's t-distribution, without scipy
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz).
    Numerical Recipes, section 6.4."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (lgamma(a + b) - lgamma(a) - lgamma(b)
           + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_pdf(x, df: float):
    """Density of the t-distribution with `df` degrees of freedom. Accepts an
    array, so it can be drawn."""
    x = np.asarray(x, dtype=float)
    c = exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / sqrt(df * pi)
    return c * (1.0 + x * x / df) ** (-(df + 1) / 2)


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for the t-distribution with `df` degrees of freedom.
    Excel: =T.DIST(t, df, TRUE)."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if np.isinf(df):
        return normal_cdf(t)
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_inv(p: float, df: float) -> float:
    """The value with area p to its LEFT under the t-distribution.
    Excel: =T.INV(p, df). Solved by bisection on t_cdf, which is slow by
    numerical standards and instant by human ones."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    if np.isinf(df):
        return normal_inv(p)
    if p == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p:
        lo *= 2.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2.0


def critical_value(level: float | None = None, alpha: float | None = None,
                   tails: int = 2, df: float | None = None) -> float:
    """The positive critical value for a confidence `level` or a test at
    significance `alpha`. `df=None` gives z (normal); otherwise t.

    A two-tailed procedure splits alpha between the tails, so it uses the
    alpha/2 point; a one-tailed test puts all of alpha in one tail.
    """
    if (level is None) == (alpha is None):
        raise ValueError("give exactly one of level or alpha")
    if alpha is None:
        alpha = 1.0 - level
    upper = 1.0 - alpha / tails
    return normal_inv(upper) if df is None else t_inv(upper, df)


def critical_value_table(levels=(0.90, 0.95, 0.99),
                         dfs=(5, 10, 20, 30, 60, 100)) -> pd.DataFrame:
    """Two-tailed critical values: t at several df against z. The t value is
    always larger, and the gap closes as df grows."""
    rows = {}
    for df in dfs:
        rows[f"t, df = {df}"] = {f"{level:.0%}": critical_value(level, df=df)
                                 for level in levels}
    rows["z (normal)"] = {f"{level:.0%}": critical_value(level)
                          for level in levels}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def _fpc(n: int, N: int | None) -> float:
    return 1.0 if N is None else sqrt((N - n) / (N - 1))


def mean_interval(xbar: float, sd: float, n: int, level: float = 0.95,
                  sigma_known: bool = False, N: int | None = None
                  ) -> pd.Series:
    """Confidence interval for a population mean.

    sigma_known=False (the usual case): `sd` is the SAMPLE standard deviation
    s, the critical value comes from t with n - 1 df, and Excel's
    =CONFIDENCE.T(alpha, s, n) returns the margin of error.

    sigma_known=True: `sd` is the population sigma and the critical value is
    z. Excel: =CONFIDENCE.NORM(alpha, sigma, n).

    N, if given, applies the finite population correction from Chapter 7.
    """
    df = None if sigma_known else n - 1
    crit = critical_value(level, df=df)
    se = sd / sqrt(n) * _fpc(n, N)
    me = crit * se
    alpha = 1 - level
    excel = (f"=CONFIDENCE.NORM({alpha:.4g}, {sd:.6g}, {n})" if sigma_known
             else f"=CONFIDENCE.T({alpha:.4g}, {sd:.6g}, {n})")
    return pd.Series({
        "estimate (x-bar)": xbar,
        "standard error": se,
        "distribution": "z" if sigma_known else f"t, df = {n - 1}",
        "critical value": crit,
        "margin of error": me,
        "lower": xbar - me,
        "upper": xbar + me,
        "Excel margin of error": excel if N is None else "(apply the FPC by hand)",
    })


def mean_interval_from_data(x, level: float = 0.95) -> pd.Series:
    """t-interval straight from the data: computes x-bar and s first."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    out = mean_interval(float(x.mean()), float(x.std(ddof=1)), len(x), level)
    head = pd.Series({"n": len(x), "s (sample sd)": float(x.std(ddof=1))})
    return pd.concat([head, out])


def proportion_interval(x: int, n: int, level: float = 0.95,
                        N: int | None = None) -> pd.Series:
    """Confidence interval for a population proportion from x successes in n.

    The standard error uses the SAMPLE proportion p-hat, because the interval
    makes no claim about p. (A hypothesis test does make one, and uses it:
    see proportion_test.)
    """
    p_hat = x / n
    crit = critical_value(level)
    se = sqrt(p_hat * (1 - p_hat) / n) * _fpc(n, N)
    me = crit * se
    return pd.Series({
        "successes x": x, "n": n, "estimate (p-hat)": p_hat,
        "n p-hat (want >= 5)": n * p_hat,
        "n (1 - p-hat) (want >= 5)": n * (1 - p_hat),
        "standard error": se, "critical value z": crit,
        "margin of error": me, "lower": p_hat - me, "upper": p_hat + me,
        "Excel critical value": f"=NORM.S.INV({1 - (1 - level) / 2:.4g})",
    })


def interval_levels_table(xbar: float, sd: float, n: int,
                          levels=(0.80, 0.90, 0.95, 0.99),
                          sigma_known: bool = False) -> pd.DataFrame:
    """The same sample at several confidence levels. More confidence costs
    width; nothing else changes."""
    rows = []
    for level in levels:
        r = mean_interval(xbar, sd, n, level, sigma_known)
        rows.append({"confidence": f"{level:.0%}",
                     "critical value": r["critical value"],
                     "margin of error": r["margin of error"],
                     "lower": r["lower"], "upper": r["upper"],
                     "width": r["upper"] - r["lower"]})
    return pd.DataFrame(rows).set_index("confidence")


def interval_sizes_table(sd: float, sizes, level: float = 0.95,
                         sigma_known: bool = False) -> pd.DataFrame:
    """Margin of error at several sample sizes, holding s and the level
    fixed. Quadrupling n halves the margin (a little better than half with t,
    because the critical value also falls)."""
    rows = []
    for n in sizes:
        r = mean_interval(0.0, sd, n, level, sigma_known)
        rows.append({"n": n, "standard error": r["standard error"],
                     "critical value": r["critical value"],
                     "margin of error": r["margin of error"]})
    return pd.DataFrame(rows).set_index("n")


# ---------------------------------------------------------------------------
# Sample size
# ---------------------------------------------------------------------------

def sample_size_mean(sigma: float, E: float, level: float = 0.95
                     ) -> pd.Series:
    """Smallest n giving a margin of error no larger than E for a mean:
    n = (z sigma / E)^2, ROUNDED UP. Rounding down would miss the target."""
    z = critical_value(level)
    raw = (z * sigma / E) ** 2
    return pd.Series({"z": z, "sigma (planning value)": sigma,
                      "target margin E": E, "(z sigma / E)^2": raw,
                      "n (round UP)": int(ceil(raw - 1e-9))}, dtype=object)


def sample_size_proportion(E: float, level: float = 0.95,
                           p: float = 0.5) -> pd.Series:
    """Smallest n giving a margin of error no larger than E for a proportion:
    n = z^2 p (1 - p) / E^2, rounded up. With no prior estimate use p = 0.5,
    which makes p(1 - p) as large as it can be and so is never too small."""
    z = critical_value(level)
    raw = z * z * p * (1 - p) / (E * E)
    return pd.Series({"z": z, "planning p": p, "target margin E": E,
                      "z^2 p (1-p) / E^2": raw,
                      "n (round UP)": int(ceil(raw - 1e-9))}, dtype=object)


def sample_size_table(margins, sigma: float | None = None,
                      level: float = 0.95, p: float = 0.5) -> pd.DataFrame:
    """Required n at several margins of error. Halving E quadruples n."""
    rows = []
    for E in margins:
        r = (sample_size_mean(sigma, E, level) if sigma is not None
             else sample_size_proportion(E, level, p))
        rows.append({"margin of error E": E, "n": int(r["n (round UP)"])})
    return pd.DataFrame(rows).set_index("margin of error E")


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def p_value(stat: float, alternative: str = "two-sided",
            df: float | None = None) -> float:
    """Tail area beyond the test statistic, computed ASSUMING H0 IS TRUE.

    'less' -> left tail, 'greater' -> right tail, 'two-sided' -> both tails
    (twice the smaller one). df=None means the normal distribution.
    """
    _check_alt(alternative)
    cdf = (lambda v: normal_cdf(v)) if df is None else (lambda v: t_cdf(v, df))
    if alternative == "less":
        return cdf(stat)
    if alternative == "greater":
        return 1.0 - cdf(stat)
    return min(1.0, 2.0 * min(cdf(stat), 1.0 - cdf(stat)))


def _excel_p(stat: float, alternative: str, df: float | None) -> str:
    if df is None:
        return {"less": f"=NORM.S.DIST({stat:.4f}, TRUE)",
                "greater": f"=1 - NORM.S.DIST({stat:.4f}, TRUE)",
                "two-sided": f"=2 * (1 - NORM.S.DIST({abs(stat):.4f}, TRUE))"
                }[alternative]
    return {"less": f"=T.DIST({stat:.4f}, {df:g}, TRUE)",
            "greater": f"=T.DIST.RT({stat:.4f}, {df:g})",
            "two-sided": f"=T.DIST.2T({abs(stat):.4f}, {df:g})"}[alternative]


def _decide(stat, crit, p, alpha, alternative, symbol, null_value, label):
    reject = p <= alpha
    h1 = {"less": "<", "greater": ">", "two-sided": "!="}[alternative]
    words = (f"Reject H0. At alpha = {alpha:g} the data are evidence that "
             f"the population {label} {h1} {null_value:g}."
             if reject else
             f"Do not reject H0. At alpha = {alpha:g} the data are not "
             f"sufficient evidence that the population {label} {h1} "
             f"{null_value:g}. (This does NOT show that it equals "
             f"{null_value:g}.)")
    return reject, words


def _crit_for(alpha, alternative, df):
    tails = 2 if alternative == "two-sided" else 1
    c = critical_value(alpha=alpha, tails=tails, df=df)
    return {"less": -c, "greater": c, "two-sided": c}[alternative]


def t_test(xbar: float, s: float, n: int, mu0: float,
           alternative: str = "two-sided", alpha: float = 0.05
           ) -> pd.Series:
    """One-sample t-test for a mean when sigma is unknown (the usual case).

    t = (x-bar - mu0) / (s / sqrt(n)), with n - 1 degrees of freedom. The
    numerator uses mu0, the value CLAIMED in H0, not the unknown mu.
    """
    _check_alt(alternative)
    df = n - 1
    se = s / sqrt(n)
    stat = (xbar - mu0) / se
    p = p_value(stat, alternative, df)
    crit = _crit_for(alpha, alternative, df)
    reject, words = _decide(stat, crit, p, alpha, alternative, "mu", mu0,
                            "mean is")
    return pd.Series({
        "H0 value mu0": mu0, "alternative": alternative,
        "x-bar": xbar, "s": s, "n": n, "standard error s/sqrt(n)": se,
        "test statistic t": stat, "df": df,
        "critical value": (f"+/- {crit:.4f}" if alternative == "two-sided"
                           else crit),
        "p-value": p, "alpha": alpha, "reject H0?": reject,
        "Excel p-value": _excel_p(stat, alternative, df),
        "conclusion": words,
    })


def t_test_from_data(x, mu0: float, alternative: str = "two-sided",
                     alpha: float = 0.05) -> pd.Series:
    """t-test straight from the data: computes x-bar and s first."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    return t_test(float(x.mean()), float(x.std(ddof=1)), len(x), mu0,
                  alternative, alpha)


def z_test(xbar: float, sigma: float, n: int, mu0: float,
           alternative: str = "two-sided", alpha: float = 0.05
           ) -> pd.Series:
    """z-test for a mean when the population sigma is KNOWN. Same steps as
    the t-test; the critical value and p-value come from the normal."""
    _check_alt(alternative)
    se = sigma / sqrt(n)
    stat = (xbar - mu0) / se
    p = p_value(stat, alternative)
    crit = _crit_for(alpha, alternative, None)
    reject, words = _decide(stat, crit, p, alpha, alternative, "mu", mu0,
                            "mean is")
    return pd.Series({
        "H0 value mu0": mu0, "alternative": alternative,
        "x-bar": xbar, "sigma (known)": sigma, "n": n,
        "standard error sigma/sqrt(n)": se, "test statistic z": stat,
        "critical value": (f"+/- {crit:.4f}" if alternative == "two-sided"
                           else crit),
        "p-value": p, "alpha": alpha, "reject H0?": reject,
        "Excel p-value": _excel_p(stat, alternative, None),
        "conclusion": words,
    })


def proportion_test(x: int, n: int, p0: float,
                    alternative: str = "two-sided", alpha: float = 0.05
                    ) -> pd.Series:
    """z-test for a population proportion.

    The standard error uses p0, the value CLAIMED in H0: the test asks how
    surprising p-hat would be IF H0 were true, so it computes the spread H0
    implies. (The confidence interval, which claims nothing, uses p-hat.)
    """
    _check_alt(alternative)
    p_hat = x / n
    se = sqrt(p0 * (1 - p0) / n)
    stat = (p_hat - p0) / se
    p = p_value(stat, alternative)
    crit = _crit_for(alpha, alternative, None)
    reject, words = _decide(stat, crit, p, alpha, alternative, "p", p0,
                            "proportion is")
    return pd.Series({
        "H0 value p0": p0, "alternative": alternative,
        "successes x": x, "n": n, "p-hat": p_hat,
        "n p0 (want >= 5)": n * p0, "n (1 - p0) (want >= 5)": n * (1 - p0),
        "standard error sqrt(p0 (1-p0) / n)": se, "test statistic z": stat,
        "critical value": (f"+/- {crit:.4f}" if alternative == "two-sided"
                           else crit),
        "p-value": p, "alpha": alpha, "reject H0?": reject,
        "Excel p-value": _excel_p(stat, alternative, None),
        "conclusion": words,
    })


def interval_test_agreement(xbar: float, s: float, n: int, null_values,
                            alpha: float = 0.05) -> pd.DataFrame:
    """For several claimed values mu0: is mu0 inside the (1 - alpha) t
    interval, and does the two-sided t-test at alpha reject it? The two
    columns always disagree, because they are the same calculation: the
    interval is exactly the set of mu0 values the test would NOT reject."""
    ci = mean_interval(xbar, s, n, 1 - alpha)
    rows = []
    for mu0 in null_values:
        t = t_test(xbar, s, n, mu0, "two-sided", alpha)
        rows.append({"claimed mu0": mu0,
                     "inside the interval?": ci["lower"] <= mu0 <= ci["upper"],
                     "two-sided p-value": t["p-value"],
                     "test rejects H0?": bool(t["reject H0?"])})
    return pd.DataFrame(rows).set_index("claimed mu0")


# ---------------------------------------------------------------------------
# Type II error and power (sigma known, as in the textbook)
# ---------------------------------------------------------------------------

def critical_mean(mu0: float, sigma: float, n: int, alpha: float = 0.05,
                  alternative: str = "greater"):
    """The sample mean at the edge of the rejection region (Formula 9.5):
    mu0 + z_alpha SE for an upper test, mu0 - z_alpha SE for a lower test, and
    both for a two-sided test (z_{alpha/2})."""
    _check_alt(alternative)
    se = sigma / sqrt(n)
    if alternative == "two-sided":
        z = critical_value(alpha=alpha, tails=2)
        return (mu0 - z * se, mu0 + z * se)
    z = critical_value(alpha=alpha, tails=1)
    return mu0 + z * se if alternative == "greater" else mu0 - z * se


def power_mean(mu0: float, sigma: float, n: int, mu_true: float,
               alpha: float = 0.05, alternative: str = "greater"
               ) -> pd.Series:
    """beta and power for a z-test of a mean, if the true mean is mu_true.

    beta = P(do not reject H0 | the mean is really mu_true). The rejection
    boundary is fixed by H0 and alpha; beta is the area of the TRUE sampling
    distribution that lands on the do-not-reject side of it.
    """
    se = sigma / sqrt(n)
    cut = critical_mean(mu0, sigma, n, alpha, alternative)
    if alternative == "greater":
        beta = normal_cdf(cut, mu_true, se)
        cut_s = f"{cut:.4f}"
    elif alternative == "less":
        beta = 1.0 - normal_cdf(cut, mu_true, se)
        cut_s = f"{cut:.4f}"
    else:
        lo, hi = cut
        beta = normal_cdf(hi, mu_true, se) - normal_cdf(lo, mu_true, se)
        cut_s = f"{lo:.4f} and {hi:.4f}"
    return pd.Series({"mu0 (H0)": mu0, "true mean": mu_true, "sigma": sigma,
                      "n": n, "alpha": alpha, "standard error": se,
                      "critical x-bar": cut_s, "beta = P(Type II error)": beta,
                      "power = 1 - beta": 1.0 - beta})


def power_table(mu0: float, sigma: float, n: int, mus, alpha: float = 0.05,
                alternative: str = "greater") -> pd.DataFrame:
    """beta and power at several possible true means: the numbers behind a
    power curve."""
    rows = []
    for m in mus:
        r = power_mean(mu0, sigma, n, m, alpha, alternative)
        rows.append({"true mean": m, "beta": r["beta = P(Type II error)"],
                     "power": r["power = 1 - beta"]})
    return pd.DataFrame(rows).set_index("true mean")


def power_by_n(mu0: float, sigma: float, sizes, mu_true: float,
               alpha: float = 0.05, alternative: str = "greater"
               ) -> pd.DataFrame:
    """Power to detect the same true mean at several sample sizes. Sample
    size is the lever a manager actually controls."""
    rows = []
    for n in sizes:
        r = power_mean(mu0, sigma, n, mu_true, alpha, alternative)
        rows.append({"n": n, "beta": r["beta = P(Type II error)"],
                     "power": r["power = 1 - beta"]})
    return pd.DataFrame(rows).set_index("n")


def n_for_power(mu0: float, sigma: float, mu_true: float,
                power: float = 0.80, alpha: float = 0.05,
                alternative: str = "greater") -> pd.Series:
    """Smallest n for a one-sided z-test to reach the target power:
    n = ((z_alpha + z_beta) sigma / |mu_true - mu0|)^2, rounded up."""
    tails = 2 if alternative == "two-sided" else 1
    za = critical_value(alpha=alpha, tails=tails)
    zb = normal_inv(power)
    raw = ((za + zb) * sigma / abs(mu_true - mu0)) ** 2
    n = int(ceil(raw - 1e-9))
    return pd.Series({"z_alpha": za, "z_beta": zb,
                      "difference to detect": abs(mu_true - mu0),
                      "raw n": raw, "n (round UP)": n,
                      "power at that n":
                          power_mean(mu0, sigma, n, mu_true, alpha,
                                     alternative)["power = 1 - beta"]},
                     dtype=object)


def alpha_beta_table(mu0: float, sigma: float, n: int, mu_true: float,
                     alphas=(0.001, 0.01, 0.025, 0.05, 0.10, 0.20),
                     alternative: str = "greater") -> pd.DataFrame:
    """beta at several alphas, same n and same true mean. Lowering alpha
    raises beta: there is no free lunch except a bigger sample."""
    rows = []
    for a in alphas:
        r = power_mean(mu0, sigma, n, mu_true, a, alternative)
        rows.append({"alpha": a, "critical x-bar": r["critical x-bar"],
                     "beta": r["beta = P(Type II error)"],
                     "power": r["power = 1 - beta"]})
    return pd.DataFrame(rows).set_index("alpha")


def power_proportion(p0: float, n: int, p_true: float, alpha: float = 0.05,
                     alternative: str = "greater") -> pd.Series:
    """beta and power for a one-sided proportion test (Formula 9.7). The
    critical p-hat uses p0's standard error; beta uses p_true's."""
    if alternative == "two-sided":
        raise ValueError("one-sided tests only")
    z = critical_value(alpha=alpha, tails=1)
    se0 = sqrt(p0 * (1 - p0) / n)
    se1 = sqrt(p_true * (1 - p_true) / n)
    if alternative == "greater":
        cut = p0 + z * se0
        beta = normal_cdf(cut, p_true, se1)
    else:
        cut = p0 - z * se0
        beta = 1.0 - normal_cdf(cut, p_true, se1)
    return pd.Series({"p0 (H0)": p0, "true p": p_true, "n": n,
                      "alpha": alpha, "critical p-hat": cut,
                      "beta = P(Type II error)": beta,
                      "power = 1 - beta": 1.0 - beta})


# ---------------------------------------------------------------------------
# Simulation: checking the procedures, not taking them on faith
# ---------------------------------------------------------------------------

def _draw(population, n, reps, seed):
    rng = np.random.default_rng(seed)
    if callable(population):
        return population(rng, (reps, n))
    pop = np.asarray(population, dtype=float)
    return pop[rng.integers(0, len(pop), size=(reps, n))]


def simulate_intervals(population, n: int, mu: float, level: float = 0.95,
                       reps: int = 100, seed: int = 20261015
                       ) -> pd.DataFrame:
    """Draw `reps` fresh samples of size n, build a t-interval from each, and
    record whether it captured the true mean mu.

    `population` is either an array to sample from (with replacement, so it
    behaves like an infinite population of that shape) or a function
    f(rng, shape) that returns draws.
    """
    x = _draw(population, n, reps, seed)
    xbar = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    me = critical_value(level, df=n - 1) * s / sqrt(n)
    out = pd.DataFrame({"x-bar": xbar, "s": s, "lower": xbar - me,
                        "upper": xbar + me})
    out["captured mu?"] = (out["lower"] <= mu) & (mu <= out["upper"])
    out.index = pd.RangeIndex(1, reps + 1, name="sample")
    return out


def simulate_t_statistics(population, n: int, mu: float, reps: int = 20_000,
                          seed: int = 20261015) -> pd.Series:
    """The t-statistic (x-bar - mu)/(s/sqrt(n)) from `reps` samples drawn
    from a population whose mean really is mu. If the t-distribution is the
    right model, these follow t with n - 1 df."""
    x = _draw(population, n, reps, seed)
    t = (x.mean(axis=1) - mu) / (x.std(axis=1, ddof=1) / sqrt(n))
    return pd.Series(t, name=f"t statistics, n = {n}")


def t_tail_rates(t_stats, n: int, alpha: float = 0.05) -> pd.Series:
    """How often simulated t-statistics fall beyond the one-tailed critical
    values, against the nominal alpha. These are the actual Type I error
    rates of a lower-tailed and an upper-tailed test when H0 is true."""
    t = np.asarray(t_stats, dtype=float)
    c = critical_value(alpha=alpha, tails=1, df=n - 1)
    return pd.Series({"lower-tailed test rejects": float((t < -c).mean()),
                      "upper-tailed test rejects": float((t > c).mean()),
                      "two-sided test rejects":
                          float((np.abs(t) > critical_value(
                              alpha=alpha, tails=2, df=n - 1)).mean()),
                      "nominal alpha": alpha})


def simulate_rejections(population, n: int, mu0: float,
                        alternative: str = "greater", alpha: float = 0.05,
                        reps: int = 20_000, seed: int = 20261015,
                        sigma: float | None = None) -> pd.Series:
    """Run the test on `reps` fresh samples and count how often it rejects.

    If the population's true mean equals mu0, the rejection rate estimates
    the Type I error rate (it should be close to alpha). If not, it
    estimates the POWER, and 1 minus it estimates beta. With `sigma` given
    the test is a z-test; otherwise a t-test.
    """
    _check_alt(alternative)
    x = _draw(population, n, reps, seed)
    xbar = x.mean(axis=1)
    if sigma is None:
        stat = (xbar - mu0) / (x.std(axis=1, ddof=1) / sqrt(n))
        df = n - 1
    else:
        stat = (xbar - mu0) / (sigma / sqrt(n))
        df = None
    tails = 2 if alternative == "two-sided" else 1
    c = critical_value(alpha=alpha, tails=tails, df=df)
    rej = {"greater": stat > c, "less": stat < -c,
           "two-sided": np.abs(stat) > c}[alternative]
    rate = float(rej.mean())
    return pd.Series({"samples": reps, "rejected H0": int(rej.sum()),
                      "rejection rate": rate,
                      "failed to reject (rate)": 1.0 - rate}, dtype=object)


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

def error_table() -> pd.DataFrame:
    """The four outcomes of a test. 'Positive' means the test REJECTS H0,
    that is, it reports finding an effect; 'negative' means it does not."""
    return pd.DataFrame(
        {"H0 is actually TRUE (no effect)": [
            "Type I error = FALSE POSITIVE (probability alpha)",
            "Correct: true negative (probability 1 - alpha)"],
         "H0 is actually FALSE (real effect)": [
            "Correct: true positive (probability 1 - beta = power)",
            "Type II error = FALSE NEGATIVE (probability beta)"]},
        index=["Reject H0  (test is 'positive')",
               "Do not reject H0  (test is 'negative')"])


def excel_crosswalk() -> pd.DataFrame:
    """The Excel function for each step, with what it returns."""
    rows = [
        ("z critical value, two-tailed", "=NORM.S.INV(1 - alpha/2)",
         "1.96 at alpha = 0.05"),
        ("z critical value, one-tailed", "=NORM.S.INV(1 - alpha)",
         "1.645 at alpha = 0.05"),
        ("t critical value, two-tailed", "=T.INV.2T(alpha, df)",
         "positive value"),
        ("t critical value, one-tailed", "=ABS(T.INV(alpha, df))",
         "T.INV returns the LEFT-tail (negative) value"),
        ("margin of error, sigma known", "=CONFIDENCE.NORM(alpha, sigma, n)",
         "z x sigma / SQRT(n)"),
        ("margin of error, sigma unknown", "=CONFIDENCE.T(alpha, s, n)",
         "t x s / SQRT(n)"),
        ("p-value, t, lower tail", "=T.DIST(t, df, TRUE)", "area to the left"),
        ("p-value, t, upper tail", "=T.DIST.RT(t, df)", "area to the right"),
        ("p-value, t, two-tailed", "=T.DIST.2T(ABS(t), df)",
         "needs a NON-NEGATIVE t"),
        ("p-value, z, lower tail", "=NORM.S.DIST(z, TRUE)", "area to the left"),
        ("p-value, z, upper tail", "=1 - NORM.S.DIST(z, TRUE)",
         "area to the right"),
        ("p-value, z, two-tailed", "=2*(1 - NORM.S.DIST(ABS(z), TRUE))",
         "both tails"),
        ("z-test from raw data", "=Z.TEST(range, mu0, sigma)",
         "UPPER-tail p-value only; lower tail is 1 - Z.TEST"),
    ]
    return pd.DataFrame(rows, columns=["step", "Excel", "note"]
                        ).set_index("step")
