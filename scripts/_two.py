"""Two-population comparisons for MBA 775, Chapter 10, and the Chapter 11
slice on analysis of variance.

Every test this week has the same shape:

    (difference in estimates - difference claimed in H0)
    ----------------------------------------------------
          standard error of the difference

Only the standard error changes, and the DESIGN decides which one:

  * independent samples, sigmas unknown  ->  Welch (the default) or pooled
  * the same units measured twice         ->  paired: work with differences
  * two proportions                       ->  pooled p-hat for the test,
                                              separate p-hats for the interval

The ANOVA functions extend the two-sample comparison to k groups with one
test, and show why running every pairwise t-test inflates false positives.

The t and F distributions are computed from first principles in _infer.py
and here, so nothing needs scipy. You do not need to read this file to do
the coursework.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, exp, lgamma, log, sqrt

import numpy as np
import pandas as pd

from _cont import normal_cdf, normal_inv
from _infer import (_betai, t_cdf, critical_value, p_value, _excel_p,
                    ALTERNATIVES, _check_alt)

__all__ = [
    "f_pdf", "f_cdf", "f_inv",
    "two_sample_t", "two_sample_t_from_data", "two_sample_z",
    "welch_df", "diff_interval",
    "paired_t", "paired_t_from_data", "paired_vs_independent",
    "pairing_variance",
    "two_proportion_test", "two_proportion_interval",
    "ab_sample_size", "ab_power",
    "permutation_test",
    "familywise_error", "simulate_pairwise_false_positives",
    "one_way_anova", "block_anova", "pairwise_comparisons",
    "two_sample_catalog",
]


# ---------------------------------------------------------------------------
# The F-distribution, without scipy
# ---------------------------------------------------------------------------

def f_pdf(x, d1: float, d2: float):
    """Density of the F-distribution with (d1, d2) degrees of freedom."""
    x = np.asarray(x, dtype=float)
    c = exp(lgamma((d1 + d2) / 2) - lgamma(d1 / 2) - lgamma(d2 / 2)
            + (d1 / 2) * log(d1 / d2))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = c * x ** (d1 / 2 - 1) * (1 + d1 * x / d2) ** (-(d1 + d2) / 2)
    return np.where(x > 0, out, 0.0)


def f_cdf(x: float, d1: float, d2: float) -> float:
    """P(F <= x). Excel: =F.DIST(x, d1, d2, TRUE)."""
    if x <= 0:
        return 0.0
    return _betai(d1 / 2.0, d2 / 2.0, d1 * x / (d1 * x + d2))


def f_inv(p: float, d1: float, d2: float) -> float:
    """The value with area p to its left. Excel: =F.INV(p, d1, d2); the
    upper-tail critical value at alpha is =F.INV.RT(alpha, d1, d2)."""
    lo, hi = 0.0, 1.0
    while f_cdf(hi, d1, d2) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f_cdf(mid, d1, d2) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Two independent means
# ---------------------------------------------------------------------------

def welch_df(s1: float, n1: int, s2: float, n2: int) -> float:
    """Welch-Satterthwaite degrees of freedom (Formula 10.11), NOT rounded.
    Software uses the fraction; Excel's Data Analysis tool rounds it to the
    nearest whole number; the textbook rounds down by hand."""
    a, b = s1 * s1 / n1, s2 * s2 / n2
    return (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))


def _se_df(s1, n1, s2, n2, pooled):
    if pooled:
        sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
        return sqrt(sp2 * (1 / n1 + 1 / n2)), n1 + n2 - 2, sp2
    return sqrt(s1 ** 2 / n1 + s2 ** 2 / n2), welch_df(s1, n1, s2, n2), None


def _words(reject, alpha, alternative, c, what):
    h1 = {"less": "<", "greater": ">", "two-sided": "!="}[alternative]
    if reject:
        return (f"Reject H0. At alpha = {alpha:g} the data are evidence that "
                f"{what} {h1} {c:g}.")
    return (f"Do not reject H0. At alpha = {alpha:g} the data are not "
            f"sufficient evidence that {what} {h1} {c:g}. (This does NOT "
            f"show that the difference is {c:g}.)")


def two_sample_t(xbar1: float, s1: float, n1: int,
                 xbar2: float, s2: float, n2: int,
                 diff0: float = 0.0, alternative: str = "two-sided",
                 alpha: float = 0.05, pooled: bool = False) -> pd.Series:
    """t-test for mu1 - mu2 from two INDEPENDENT samples, sigmas unknown.

    pooled=False (the default) is Welch's test: no assumption that the two
    population variances are equal. pooled=True assumes they are equal and
    uses the pooled variance with n1 + n2 - 2 df. When in doubt, use Welch;
    when the variances really are equal it gives nearly the same answer.
    """
    _check_alt(alternative)
    se, df, sp2 = _se_df(s1, n1, s2, n2, pooled)
    diff = xbar1 - xbar2
    stat = (diff - diff0) / se
    p = p_value(stat, alternative, df)
    tails = 2 if alternative == "two-sided" else 1
    crit = critical_value(alpha=alpha, tails=tails, df=df)
    reject = p <= alpha
    out = {
        "method": "pooled (equal variances)" if pooled else "Welch (unequal variances)",
        "x-bar 1": xbar1, "x-bar 2": xbar2,
        "difference x-bar1 - x-bar2": diff, "H0 difference": diff0,
        "alternative": alternative,
    }
    if pooled:
        out["pooled variance"] = sp2
    out.update({
        "standard error of the difference": se, "df": df,
        "test statistic t": stat,
        "critical value": (f"+/- {crit:.4f}" if tails == 2 else
                           (crit if alternative == "greater" else -crit)),
        "p-value": p, "alpha": alpha, "reject H0?": reject,
        "Excel tool": ("t-Test: Two-Sample Assuming Equal Variances" if pooled
                       else "t-Test: Two-Sample Assuming Unequal Variances"),
        "conclusion": _words(reject, alpha, alternative, diff0, "mu1 - mu2"),
    })
    return pd.Series(out)


def two_sample_t_from_data(x1, x2, diff0: float = 0.0,
                           alternative: str = "two-sided",
                           alpha: float = 0.05, pooled: bool = False
                           ) -> pd.Series:
    """two_sample_t computed from the raw data."""
    a = np.asarray(pd.Series(x1).dropna(), dtype=float)
    b = np.asarray(pd.Series(x2).dropna(), dtype=float)
    out = two_sample_t(a.mean(), a.std(ddof=1), len(a), b.mean(),
                       b.std(ddof=1), len(b), diff0, alternative, alpha,
                       pooled)
    head = pd.Series({"n1": len(a), "s1": a.std(ddof=1),
                      "n2": len(b), "s2": b.std(ddof=1)})
    return pd.concat([head, out])


def two_sample_z(xbar1: float, sigma1: float, n1: int,
                 xbar2: float, sigma2: float, n2: int,
                 diff0: float = 0.0, alternative: str = "two-sided",
                 alpha: float = 0.05) -> pd.Series:
    """z-test for mu1 - mu2 when both population sigmas are KNOWN."""
    _check_alt(alternative)
    se = sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2)
    diff = xbar1 - xbar2
    stat = (diff - diff0) / se
    p = p_value(stat, alternative)
    tails = 2 if alternative == "two-sided" else 1
    crit = critical_value(alpha=alpha, tails=tails)
    reject = p <= alpha
    return pd.Series({
        "difference x-bar1 - x-bar2": diff, "H0 difference": diff0,
        "standard error of the difference": se, "test statistic z": stat,
        "critical value": (f"+/- {crit:.4f}" if tails == 2 else
                           (crit if alternative == "greater" else -crit)),
        "p-value": p, "alpha": alpha, "reject H0?": reject,
        "Excel tool": "z-Test: Two Sample for Means",
        "conclusion": _words(reject, alpha, alternative, diff0, "mu1 - mu2"),
    })


def diff_interval(xbar1: float, s1: float, n1: int,
                  xbar2: float, s2: float, n2: int,
                  level: float = 0.95, pooled: bool = False,
                  sigma_known: bool = False) -> pd.Series:
    """Confidence interval for mu1 - mu2 from independent samples. If it
    excludes 0, a two-sided test at alpha = 1 - level rejects 'no
    difference'; if it includes 0, the test does not."""
    if sigma_known:
        se, df = sqrt(s1 ** 2 / n1 + s2 ** 2 / n2), None
    else:
        se, df, _ = _se_df(s1, n1, s2, n2, pooled)
    crit = critical_value(level, df=df)
    diff = xbar1 - xbar2
    lo, hi = diff - crit * se, diff + crit * se
    return pd.Series({"difference": diff, "standard error": se,
                      "df": "z" if df is None else df,
                      "critical value": crit, "margin of error": crit * se,
                      "lower": lo, "upper": hi,
                      "contains 0?": lo <= 0 <= hi})


# ---------------------------------------------------------------------------
# Paired samples
# ---------------------------------------------------------------------------

def paired_t(dbar: float, sd: float, n: int, diff0: float = 0.0,
             alternative: str = "two-sided", alpha: float = 0.05,
             level: float | None = None) -> pd.Series:
    """Matched-pairs t-test: a ONE-sample t-test on the differences
    d = x1 - x2, with n - 1 df (n = number of pairs)."""
    _check_alt(alternative)
    se = sd / sqrt(n)
    stat = (dbar - diff0) / se
    df = n - 1
    p = p_value(stat, alternative, df)
    tails = 2 if alternative == "two-sided" else 1
    crit = critical_value(alpha=alpha, tails=tails, df=df)
    reject = p <= alpha
    lvl = 1 - alpha if level is None else level
    ci_c = critical_value(lvl, df=df)
    return pd.Series({
        "pairs n": n, "mean difference d-bar": dbar, "s_d": sd,
        "standard error s_d/sqrt(n)": se, "df": df,
        "test statistic t": stat,
        "critical value": (f"+/- {crit:.4f}" if tails == 2 else
                           (crit if alternative == "greater" else -crit)),
        "p-value": p, "alpha": alpha, "reject H0?": reject,
        f"{lvl:.0%} interval for mu_d: lower": dbar - ci_c * se,
        f"{lvl:.0%} interval for mu_d: upper": dbar + ci_c * se,
        "Excel tool": "t-Test: Paired Two Sample for Means",
        "Excel p-value": _excel_p(stat, alternative, df),
        "conclusion": _words(reject, alpha, alternative, diff0,
                             "the mean difference mu_d"),
    })


def paired_t_from_data(x1, x2, diff0: float = 0.0,
                       alternative: str = "two-sided", alpha: float = 0.05,
                       level: float | None = None) -> pd.Series:
    """paired_t from two equal-length columns of matched observations."""
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired data need the same number of observations")
    d = a - b
    out = paired_t(d.mean(), d.std(ddof=1), len(d), diff0, alternative,
                   alpha, level)
    head = pd.Series({"correlation of the pairs r": np.corrcoef(a, b)[0, 1]})
    return pd.concat([head, out])


def paired_vs_independent(x1, x2, alternative: str = "two-sided"
                          ) -> pd.DataFrame:
    """The SAME numbers analysed both ways. When the pairs are positively
    correlated, the paired test's standard error is much smaller."""
    p = paired_t_from_data(x1, x2, 0, alternative)
    w = two_sample_t_from_data(x1, x2, 0, alternative)
    return pd.DataFrame({
        "paired (correct design)": {
            "difference": p["mean difference d-bar"],
            "standard error": p["standard error s_d/sqrt(n)"],
            "df": p["df"], "t": p["test statistic t"],
            "p-value": p["p-value"]},
        "independent (ignores the pairing)": {
            "difference": w["difference x-bar1 - x-bar2"],
            "standard error": w["standard error of the difference"],
            "df": w["df"], "t": w["test statistic t"],
            "p-value": w["p-value"]},
    })


def pairing_variance(x1, x2) -> pd.Series:
    """Var(d) = s1^2 + s2^2 - 2 r s1 s2. The last term is what pairing buys:
    the more the pairs move together (r near 1), the less the differences
    vary."""
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    r = np.corrcoef(a, b)[0, 1]
    return pd.Series({"s1^2": s1 ** 2, "s2^2": s2 ** 2, "r": r,
                      "-2 r s1 s2": -2 * r * s1 * s2,
                      "Var(d) = sum": s1 ** 2 + s2 ** 2 - 2 * r * s1 * s2,
                      "Var(d) directly": (a - b).var(ddof=1),
                      "Var(d) if the pairing is ignored (r = 0)":
                          s1 ** 2 + s2 ** 2})


# ---------------------------------------------------------------------------
# Two proportions and A/B tests
# ---------------------------------------------------------------------------

def two_proportion_test(x1: int, n1: int, x2: int, n2: int,
                        alternative: str = "two-sided", alpha: float = 0.05,
                        diff0: float = 0.0) -> pd.Series:
    """z-test for p1 - p2.

    With H0: p1 = p2 the standard error uses the POOLED proportion
    (x1 + x2)/(n1 + n2), because H0 says there is one common p. (For a
    nonzero claimed difference there is no common p, and the separate
    p-hats are used instead.)
    """
    _check_alt(alternative)
    p1, p2 = x1 / n1, x2 / n2
    if diff0 == 0:
        pbar = (x1 + x2) / (n1 + n2)
        se = sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    else:
        pbar = None
        se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    stat = (p1 - p2 - diff0) / se
    p = p_value(stat, alternative)
    tails = 2 if alternative == "two-sided" else 1
    crit = critical_value(alpha=alpha, tails=tails)
    reject = p <= alpha
    out = {"p-hat 1": p1, "p-hat 2": p2, "difference": p1 - p2,
           "H0 difference": diff0}
    if pbar is not None:
        out["pooled p-bar"] = pbar
    out.update({
        "standard error": se, "test statistic z": stat,
        "critical value": (f"+/- {crit:.4f}" if tails == 2 else
                           (crit if alternative == "greater" else -crit)),
        "p-value": p, "alpha": alpha, "reject H0?": reject,
        "Excel p-value": _excel_p(stat, alternative, None),
        "conclusion": _words(reject, alpha, alternative, diff0, "p1 - p2"),
    })
    return pd.Series(out)


def two_proportion_interval(x1: int, n1: int, x2: int, n2: int,
                            level: float = 0.95) -> pd.Series:
    """Confidence interval for p1 - p2, using each sample's own p-hat (no
    pooling: the interval makes no claim that p1 = p2)."""
    p1, p2 = x1 / n1, x2 / n2
    se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = critical_value(level)
    d = p1 - p2
    return pd.Series({"p-hat 1": p1, "p-hat 2": p2, "difference": d,
                      "standard error (unpooled)": se, "critical value z": z,
                      "margin of error": z * se, "lower": d - z * se,
                      "upper": d + z * se,
                      "contains 0?": d - z * se <= 0 <= d + z * se})


def ab_sample_size(p1: float, p2: float, alpha: float = 0.05,
                   power: float = 0.80, two_sided: bool = True
                   ) -> pd.Series:
    """Visitors needed PER ARM for a two-proportion z-test to detect a
    change from p1 to p2 with the given power. Round up."""
    za = critical_value(alpha=alpha, tails=2 if two_sided else 1)
    zb = normal_inv(power)
    pbar = (p1 + p2) / 2
    raw = ((za * sqrt(2 * pbar * (1 - pbar))
            + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
           / (p1 - p2) ** 2)
    return pd.Series({"baseline p1": p1, "target p2": p2,
                      "lift (points)": 100 * (p2 - p1),
                      "alpha": alpha, "power": power,
                      "raw n per arm": raw,
                      "n per arm (round UP)": int(ceil(raw - 1e-9)),
                      "total visitors": 2 * int(ceil(raw - 1e-9))},
                     dtype=object)


def ab_power(p1: float, p2: float, n: int, alpha: float = 0.05,
             two_sided: bool = True) -> float:
    """Approximate power of a two-proportion test with n per arm."""
    za = critical_value(alpha=alpha, tails=2 if two_sided else 1)
    pbar = (p1 + p2) / 2
    se0 = sqrt(2 * pbar * (1 - pbar) / n)
    se1 = sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    d = abs(p2 - p1)
    return 1 - normal_cdf((za * se0 - d) / se1)


# ---------------------------------------------------------------------------
# A permutation test: the most intuitive two-sample test
# ---------------------------------------------------------------------------

def permutation_test(x1, x2, reps: int = 20_000,
                     alternative: str = "two-sided",
                     seed: int = 20261022) -> pd.Series:
    """If the group labels made no difference (H0), any shuffle of the labels
    is as likely as the real one. Shuffle them many times and count how often
    the shuffled difference in means is at least as extreme as the observed
    one. That share IS a p-value, with no formula and no normality."""
    _check_alt(alternative)
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    pooled = np.concatenate([a, b])
    n1 = len(a)
    obs = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:n1].mean() - perm[n1:].mean()
    if alternative == "greater":
        p = float((diffs >= obs).mean())
    elif alternative == "less":
        p = float((diffs <= obs).mean())
    else:
        p = float((np.abs(diffs) >= abs(obs)).mean())
    out = pd.Series({"observed difference": obs, "shuffles": reps,
                     "permutation p-value": p})
    out.attrs["shuffled differences"] = diffs
    return out


# ---------------------------------------------------------------------------
# Many groups: multiple comparisons and one-way ANOVA (Chapter 11 slice)
# ---------------------------------------------------------------------------

def familywise_error(groups=(2, 3, 4, 5, 6, 8, 10), alpha: float = 0.05
                     ) -> pd.DataFrame:
    """Running every pairwise test among k groups, each at alpha. If all the
    true means are equal and the tests were independent, the chance of at
    least one false positive is 1 - (1 - alpha)^m, with m = k(k-1)/2."""
    rows = []
    for k in groups:
        m = k * (k - 1) // 2
        rows.append({"groups k": k, "pairwise tests m": m,
                     "P(at least one false positive)": 1 - (1 - alpha) ** m,
                     "Bonferroni alpha per test": alpha / m})
    return pd.DataFrame(rows).set_index("groups k")


def simulate_pairwise_false_positives(k: int = 4, n: int = 20,
                                      reps: int = 5_000, alpha: float = 0.05,
                                      seed: int = 20261022) -> pd.Series:
    """Draw k groups from the SAME normal population (every H0 true), run all
    pairwise Welch t-tests and one ANOVA, and count how often each approach
    reports at least one significant difference."""
    rng = np.random.default_rng(seed)
    any_pair = 0
    anova_rej = 0
    for _ in range(reps):
        g = rng.normal(0, 1, (k, n))
        m = g.mean(axis=1)
        v = g.var(axis=1, ddof=1)
        hit = False
        for i, j in combinations(range(k), 2):
            se = sqrt(v[i] / n + v[j] / n)
            t = (m[i] - m[j]) / se
            df = welch_df(sqrt(v[i]), n, sqrt(v[j]), n)
            if 2 * (1 - t_cdf(abs(t), df)) <= alpha:
                hit = True
                break
        any_pair += hit
        ssb = n * ((m - m.mean()) ** 2).sum()
        ssw = ((n - 1) * v).sum()
        F = (ssb / (k - 1)) / (ssw / (k * n - k))
        anova_rej += (1 - f_cdf(F, k - 1, k * n - k)) <= alpha
    m_tests = k * (k - 1) // 2
    return pd.Series({"groups": k, "per group": n, "simulations": reps,
                      "pairwise tests each time": m_tests,
                      "share with >= 1 'significant' pair": any_pair / reps,
                      "share where ANOVA rejects": anova_rej / reps,
                      "nominal alpha": alpha}, dtype=object)


def one_way_anova(groups: dict, alpha: float = 0.05):
    """One-way ANOVA laid out as Excel's 'Anova: Single Factor' prints it.
    Returns (summary table, ANOVA table).

    F = (variation BETWEEN group means per df) / (variation WITHIN groups
    per df). If every population mean is equal, F is usually near 1; a large
    F says the group means differ by more than the noise can explain.
    """
    names = list(groups)
    data = [np.asarray(groups[g], dtype=float) for g in names]
    allx = np.concatenate(data)
    grand = allx.mean()
    k, N = len(data), len(allx)
    summary = pd.DataFrame({
        "count": [len(d) for d in data], "sum": [d.sum() for d in data],
        "average": [d.mean() for d in data],
        "variance": [d.var(ddof=1) for d in data]}, index=names)
    ssb = sum(len(d) * (d.mean() - grand) ** 2 for d in data)
    ssw = sum(((d - d.mean()) ** 2).sum() for d in data)
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw
    p = 1 - f_cdf(F, dfb, dfw)
    table = pd.DataFrame({
        "SS": [ssb, ssw, ssb + ssw], "df": [dfb, dfw, N - 1],
        "MS": [msb, msw, np.nan], "F": [F, np.nan, np.nan],
        "P-value": [p, np.nan, np.nan],
        "F crit": [f_inv(1 - alpha, dfb, dfw), np.nan, np.nan]},
        index=["Between Groups", "Within Groups", "Total"])
    return summary, table


def block_anova(frame: pd.DataFrame, y: str, factor: str, block: str,
                alpha: float = 0.05) -> pd.DataFrame:
    """ANOVA with a blocking variable (an additive two-factor model), for a
    BALANCED design (the same number of observations in every
    factor-by-block cell).

    The block soaks up variation we already know about (route, store, day),
    so it is no longer counted as noise. The factor's F-test then compares
    its variation with a smaller error term: blocking is pairing for more
    than two groups.
    """
    counts = frame.groupby([factor, block]).size()
    if counts.nunique() != 1:
        raise ValueError("block_anova needs a balanced design")
    v = frame[y].astype(float)
    grand = v.mean()
    sst = ((v - grand) ** 2).sum()
    ssf = (frame.groupby(factor)[y].agg(["mean", "size"])
           .pipe(lambda g: (g["size"] * (g["mean"] - grand) ** 2).sum()))
    ssb = (frame.groupby(block)[y].agg(["mean", "size"])
           .pipe(lambda g: (g["size"] * (g["mean"] - grand) ** 2).sum()))
    sse = sst - ssf - ssb
    a, b, N = frame[factor].nunique(), frame[block].nunique(), len(frame)
    dff, dfb, dfe = a - 1, b - 1, N - a - b + 1
    mse = sse / dfe
    rows = {}
    for name, ss, df in ((f"{factor} (factor)", ssf, dff),
                         (f"{block} (block)", ssb, dfb)):
        F = (ss / df) / mse
        rows[name] = {"SS": ss, "df": df, "MS": ss / df, "F": F,
                      "P-value": 1 - f_cdf(F, df, dfe),
                      "F crit": f_inv(1 - alpha, df, dfe)}
    rows["Error"] = {"SS": sse, "df": dfe, "MS": mse, "F": np.nan,
                     "P-value": np.nan, "F crit": np.nan}
    rows["Total"] = {"SS": sst, "df": N - 1, "MS": np.nan, "F": np.nan,
                     "P-value": np.nan, "F crit": np.nan}
    return pd.DataFrame(rows).T


def pairwise_comparisons(groups: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Every pairwise Welch t-test, with a Bonferroni-adjusted decision
    (compare each p-value with alpha / m). Your textbook's Tukey-Kramer
    procedure does the same job with a different table; both hold the
    family-wide false positive rate near alpha."""
    names = list(groups)
    m = len(names) * (len(names) - 1) // 2
    rows = []
    for a, b in combinations(names, 2):
        r = two_sample_t_from_data(groups[a], groups[b])
        rows.append({"pair": f"{a} - {b}",
                     "difference": r["difference x-bar1 - x-bar2"],
                     "p-value (one test)": r["p-value"],
                     f"significant at {alpha:g}?": r["p-value"] <= alpha,
                     f"significant at {alpha:g}/{m} (Bonferroni)?":
                         r["p-value"] <= alpha / m})
    return pd.DataFrame(rows).set_index("pair")


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def two_sample_catalog() -> pd.DataFrame:
    """Which test, which standard error, which df, which Excel tool."""
    rows = [
        ("Two means, independent, sigmas known (rare)",
         "z", "sqrt(s1^2/n1 + s2^2/n2) with sigmas", "-",
         "z-Test: Two Sample for Means"),
        ("Two means, independent, sigmas unknown (DEFAULT)",
         "t (Welch)", "sqrt(s1^2/n1 + s2^2/n2)", "Welch formula",
         "t-Test: Two-Sample Assuming Unequal Variances"),
        ("Two means, independent, variances assumed equal",
         "t (pooled)", "sqrt(sp^2 (1/n1 + 1/n2))", "n1 + n2 - 2",
         "t-Test: Two-Sample Assuming Equal Variances"),
        ("Two means, paired (same units twice)",
         "t on d = x1 - x2", "s_d / sqrt(n)", "n - 1 (pairs)",
         "t-Test: Paired Two Sample for Means"),
        ("Two proportions, test of p1 = p2",
         "z", "sqrt(p-bar (1 - p-bar)(1/n1 + 1/n2)), pooled", "-",
         "NORM.S.DIST on the z"),
        ("Two proportions, confidence interval",
         "z", "sqrt(p1(1-p1)/n1 + p2(1-p2)/n2), unpooled", "-",
         "NORM.S.INV for the critical value"),
        ("Three or more means",
         "F (one-way ANOVA)", "MS within", "k - 1 and N - k",
         "Anova: Single Factor"),
    ]
    return pd.DataFrame(rows, columns=["situation", "statistic",
                                       "standard error", "df",
                                       "Excel"]).set_index("situation")
