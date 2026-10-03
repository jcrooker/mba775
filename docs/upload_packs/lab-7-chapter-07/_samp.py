"""Sampling and sampling distribution helpers for MBA 775, Chapter 7.

Chapters 5 and 6 described data generating mechanisms: rules that produce
values. This chapter turns the telescope around. We hold ONE sample, compute
ONE statistic from it, and ask how that number would have varied had we drawn
a different sample. The answer is a probability distribution of its own, the
sampling distribution, and its standard deviation, the standard error, is the
most important number in the second half of the course.

The functions here do three kinds of work:

  * Draw samples by each design (simple random, systematic, stratified,
    cluster) and estimate from them correctly, including the weighting a
    stratified sample needs.
  * Enumerate or simulate a sampling distribution, so it can be looked at
    instead of taken on faith.
  * Answer questions about a sample mean or a sample proportion, showing the
    standard error, the z-score, and the Excel formula, and, for counts,
    the exact answer beside the normal approximation.

Nothing here needs scipy.

You do not need to read this file to do the coursework.
"""

from __future__ import annotations

from itertools import combinations, product
from math import comb, sqrt

import numpy as np
import pandas as pd

from _cont import normal_cdf, normal_inv, normal_area
from _dist import binomial_table, hypergeometric_pmf, probability_of

__all__ = [
    "simple_random_sample", "systematic_sample", "proportional_allocation",
    "stratified_sample", "stratified_estimate", "cluster_sample",
    "design_comparison",
    "all_sample_errors", "enumerate_sample_means",
    "fpc", "se_mean", "se_proportion", "fpc_table",
    "xbar_probability", "xbar_interval", "interval_table",
    "proportion_question", "simulate_sample_means", "tail_check",
    "bootstrap", "bootstrap_interval", "selection_bias_demo",
]


def _fmt(v: float) -> str:
    return f"{v:.6g}"


# ---------------------------------------------------------------------------
# Sampling designs
# ---------------------------------------------------------------------------

def simple_random_sample(population, n: int, seed: int | None = None):
    """n members drawn WITHOUT replacement, every member equally likely.

    Without replacement is what "a simple random sample of 30 members"
    means: nobody is chosen twice. (Excel's Sampling tool draws WITH
    replacement, which is why it can hand you the same ID twice.)
    """
    rng = np.random.default_rng(seed)
    pop = pd.Series(population) if not isinstance(population, pd.DataFrame) \
        else population
    if n > len(pop):
        raise ValueError(f"Cannot draw {n} without replacement from {len(pop)}.")
    idx = rng.choice(len(pop), size=n, replace=False)
    return pop.iloc[np.sort(idx)]


def systematic_sample(population, n: int, start: int | None = None,
                      seed: int | None = None):
    """Every k-th member, in the order the population is listed, starting
    from a random position among the first k. k = N // n.

    The list order is kept on purpose. Systematic sampling inherits whatever
    pattern the list has, which is exactly where periodicity comes from.
    """
    pop = pd.Series(population) if not isinstance(population, pd.DataFrame) \
        else population
    N = len(pop)
    k = N // n
    if k < 1:
        raise ValueError("The sample cannot be larger than the population.")
    if start is None:
        start = int(np.random.default_rng(seed).integers(0, k))
    positions = start + k * np.arange(n)
    out = pop.iloc[positions]
    out.attrs.update({"k": k, "start (0-based)": start})
    return out


def proportional_allocation(counts: dict, n: int) -> pd.DataFrame:
    """How many to sample from each stratum so the sample mirrors the
    population: n times each stratum's share. Rounded so the total is
    exactly n (largest remainders get the leftover units)."""
    names = list(counts)
    N = sum(counts.values())
    raw = np.array([n * counts[s] / N for s in names])
    alloc = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - alloc))[: n - alloc.sum()]:
        alloc[i] += 1
    return pd.DataFrame({
        "population count": [counts[s] for s in names],
        "share of population": [counts[s] / N for s in names],
        "n x share": raw,
        "sample size": alloc,
    }, index=pd.Index(names, name="stratum"))


def stratified_sample(frame: pd.DataFrame, stratum: str, sizes: dict,
                      seed: int | None = None) -> pd.DataFrame:
    """A simple random sample, without replacement, of the given size from
    each stratum."""
    rng = np.random.default_rng(seed)
    parts = []
    for s, n_h in sizes.items():
        block = frame[frame[stratum] == s]
        idx = rng.choice(len(block), size=n_h, replace=False)
        parts.append(block.iloc[np.sort(idx)])
    return pd.concat(parts)


def stratified_estimate(sample: pd.DataFrame, stratum: str, value: str,
                        population_counts: dict) -> pd.DataFrame:
    """Estimate the population mean from a stratified sample.

    The estimator is the population-share-weighted average of the STRATUM
    MEANS:  sum over strata of W_h x (mean of stratum h in the sample),
    where W_h is stratum h's share of the population.

    Equivalently, each sampled observation gets weight W_h / n_h. Weighting
    each observation by W_h alone is a common mistake: it double-counts the
    strata you happened to sample more heavily. The `attrs` carry the
    correct estimate, the unweighted sample mean, and that mistaken version,
    so the three can be compared.
    """
    N = sum(population_counts.values())
    rows = []
    for s, N_h in population_counts.items():
        y = sample.loc[sample[stratum] == s, value]
        rows.append({"stratum": s, "N_h": N_h, "W_h = N_h / N": N_h / N,
                     "n_h": len(y), "stratum sample mean": y.mean(),
                     "W_h x mean": N_h / N * y.mean()})
    t = pd.DataFrame(rows).set_index("stratum")
    w_obs = sample[stratum].map({s: population_counts[s] / N
                                 for s in population_counts})
    t.attrs.update({
        "stratified estimate": float(t["W_h x mean"].sum()),
        "unweighted sample mean": float(sample[value].mean()),
        "W_h per observation (wrong)":
            float((sample[value] * w_obs).sum() / w_obs.sum()),
    })
    return t


def cluster_sample(frame: pd.DataFrame, cluster: str, n_clusters: int,
                   seed: int | None = None) -> pd.DataFrame:
    """One-stage cluster sample: choose n_clusters clusters at random and keep
    EVERY member of each chosen cluster."""
    rng = np.random.default_rng(seed)
    names = frame[cluster].unique()
    chosen = rng.choice(names, size=n_clusters, replace=False)
    out = frame[frame[cluster].isin(chosen)]
    out.attrs["clusters chosen"] = sorted(chosen.tolist())
    return out


def design_comparison(frame: pd.DataFrame, value: str, designs: dict,
                      reps: int = 2_000, seed: int = 20261003) -> pd.DataFrame:
    """Run each sampling design many times and summarise its estimates.

    `designs` maps a label to a function f(frame, rng_seed) -> estimate.
    For each design: the average estimate, its bias against the true
    population mean, the standard deviation of the estimates (the design's
    standard error), and the root mean squared error, which combines both.
    """
    truth = float(frame[value].mean())
    rng = np.random.default_rng(seed)
    rows = []
    for label, f in designs.items():
        est = np.array([f(frame, int(rng.integers(1 << 31)))
                        for _ in range(reps)])
        rows.append({"design": label,
                     "average estimate": est.mean(),
                     "bias": est.mean() - truth,
                     "standard error": est.std(ddof=1),
                     "RMSE": sqrt(((est - truth) ** 2).mean())})
    out = pd.DataFrame(rows).set_index("design")
    out.attrs["population mean"] = truth
    return out


# ---------------------------------------------------------------------------
# Sampling error, by enumeration
# ---------------------------------------------------------------------------

def all_sample_errors(population, n: int) -> pd.DataFrame:
    """Every possible sample of size n (without replacement) from a small
    population, with its mean and its sampling error x-bar minus mu."""
    pop = list(population)
    mu = float(np.mean(pop))
    if comb(len(pop), n) > 200_000:
        raise ValueError("Too many samples to list.")
    rows = []
    for s in combinations(range(len(pop)), n):
        vals = [pop[i] for i in s]
        m = float(np.mean(vals))
        rows.append({"sample": ", ".join(f"{v:g}" for v in vals),
                     "sample mean": m, "sampling error": m - mu})
    out = pd.DataFrame(rows)
    out.attrs.update({"mu": mu, "samples": len(out)})
    return out


def enumerate_sample_means(values, n: int, probabilities=None
                           ) -> pd.DataFrame:
    """The exact sampling distribution of x-bar for samples of size n drawn
    WITH replacement from a small discrete population: every ordered sample,
    its probability, and the resulting distribution of the sample mean."""
    values = list(values)
    if probabilities is None:
        probabilities = [1 / len(values)] * len(values)
    dist = {}
    for combo in product(range(len(values)), repeat=n):
        m = round(float(np.mean([values[i] for i in combo])), 10)
        p = float(np.prod([probabilities[i] for i in combo]))
        dist[m] = dist.get(m, 0.0) + p
    out = pd.DataFrame({"P(x-bar)": pd.Series(dist)}).sort_index()
    out.index.name = "x-bar"
    out["number of samples"] = (out["P(x-bar)"] * len(values) ** n).round(6)
    mu = float(np.dot(values, probabilities))
    var = float(np.dot((np.array(values) - mu) ** 2, probabilities))
    m_x = float((out.index * out["P(x-bar)"]).sum())
    v_x = float((((out.index - m_x) ** 2) * out["P(x-bar)"]).sum())
    out.attrs.update({"mu": mu, "sigma": sqrt(var), "n": n,
                      "samples": len(values) ** n,
                      "mean of x-bar": m_x, "sd of x-bar": sqrt(v_x),
                      "sigma / sqrt(n)": sqrt(var) / sqrt(n)})
    return out


# ---------------------------------------------------------------------------
# Standard errors
# ---------------------------------------------------------------------------

def fpc(n: int, N: int | None) -> float:
    """The finite population correction sqrt((N - n)/(N - 1)). Returns 1 when
    N is None (an effectively infinite population)."""
    if N is None:
        return 1.0
    if not 0 < n <= N:
        raise ValueError("Need 0 < n <= N.")
    return sqrt((N - n) / (N - 1)) if N > 1 else 0.0


def se_mean(sigma: float, n: int, N: int | None = None) -> float:
    """Standard error of the mean, sigma / sqrt(n), times the finite
    population correction when N is given."""
    return sigma / sqrt(n) * fpc(n, N)


def se_proportion(p: float, n: int, N: int | None = None) -> float:
    """Standard error of the proportion, sqrt(p(1 - p)/n), times the finite
    population correction when N is given."""
    return sqrt(p * (1 - p) / n) * fpc(n, N)


def fpc_table(sigma: float, N: int, sizes) -> pd.DataFrame:
    """The standard error with and without the finite population correction,
    for several sample sizes from a population of N. At n = N the corrected
    standard error is zero: a census has no sampling error."""
    rows = []
    for n in sizes:
        rows.append({"n": n, "n / N": n / N,
                     "sigma / sqrt(n)": sigma / sqrt(n),
                     "FPC sqrt((N-n)/(N-1))": fpc(n, N),
                     "corrected standard error": se_mean(sigma, n, N)})
    return pd.DataFrame(rows).set_index("n")


# ---------------------------------------------------------------------------
# Questions about a sample mean
# ---------------------------------------------------------------------------

def xbar_probability(mu: float, sigma: float, n: int, lower=None, upper=None,
                     N: int | None = None, explain: bool = False):
    """P(lower < x-bar < upper) when x-bar is normal with mean mu and standard
    error sigma/sqrt(n) (corrected if N is given).

    The z-score divides by the STANDARD ERROR, not by sigma. Dividing by
    sigma answers a question about one individual, not about an average.
    """
    se = se_mean(sigma, n, N)
    prob = normal_area(mu, se, lower, upper)
    if not explain:
        return prob
    out = normal_area(mu, se, lower, upper, explain=True)
    out = out.rename({"event": "event (x-bar)"})
    head = pd.Series({"standard error": se,
                      "finite population correction": fpc(n, N)})
    return pd.concat([head, out])


def xbar_interval(mu: float, sigma: float, n: int, level: float = 0.95,
                  N: int | None = None) -> pd.Series:
    """The symmetric interval around mu that contains `level` of all sample
    means: mu +/- z x standard error."""
    z = normal_inv(0.5 + level / 2)
    se = se_mean(sigma, n, N)
    return pd.Series({"level": level, "z": z, "standard error": se,
                      "margin z x SE": z * se,
                      "lower": mu - z * se, "upper": mu + z * se})


def interval_table(mu: float, sigma: float, sizes, level: float = 0.95
                   ) -> pd.DataFrame:
    """The 95% interval of sample means at several sample sizes."""
    rows = []
    for n in sizes:
        r = xbar_interval(mu, sigma, n, level)
        rows.append({"n": n, "standard error": r["standard error"],
                     "lower": r["lower"], "upper": r["upper"],
                     "width": r["upper"] - r["lower"]})
    return pd.DataFrame(rows).set_index("n")


# ---------------------------------------------------------------------------
# Questions about a sample proportion
# ---------------------------------------------------------------------------

_COUNT_EDGES = {
    # phrase: (whole numbers included as an interval of counts, corrected edges)
    "at most": lambda x: ((None, x), (None, x + 0.5)),
    "fewer than": lambda x: ((None, x - 1), (None, x - 0.5)),
    "at least": lambda x: ((x, None), (x - 0.5, None)),
    "more than": lambda x: ((x + 1, None), (x + 0.5, None)),
    "exactly": lambda x: ((x, x), (x - 0.5, x + 0.5)),
}


def proportion_question(p: float, n: int, phrase: str, x: int,
                        N: int | None = None) -> pd.Series:
    """A question about the NUMBER of successes in a sample ("84 or fewer
    of 200"), answered three ways.

    - exact: binomial, or hypergeometric when a finite N is given
      (R = round(pN) successes in the population);
    - normal on the proportion scale WITH the continuity correction
      (half a count is 0.5/n on the proportion scale);
    - normal WITHOUT it, as the textbook does in Chapter 7.

    The standard error includes the finite population correction when N is
    given.
    """
    key = phrase.strip().lower()
    (lo_c, hi_c), (lo_x, hi_x) = _COUNT_EDGES[key](x)
    se = se_proportion(p, n, N)
    # exact
    if N is None:
        tbl = binomial_table(n, p)
        exact = probability_of(tbl, key, x)
        exact_label = "binomial"
    else:
        R = round(p * N)
        pmf = [hypergeometric_pmf(k, n, R, N) for k in range(n + 1)]
        ks = range(n + 1)
        lo = lo_c if lo_c is not None else 0
        hi = hi_c if hi_c is not None else n
        exact = float(sum(pmf[k] for k in ks if lo <= k <= hi))
        exact_label = f"hypergeometric (N = {N}, R = {R})"
    corrected = normal_area(p, se, None if lo_x is None else lo_x / n,
                            None if hi_x is None else hi_x / n)
    naive_lo = None if lo_x is None else x / n
    naive_hi = None if hi_x is None else x / n
    if key == "exactly":
        naive_lo = naive_hi = x / n
    uncorrected = normal_area(p, se, naive_lo, naive_hi)
    return pd.Series({
        "question": f"{key} {x} of {n}",
        "p-hat at x": x / n,
        "standard error": se,
        "z, uncorrected": (x / n - p) / se,
        f"exact ({exact_label})": exact,
        "normal, with continuity correction": corrected,
        "normal, no correction": uncorrected,
    })


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_sample_means(population, n: int, reps: int = 5_000,
                          seed: int = 20261003,
                          replace: bool = True) -> pd.Series:
    """Draw `reps` FRESH samples of size n from the population and return
    their means. Each sample is drawn from the population itself, never from
    an earlier sample: that is the difference between a sampling
    distribution and a bootstrap."""
    pop = np.asarray(population, dtype=float)
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, len(pop), size=(reps, n)) if replace else
           np.array([rng.choice(len(pop), n, replace=False)
                     for _ in range(reps)]))
    return pd.Series(pop[idx].mean(axis=1), name=f"means, n = {n}")


def tail_check(means, mu: float, se: float, z: float = 1.96) -> pd.Series:
    """Share of simulated sample means beyond mu +/- z x SE on each side,
    against the normal model's 2.5% per tail. Unequal tails mean the
    sampling distribution is still skewed: n is not yet large enough."""
    m = np.asarray(means, dtype=float)
    return pd.Series({"below mu - 1.96 SE": float((m < mu - z * se).mean()),
                      "above mu + 1.96 SE": float((m > mu + z * se).mean()),
                      "normal model, each tail": 1 - normal_cdf(z)})


def bootstrap(sample, statistic=np.mean, reps: int = 10_000,
              seed: int = 20261003) -> pd.Series:
    """Resample THE ONE SAMPLE YOU HAVE, with replacement, at its own size,
    and recompute the statistic each time. The spread of the results
    estimates the standard error without a formula."""
    x = np.asarray(sample, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    return pd.Series([statistic(x[i]) for i in idx], name="bootstrap")


def bootstrap_interval(boot, level: float = 0.95) -> pd.Series:
    """The middle `level` of the bootstrap statistics (percentile method)."""
    b = np.sort(np.asarray(boot, dtype=float))
    a = (1 - level) / 2
    return pd.Series({"bootstrap standard error": float(b.std(ddof=1)),
                      "lower": float(np.quantile(b, a)),
                      "upper": float(np.quantile(b, 1 - a))})


def selection_bias_demo(ratings, review_prob: dict, sizes,
                        seed: int = 20261003) -> pd.DataFrame:
    """What a self-selected sample does as it grows.

    `ratings` is the population of customer ratings; `review_prob` maps each
    rating to the chance that a customer with that rating writes a review.
    For each sample size, the mean of the reviews actually written is
    compared with the population mean. The standard error shrinks; the bias
    does not.
    """
    r = np.asarray(ratings)
    rng = np.random.default_rng(seed)
    truth = float(r.mean())
    rows = []
    for n in sizes:
        pool = rng.choice(r, size=n * 20, replace=True)
        keep = rng.random(len(pool)) < np.vectorize(review_prob.get)(pool)
        rev = pool[keep][:n]
        rows.append({"reviews": len(rev), "mean of reviews": rev.mean(),
                     "standard error s/sqrt(n)": rev.std(ddof=1) / sqrt(len(rev)),
                     "population mean": truth,
                     "bias": rev.mean() - truth})
    return pd.DataFrame(rows).set_index("reviews")
