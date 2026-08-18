"""Chart and table helpers for MBA 775.

These replace the R functions in LIB_Data_Visualizations.R. Each one produces
a presentation-quality figure from a single call, so the code stays out of the
way and the discussion can be about what the picture shows.

You do not need to read this file to do the coursework. It is here so the
charts look consistent and so nobody has to learn matplotlib to make one.

Every function that draws also RETURNS the numbers behind the picture, because
a chart you cannot check is a chart you cannot defend.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "frequency_table", "contingency_table",
    "histogram", "binned_bar", "bar_chart", "pareto_chart", "pie_chart",
    "ogive", "heatmap", "stem_and_leaf", "scatter", "time_series",
    "UNLV_SCARLET", "UNLV_GRAY",
]

UNLV_SCARLET = "#a03123"
UNLV_GRAY = "#666666"
UNLV_LIGHT = "#f7f2f1"

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": UNLV_GRAY,
    "axes.titlecolor": UNLV_SCARLET,
    "axes.titlesize": 12,
    "axes.titleweight": "600",
    "axes.labelcolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": UNLV_GRAY,
    "ytick.color": UNLV_GRAY,
    "grid.color": "#e6e6e6",
    "font.size": 10,
})


def _finish(ax, title, xlab, ylab, grid_axis="y"):
    if title:
        ax.set_title(title)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    ax.grid(axis=grid_axis, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def frequency_table(x, bins=None, cumulative=False, pct=True, decimals=1):
    """Count how many observations fall in each class.

    `x` may be numeric (it will be binned) or categorical (each distinct value
    is its own class). Missing values are reported separately rather than
    silently dropped -- how many observations you *lost* is part of the result.

    Returns a DataFrame with Category, Frequency, and optionally PCT and
    CUM_PCT.
    """
    s = pd.Series(x)
    n_missing = int(s.isna().sum())
    s = s.dropna()

    if pd.api.types.is_numeric_dtype(s):
        if bins is None:
            bins = _sturges_bins(s)
        cats = pd.cut(s, bins=bins, include_lowest=True)
        counts = cats.value_counts().sort_index()
        labels = [str(i) for i in counts.index]
    else:
        counts = s.value_counts().sort_index()
        labels = list(counts.index)

    out = pd.DataFrame({"Category": labels, "Frequency": counts.to_numpy()})
    if pct:
        out["PCT"] = (100 * out["Frequency"] / out["Frequency"].sum()).round(decimals)
    if cumulative:
        out["CUM_FREQ"] = out["Frequency"].cumsum()
        if pct:
            out["CUM_PCT"] = out["PCT"].cumsum().round(decimals)

    if n_missing:
        print(f"NOTE: {n_missing:,} observation(s) had no value and are excluded "
              f"from this table. The percentages below are of the "
              f"{len(s):,} that did.")
    return out


def _sturges_bins(s):
    """Number of classes by the 2^k >= n rule, the textbook's rule of thumb."""
    n = len(s)
    k = 1
    while 2 ** k < n:
        k += 1
    return max(5, min(k, 20))


def contingency_table(x, y, x_bins=6, y_bins=6, pct=False, x_name="x", y_name="y"):
    """Cross-tabulate two variables, binning them first if they are numeric."""
    xs, ys = pd.Series(x), pd.Series(y)
    keep = xs.notna() & ys.notna()
    xs, ys = xs[keep], ys[keep]

    if pd.api.types.is_numeric_dtype(xs):
        xs = pd.cut(xs, bins=x_bins, include_lowest=True)
    if pd.api.types.is_numeric_dtype(ys):
        ys = pd.cut(ys, bins=y_bins, include_lowest=True)

    table = pd.crosstab(ys, xs)
    table.index.name = y_name
    table.columns.name = x_name
    if pct:
        table = (100 * table / table.to_numpy().sum()).round(1)
    return table


# ---------------------------------------------------------------------------
# Distribution charts
# ---------------------------------------------------------------------------

def histogram(x, bins=None, title=None, xlab=None, ylab="Frequency",
              show_mean=True, show_median=True, figsize=(9, 4.5)):
    """Histogram with the mean and median marked, since where those sit
    relative to each other is what tells you the distribution is skewed."""
    s = pd.Series(x).dropna()
    if bins is None:
        bins = _sturges_bins(s)

    fig, ax = plt.subplots(figsize=figsize)
    counts, edges, _ = ax.hist(s, bins=bins, color=UNLV_SCARLET,
                               alpha=0.85, edgecolor="white")

    if show_mean:
        ax.axvline(s.mean(), color="black", linewidth=1.6,
                   label=f"mean = {s.mean():.2f}")
    if show_median:
        ax.axvline(s.median(), color="black", linewidth=1.6, linestyle="--",
                   label=f"median = {s.median():.2f}")
    if show_mean or show_median:
        ax.legend(frameon=False, fontsize=9)

    _finish(ax, title, xlab, ylab)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({
        "lower": edges[:-1].round(3),
        "upper": edges[1:].round(3),
        "count": counts.astype(int),
    })


def binned_bar(x, y, bins=6, title=None, xlab=None, ylab=None,
               statistic="mean", figsize=(9, 4.5)):
    """Average (or median) of `y` within equal-width bins of `x`.

    Also returns the count in each bin. Read those counts before believing a
    bar: a tall bar over three observations is not evidence.
    """
    df = pd.DataFrame({"x": pd.Series(x), "y": pd.Series(y)}).dropna()
    df["bin"] = pd.cut(df["x"], bins=bins, include_lowest=True)
    grouped = df.groupby("bin", observed=False)["y"]
    heights = getattr(grouped, statistic)()
    counts = grouped.size()

    fig, ax = plt.subplots(figsize=figsize)
    positions = range(len(heights))
    ax.bar(positions, heights.to_numpy(), color=UNLV_SCARLET, alpha=0.85)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(i) for i in heights.index], rotation=45,
                       ha="right", fontsize=8)
    for p, (h, c) in enumerate(zip(heights.to_numpy(), counts.to_numpy())):
        if np.isfinite(h):
            ax.text(p, h, f"n={c}", ha="center", va="bottom", fontsize=7.5,
                    color=UNLV_GRAY)
    _finish(ax, title, xlab, ylab)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({statistic: heights.round(2), "n": counts})


def bar_chart(categories, values, title=None, xlab=None, ylab="Count",
              horizontal=False, figsize=(9, 4.5)):
    """Plain bar chart for categorical counts."""
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        ax.barh(list(categories), list(values), color=UNLV_SCARLET, alpha=0.85)
        _finish(ax, title, ylab, xlab, grid_axis="x")
    else:
        ax.bar(list(categories), list(values), color=UNLV_SCARLET, alpha=0.85)
        _finish(ax, title, xlab, ylab)
        if max(len(str(c)) for c in categories) > 8:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"Category": list(categories), "Value": list(values)})


def pareto_chart(x, title=None, xlab=None, ylab="Frequency", figsize=(9, 5)):
    """Bars sorted from most to least frequent, with a cumulative percentage
    line. The point is to see how few categories account for how much."""
    s = pd.Series(x).dropna()
    counts = s.value_counts().sort_values(ascending=False)
    cum_pct = 100 * counts.cumsum() / counts.sum()

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(counts))
    ax.bar(positions, counts.to_numpy(), color=UNLV_SCARLET, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in counts.index], rotation=30,
                       ha="right", fontsize=8.5)
    _finish(ax, title, xlab, ylab)

    ax2 = ax.twinx()
    ax2.plot(positions, cum_pct.to_numpy(), color="black", marker="o",
             linewidth=1.6, markersize=4)
    ax2.set_ylabel("Cumulative percent")
    ax2.set_ylim(0, 105)
    ax2.grid(False)
    ax2.spines[["top"]].set_visible(False)

    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"Category": counts.index, "Frequency": counts.to_numpy(),
                         "CUM_PCT": cum_pct.round(1).to_numpy()})


def pie_chart(labels, values, title=None, figsize=(7, 5.5)):
    """Pie chart in a scarlet gradient.

    Use sparingly. Beyond three or four slices people cannot compare angles,
    and a bar chart will communicate the same thing more accurately.
    """
    labels, values = list(labels), list(values)
    shades = [mpl.colors.to_hex(c) for c in
              plt.cm.Reds(np.linspace(0.45, 0.9, len(values)))]

    fig, ax = plt.subplots(figsize=figsize)
    total = sum(values)
    ax.pie(values, labels=labels, colors=shades, startangle=90,
           autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100)):,})",
           textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white"})
    ax.set_title(title or "", color=UNLV_SCARLET, fontweight="600")
    ax.axis("equal")
    fig.tight_layout()
    plt.show()

    if len(values) > 4:
        print(f"NOTE: {len(values)} slices. Readers cannot reliably compare "
              f"more than three or four angles -- consider a bar chart.")
    return pd.DataFrame({"Category": labels, "Frequency": values,
                         "PCT": (100 * np.array(values) / total).round(1)})


def ogive(x, bins=None, title=None, xlab=None, figsize=(9, 4.5)):
    """Cumulative relative frequency curve.

    A point on the curve reads: this percent of observations are at or below
    that value on the horizontal axis.
    """
    s = pd.Series(x).dropna()
    if bins is None:
        bins = _sturges_bins(s)
    counts, edges = np.histogram(s, bins=bins)
    cum_pct = 100 * np.cumsum(counts) / counts.sum()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(edges[1:], cum_pct, color=UNLV_SCARLET, marker="o",
            linewidth=1.8, markersize=4)
    ax.axhline(50, color=UNLV_GRAY, linewidth=0.9, linestyle=":")
    ax.set_ylim(0, 105)
    _finish(ax, title, xlab, "Cumulative percent")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"upper_bound": edges[1:].round(3),
                         "count": counts, "CUM_PCT": cum_pct.round(1)})


def heatmap(table, title=None, xlab=None, ylab=None, figsize=(8, 6)):
    """Shade a contingency table so the concentrations are visible."""
    fig, ax = plt.subplots(figsize=figsize)
    values = table.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="Reds", aspect="auto")

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index], fontsize=8)

    hi = np.nanmax(values) if values.size else 1
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = values[i, j]
            if v:
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * hi else "#333333")
    ax.grid(False)
    _finish(ax, title, xlab, ylab, grid_axis="both")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Frequency")
    fig.tight_layout()
    plt.show()
    return table


def stem_and_leaf(x, decimals=1):
    """Text stem-and-leaf display.

    Every observation is still visible, unlike a histogram, which is the whole
    appeal of the technique.
    """
    s = pd.Series(x).dropna().round(decimals)
    scale = 10 ** decimals
    stems = np.floor(s).astype(int)
    leaves = ((s - stems) * scale).round().astype(int)

    lines = []
    for stem in range(stems.min(), stems.max() + 1):
        row = sorted(leaves[stems == stem].tolist())
        lines.append(f"{stem:>5} | " + " ".join(str(v) for v in row))
    text = "\n".join(lines)
    print(text)
    print(f"\n{len(s):,} observations. Stem = whole number, "
          f"leaf = first decimal place.")
    return text


# ---------------------------------------------------------------------------
# Relationship charts
# ---------------------------------------------------------------------------

def scatter(x, y, title=None, xlab=None, ylab=None, labels=None,
            figsize=(9, 5.5)):
    """Every observation, no summarising. Usually the first chart to draw."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color=UNLV_SCARLET, alpha=0.75, s=38, edgecolor="white")
    if labels is not None:
        for xi, yi, li in zip(x, y, labels):
            ax.annotate(str(li), (xi, yi), fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
    _finish(ax, title, xlab, ylab, grid_axis="both")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"x": x, "y": y})


def time_series(dates, values, title=None, ylab=None, figsize=(10, 4.5),
                shade=None, shade_label=None):
    """Line chart over time, optionally shading periods (recessions, say)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pd.to_datetime(dates), values, color=UNLV_SCARLET, linewidth=1.4)

    if shade is not None:
        d = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
        flag = pd.Series(list(shade)).astype(bool).reset_index(drop=True)
        start = None
        for i, on in enumerate(flag):
            if on and start is None:
                start = d[i]
            elif not on and start is not None:
                ax.axvspan(start, d[i], color=UNLV_GRAY, alpha=0.18, linewidth=0)
                start = None
        if start is not None:
            ax.axvspan(start, d.iloc[-1], color=UNLV_GRAY, alpha=0.18, linewidth=0)
        if shade_label:
            ax.text(0.01, 0.02, f"shaded: {shade_label}", transform=ax.transAxes,
                    fontsize=8, color=UNLV_GRAY)

    _finish(ax, title, None, ylab)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})
