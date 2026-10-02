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
    "box_plot", "normal_curve", "returns_bar",
    "probability_tree", "convergence_plot", "venn", "pmf_chart",
    "density_chart", "density_histogram", "approximation_chart",
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

    if pd.api.types.is_numeric_dtype(s) and isinstance(bins, str):
        # Ungrouped: one class per value, the way the textbook's first
        # frequency distribution is built. Every whole number between the
        # smallest and the largest gets a row EVEN IF NOTHING LANDS THERE --
        # an empty class is a finding, and dropping it hides the finding.
        if bins != "each":
            raise ValueError('bins must be a number, a sequence, or "each"')
        lo, hi = int(np.floor(s.min())), int(np.ceil(s.max()))
        values = list(range(lo, hi + 1))
        counts = pd.Series([int((s == v).sum()) for v in values], index=values)
        labels = [str(v) for v in values]
    elif pd.api.types.is_numeric_dtype(s):
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


def discrete_histogram(x, title=None, xlab=None, ylab="Frequency",
                       figsize=(9, 4.5)):
    """Histogram for DISCRETE data: one bar per whole-number value, with gaps.

    Two conventions are doing work here, and both are the textbook's:

      * one class per value, not a range of values, because the values are
        counts and there is nothing between 2 and 3 to put in a class;
      * gaps between the bars, because the data are discrete. A histogram of
        continuous data has bars that touch. The gap is not decoration -- it
        is the chart telling the reader which kind of variable this is.

    Values with a count of zero still get a position on the axis. A missing
    class is information.
    """
    s = pd.Series(x).dropna()
    lo, hi = int(np.floor(s.min())), int(np.ceil(s.max()))
    values = list(range(lo, hi + 1))
    counts = [int((s == v).sum()) for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(values, counts, width=0.55, color=UNLV_SCARLET, alpha=0.85)
    ax.set_xticks(values)
    for v, c in zip(values, counts):
        if c == 0:
            ax.text(v, 0.15, "none", ha="center", va="bottom", fontsize=7.5,
                    color=UNLV_SCARLET, style="italic")
    _finish(ax, title, xlab, ylab)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"value": values, "count": counts})


def connected_scatter(x, y, labels=None, title=None, xlab=None, ylab=None,
                      logx=False, annotate_every=1, figsize=(10, 5.5)):
    """A scatter whose points are joined in order -- a *time path*.

    Two variables plotted against each other, with the points connected in the
    order they occurred. It is not a time series (neither axis is time) and it
    is not an ordinary scatter (the order of the points carries meaning). What
    it shows is where a thing has travelled through a two-variable space.

    `logx` puts the horizontal axis on a log scale, which is the honest choice
    when the variable spans more than an order of magnitude -- and which is a
    decision worth showing the reader rather than making silently.
    """
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, ys, color=UNLV_SCARLET, linewidth=2.6, zorder=2,
            solid_capstyle="round")
    ax.scatter(xs, ys, s=34, color=UNLV_SCARLET, zorder=3,
               edgecolor="white", linewidth=1.1)

    if labels is not None:
        for i, (xi, yi, li) in enumerate(zip(xs, ys, labels)):
            if i % annotate_every == 0:
                ax.annotate(str(li), (xi, yi), textcoords="offset points",
                            xytext=(7, 7), fontsize=8.5, color=UNLV_GRAY)

    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.set_xticks([1000, 2000, 5000, 10000, 20000])
        ax.minorticks_off()

    _finish(ax, title, xlab, ylab, grid_axis="both")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"x": xs, "y": ys})


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


# ---------------------------------------------------------------------------
# Distribution and spread charts (Chapter 3)
# ---------------------------------------------------------------------------

def box_plot(x, title=None, xlab=None, labels=None, figsize=(9, 4.0),
             annotate=True):
    """Box-and-whisker plot, drawn from the course's own quartile rule.

    matplotlib's boxplot has its own quartile convention. Using it here would
    put a box on the screen whose edges disagree with the Q1 and Q3 printed
    three lines above it in the note. So the box is drawn from numbers this
    course computed, and the whiskers extend to the most extreme observation
    still inside the 1.5 x IQR fences. Points beyond the fences are plotted
    individually, which is the definition of an outlier used in this chapter.

    Pass a list of arrays with `labels` to compare several groups.
    """
    from _stats import quartiles, outlier_fences

    if labels is None:
        groups, names = [pd.Series(x).dropna()], [""]
    else:
        groups = [pd.Series(g).dropna() for g in x]
        names = list(labels)

    fig, ax = plt.subplots(figsize=figsize)
    rows = []
    for pos, (s, nm) in enumerate(zip(groups, names)):
        q1, q2, q3 = quartiles(s)
        lo_fence, hi_fence = outlier_fences(s)
        inside = s[(s >= lo_fence) & (s <= hi_fence)]
        lo_whisker, hi_whisker = inside.min(), inside.max()
        outliers = s[(s < lo_fence) | (s > hi_fence)]

        ax.broken_barh([(q1, q3 - q1)], (pos - 0.22, 0.44),
                       facecolors=UNLV_SCARLET, alpha=0.35,
                       edgecolors=UNLV_SCARLET, linewidth=1.4)
        ax.plot([q2, q2], [pos - 0.22, pos + 0.22], color=UNLV_SCARLET,
                linewidth=2.4)
        ax.plot([lo_whisker, q1], [pos, pos], color=UNLV_GRAY, linewidth=1.2)
        ax.plot([q3, hi_whisker], [pos, pos], color=UNLV_GRAY, linewidth=1.2)
        for end in (lo_whisker, hi_whisker):
            ax.plot([end, end], [pos - 0.11, pos + 0.11], color=UNLV_GRAY,
                    linewidth=1.2)
        if len(outliers):
            ax.scatter(outliers, np.full(len(outliers), pos), s=22,
                       facecolor="none", edgecolor=UNLV_SCARLET, linewidth=1.0)
        ax.scatter([s.mean()], [pos], marker="D", s=30, color="black",
                   zorder=5, label="mean" if pos == 0 else None)

        rows.append({"group": nm or "all", "n": int(s.size), "min": s.min(),
                     "Q1": q1, "median": q2, "Q3": q3, "max": s.max(),
                     "IQR": q3 - q1, "outliers": int(len(outliers))})

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(names)
    if len(groups) == 1:
        # A single unlabelled group needs no y axis at all, and a tall empty
        # band above and below the box just wastes the figure.
        ax.set_yticks([])
        ax.set_ylim(-0.6, 0.6)
        for side in ("left",):
            ax.spines[side].set_visible(False)
    if annotate:
        ax.legend(frameon=False, fontsize=9, loc="best")
    _finish(ax, title, xlab, None, grid_axis="x")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(rows).round(4)


def normal_curve(title="The empirical rule", figsize=(9, 4.5)):
    """The bell-shaped reference curve with the 68-95-99.7 bands marked.

    This is a picture of a mathematical function, not of any data set. It is
    here so that when the empirical rule is applied to real data, the shape
    being assumed is on the page next to it.
    """
    z = np.linspace(-4, 4, 1000)
    density = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)

    fig, ax = plt.subplots(figsize=figsize)
    bands = [(3, "#f6e6e4", "99.7%"), (2, "#eccdc9", "95%"), (1, "#d9a49c", "68%")]
    for k, colour, label in bands:
        mask = (z >= -k) & (z <= k)
        ax.fill_between(z[mask], density[mask], color=colour, linewidth=0)
        ax.annotate(label, xy=(0, 0), xytext=(0, 0.055 * k),
                    ha="center", fontsize=9, color="#4a4a4a")
    ax.plot(z, density, color=UNLV_SCARLET, linewidth=2)
    for k in (1, 2, 3):
        for side in (-k, k):
            ax.plot([side, side], [0, np.exp(-side ** 2 / 2) / np.sqrt(2 * np.pi)],
                    color=UNLV_GRAY, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(-4, 5))
    ax.set_ylim(0, 0.45)
    _finish(ax, title, "z, standard deviations from the mean", "density",
            grid_axis="y")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"within": ["1 sd", "2 sd", "3 sd"],
                         "bell_shaped_percent": [68.3, 95.4, 99.7]})


def returns_bar(periods, values, title=None, xlab=None, ylab=None,
                figsize=(10, 4.5), show_mean=True):
    """Bar chart of a return series, gains and losses coloured differently.

    Useful for the range: the tallest bar and the deepest one are the two
    numbers the range is built from, and here you can see which years they
    were.
    """
    v = pd.Series(values, dtype="float64")
    colours = np.where(v >= 0, UNLV_SCARLET, UNLV_GRAY)

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(v))
    ax.bar(positions, v.to_numpy(), color=colours, alpha=0.9)
    ax.axhline(0, color="#333333", linewidth=1.0)
    if show_mean:
        ax.axhline(v.mean(), color="black", linestyle="--", linewidth=1.3,
                   label=f"mean = {v.mean():.4f}")
        ax.legend(frameon=False, fontsize=9)

    labels = [str(p) for p in periods]
    step = max(1, len(labels) // 25)
    ax.set_xticks(positions[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
    _finish(ax, title, xlab, ylab)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"period": list(periods), "value": v.to_numpy()})


# ---------------------------------------------------------------------------
# Probability figures (Chapter 4)
# ---------------------------------------------------------------------------

def probability_tree(first, second, title=None, figsize=(11, 6.5),
                     value_fmt="{:.4f}"):
    """A probability tree, drawn from the numbers rather than from an image.

    `first`  is a mapping {branch label: probability} for the first stage.
    `second` is a mapping {first branch label: {branch label: probability}}
             giving the conditional probability of each second-stage branch.

    Every path's joint probability is printed at the tip, and the function
    returns them as a table. The tree and the table cannot disagree, which is
    the point: a tree redrawn by hand after the data changes usually can.
    """
    firsts = list(first)
    paths = []
    for a in firsts:
        for b in second.get(a, {}):
            paths.append((a, b, first[a] * second[a][b]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(len(paths), 2) + 1)

    root = (0.4, (len(paths) + 1) / 2)
    ax.scatter(*root, s=45, color=UNLV_SCARLET, zorder=5)

    tip = len(paths)
    node_x, mid_x = 4.2, 8.0
    for a in firsts:
        branches = list(second.get(a, {}))
        if not branches:
            continue
        ys = [tip - i for i in range(len(branches))]
        tip -= len(branches)
        a_y = sum(ys) / len(ys)

        ax.plot([root[0], node_x], [root[1], a_y], color=UNLV_SCARLET,
                linewidth=1.5)
        ax.text((root[0] + node_x) / 2, (root[1] + a_y) / 2 + 0.16,
                f"{a}\n{value_fmt.format(first[a])}", ha="center", va="bottom",
                fontsize=8.5, color="#333333")
        ax.scatter([node_x], [a_y], s=40, color=UNLV_SCARLET, zorder=5)

        for b, y in zip(branches, ys):
            ax.plot([node_x, mid_x], [a_y, y], color=UNLV_GRAY, linewidth=1.2)
            ax.text((node_x + mid_x) / 2, (a_y + y) / 2 + 0.14,
                    f"{b}  {value_fmt.format(second[a][b])}", ha="center",
                    va="bottom", fontsize=8, color=UNLV_GRAY)
            ax.text(mid_x + 0.15, y, value_fmt.format(first[a] * second[a][b]),
                    ha="left", va="center", fontsize=8.5, color="#333333")

    if title:
        ax.set_title(title, color=UNLV_SCARLET, fontsize=12, fontweight="600")
    fig.tight_layout()
    plt.show()

    out = pd.DataFrame(paths, columns=["first", "second", "joint_probability"])
    out["path"] = out["first"] + " → " + out["second"]
    return out[["path", "first", "second", "joint_probability"]]


def convergence_plot(outcomes, target=None, title=None, xlab="Number of rolls",
                     ylab="Running empirical probability", figsize=(10, 4.5),
                     points=400, target_label="classical probability"):
    """The law of large numbers, drawn: a running empirical probability
    against the classical probability it is converging on.

    `outcomes` is a boolean-like sequence — True where the event happened.
    Pass numbers instead and the same picture is a running sample mean
    closing in on an expected value; set `target_label` to say so.
    """
    hits = pd.Series(outcomes).astype(float).to_numpy()
    n = len(hits)
    running = np.cumsum(hits) / np.arange(1, n + 1)

    # Plotting a million points draws a million points nobody can see. Sample
    # on a log scale, which is where the interesting part of convergence is.
    idx = np.unique(np.logspace(0, np.log10(n), points).astype(int)) - 1
    idx = idx[idx >= 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(idx + 1, running[idx], color=UNLV_SCARLET, linewidth=1.4)
    if target is not None:
        ax.axhline(target, color="black", linestyle="--", linewidth=1.3,
                   label=f"{target_label} = {target:.4f}")
        ax.legend(frameon=False, fontsize=9)
    ax.set_xscale("log")
    _finish(ax, title, xlab, ylab)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({
        "rolls": [10, 100, 1_000, 10_000, 100_000, n],
        "empirical_probability": [running[min(k, n) - 1]
                                  for k in [10, 100, 1_000, 10_000, 100_000, n]],
    })


def venn(a="A", b="B", shade="intersection", disjoint=False, title=None,
         figsize=(5.2, 3.6), single=False):
    """Two-event Venn diagram: a box for the sample space, a circle per event.

    `shade` picks the region drawn in scarlet:
        "intersection"  A and B          "union"       A or B (or both)
        "a"             just A           "complement"  everything outside A
        "none"          no shading
    `disjoint=True` draws the circles apart — mutually exclusive events.
    `single=True` draws only event A (for the complement rule).

    The picture is schematic, not area-proportional. It exists to make the
    words "and", "or", "not" and "cannot both happen" visible, which is what
    Venn diagrams are for.
    """
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_axis_off()

    box = Rectangle((0.3, 0.3), 9.4, 6.4, fill=False, linewidth=1.2,
                    edgecolor=UNLV_GRAY)
    ax.add_patch(box)
    ax.text(0.55, 6.35, "sample space", fontsize=8.5, color=UNLV_GRAY)

    r = 2.0
    ca, cb = ((2.7, 3.4), (7.3, 3.4)) if disjoint else ((3.9, 3.4), (6.1, 3.4))
    if single:
        ca, cb = (5.0, 3.4), None

    # Shading is done by rasterising a fine grid: it is the simplest way to
    # get "union minus intersection" style regions right without a
    # geometry library.
    if shade != "none":
        xs, ys = np.meshgrid(np.linspace(0.3, 9.7, 700),
                             np.linspace(0.3, 6.7, 500))
        in_a = (xs - ca[0]) ** 2 + (ys - ca[1]) ** 2 <= r ** 2
        in_b = (np.zeros_like(in_a, dtype=bool) if cb is None
                else (xs - cb[0]) ** 2 + (ys - cb[1]) ** 2 <= r ** 2)
        region = {"intersection": in_a & in_b, "union": in_a | in_b,
                  "a": in_a, "complement": ~in_a}[shade]
        ax.contourf(xs, ys, region.astype(float), levels=[0.5, 1.5],
                    colors=[UNLV_SCARLET], alpha=0.35)

    for c in (ca, cb):
        if c is not None:
            ax.add_patch(Circle(c, r, fill=False, linewidth=1.8,
                                edgecolor="#333333"))
    if single:
        ax.text(ca[0], ca[1], a, ha="center", va="center", fontsize=12,
                fontweight="600", color="#333333")
        ax.text(8.6, 1.0, b, ha="center", va="center", fontsize=12,
                fontweight="600", color="#333333")
    else:
        ax.text(ca[0] - (1.1 if not disjoint else 0), ca[1] + r + 0.25, a,
                ha="center", fontsize=11, fontweight="600", color="#333333")
        ax.text(cb[0] + (1.1 if not disjoint else 0), cb[1] + r + 0.25, b,
                ha="center", fontsize=11, fontweight="600", color="#333333")

    if title:
        ax.set_title(title, color=UNLV_SCARLET, fontsize=11, fontweight="600")
    fig.tight_layout()
    plt.show()
    return {"a": a, "b": b, "shade": shade, "disjoint": disjoint}


def pmf_chart(x, probabilities, title=None, xlab="x", ylab="Probability",
              highlight=None, overlay=None, overlay_label=None,
              bar_label=None, mean=None, ymax=None, figsize=(9, 4.5)):
    """A discrete probability distribution, drawn as one bar per value.

    `highlight` is a collection of x values to draw in scarlet while the rest
    go grey, which is how to show the event a question is asking about.
    `overlay` is a second set of probabilities on the same x values, drawn as
    points, for putting one distribution on top of another. `mean` marks the
    expected value with a dashed line. Set `ymax` to the same value on two
    charts that will be compared side by side, so that equal heights mean
    equal probabilities.
    """
    x = list(x)
    p = list(probabilities)
    fig, ax = plt.subplots(figsize=figsize)
    if highlight is None:
        colors = [UNLV_SCARLET] * len(x)
    else:
        chosen = set(highlight)
        colors = [UNLV_SCARLET if v in chosen else "#c9c9c9" for v in x]
    ax.bar(x, p, color=colors, alpha=0.9, width=0.8, label=bar_label)
    if overlay is not None:
        ax.plot(x, list(overlay), linestyle="none", marker="o", markersize=6,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.4, label=overlay_label)
    if mean is not None:
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.2,
                   label=f"mean = {mean:.2f}")
    if bar_label or overlay_label or mean is not None:
        ax.legend(frameon=False, fontsize=9)
    if len(x) <= 25:
        ax.set_xticks(x)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    _finish(ax, title, xlab, ylab)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"x": x, "probability": p})


# ---------------------------------------------------------------------------
# Continuous distributions (Chapter 6)
# ---------------------------------------------------------------------------

_CURVE_COLORS = [UNLV_SCARLET, "#333333", "#9FA1A4", "#d9822b"]


def density_chart(x, curves, shade=None, area=None, title=None, xlab="x",
                  ylab="density, f(x)", z_scale=None, mark=None,
                  ymax=None, figsize=(9, 4.5)):
    """One or more probability density curves, with an interval shaded.

    `x` is the grid of values and `curves` maps a label to the density at
    each grid point. `shade=(lower, upper)` shades the area under the FIRST
    curve between the two values; use None for an open end. `area` is the
    probability of that interval, printed on the chart, because the shaded
    AREA is the probability and the height of the curve is not.

    The vertical axis is labelled as a density on purpose. A density is a
    height, it can exceed 1, and it is not the probability of anything.

    `z_scale=(mu, sigma)` adds a second horizontal axis in z-score units, so
    the original scale and the standardized scale can be read off one
    picture. `mark` is a list of x values to draw as dashed reference lines.
    """
    x = np.asarray(x, dtype=float)
    if not isinstance(curves, dict):
        curves = {None: curves}
    fig, ax = plt.subplots(figsize=figsize)
    first = None
    for i, (label, f) in enumerate(curves.items()):
        f = np.asarray(f, dtype=float)
        if first is None:
            first = f
        ax.plot(x, f, color=_CURVE_COLORS[i % len(_CURVE_COLORS)],
                linewidth=2.2 if i == 0 else 1.8, label=label)
    if shade is not None:
        lo = x.min() if shade[0] is None else shade[0]
        hi = x.max() if shade[1] is None else shade[1]
        mask = (x >= lo) & (x <= hi)
        ax.fill_between(x[mask], first[mask], color=UNLV_SCARLET, alpha=0.30,
                        linewidth=0)
        for edge in (shade[0], shade[1]):
            if edge is not None:
                ax.axvline(edge, color=UNLV_SCARLET, linewidth=1.0,
                           linestyle=":")
        if area is not None:
            # A corner label, on the side away from the shading, stays legible
            # however narrow the shaded slice is.
            right_side = (lo + hi) / 2 > (x.min() + x.max()) / 2
            ax.text(0.02 if right_side else 0.98, 0.92,
                    f"shaded area = {area:.4f}", transform=ax.transAxes,
                    ha="left" if right_side else "right", fontsize=10,
                    color="#1a1a1a", fontweight="600",
                    bbox=dict(facecolor="white", edgecolor=UNLV_SCARLET,
                              boxstyle="round,pad=0.3"))
    for m in (mark or []):
        ax.axvline(m, color=UNLV_GRAY, linestyle="--", linewidth=1.0)
    if any(label is not None for label in curves):
        ax.legend(frameon=False, fontsize=9)
    if ymax is None:
        # Headroom above the tallest curve, so the area label never sits on it.
        ymax = 1.22 * max(float(np.nanmax(np.asarray(f, dtype=float)))
                          for f in curves.values())
    ax.set_ylim(0, ymax)
    ax.set_xlim(x.min(), x.max())
    _finish(ax, title, xlab, ylab)
    if z_scale is not None:
        mu, sigma = z_scale
        top = ax.secondary_xaxis(
            "top", functions=(lambda v: (v - mu) / sigma,
                              lambda z: mu + z * sigma))
        top.set_xlabel("z = (x - mu) / sigma", color=UNLV_GRAY)
        top.tick_params(colors=UNLV_GRAY)
    fig.tight_layout()
    plt.show()
    out = pd.DataFrame({"x": x})
    for label, f in curves.items():
        out[label if label is not None else "density"] = np.asarray(f)
    return out


def density_histogram(data, x=None, density=None, bins=40, title=None,
                      xlab="x", curve_label="theoretical density",
                      bar_label="sample", figsize=(9, 4.5)):
    """A histogram drawn on the DENSITY scale, so that it can sit under a
    density curve and be compared with it. Bar areas sum to 1, just as the
    area under the curve does.

    Pass `x` and `density` to draw the theoretical curve on top.
    """
    s = pd.Series(data, dtype=float).dropna()
    fig, ax = plt.subplots(figsize=figsize)
    heights, edges, _ = ax.hist(s, bins=bins, density=True,
                                color="#c9c9c9", edgecolor="white",
                                label=bar_label)
    if x is not None and density is not None:
        ax.plot(x, density, color=UNLV_SCARLET, linewidth=2.2,
                label=curve_label)
    ax.legend(frameon=False, fontsize=9)
    _finish(ax, title, xlab, "density")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"lower": edges[:-1], "upper": edges[1:],
                         "density": heights})


def approximation_chart(x, probabilities, mu, sigma, highlight=None,
                        shade=None, title=None, xlab="x",
                        figsize=(10, 4.5)):
    """A binomial distribution as bars with its normal approximation drawn
    over it.

    Bars in `highlight` are the whole numbers the question asks about, drawn
    in scarlet. `shade=(lower, upper)` shades the area under the normal curve
    that stands in for them, which, after the continuity correction, runs
    from half a unit below the first bar to half a unit above the last.
    """
    x = np.asarray(list(x), dtype=float)
    p = np.asarray(list(probabilities), dtype=float)
    chosen = set(highlight or [])
    colors = [UNLV_SCARLET if v in chosen else "#d9d9d9" for v in x]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, p, width=1.0, color=colors, edgecolor="white", alpha=0.85,
           label="binomial P(X = x)")
    grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 800)
    curve = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(grid, curve, color="black", linewidth=1.8,
            label=f"normal, mean {mu:.2f}, sd {sigma:.2f}")
    if shade is not None:
        lo = grid.min() if shade[0] is None else shade[0]
        hi = grid.max() if shade[1] is None else shade[1]
        m = (grid >= lo) & (grid <= hi)
        ax.fill_between(grid[m], curve[m], facecolor="none",
                        edgecolor="black", hatch="///", linewidth=0,
                        zorder=3, label="normal area used")
    ax.legend(frameon=False, fontsize=9)
    if len(x) <= 30:
        ax.set_xticks(x)
    _finish(ax, title, xlab, "probability (bars) / density (curve)")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"x": x, "probability": p})
