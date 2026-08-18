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
    "probability_tree", "convergence_plot",
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
                     points=400):
    """The law of large numbers, drawn: a running empirical probability
    against the classical probability it is converging on.

    `outcomes` is a boolean-like sequence — True where the event happened.
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
                   label=f"classical probability = {target:.4f}")
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
