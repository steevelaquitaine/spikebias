"""Panels of Fig. 5e-g: precision and recall of the curation model (e, f) and weights of its quality metrics (g), drawn from the source data

author: laquitainesteeve@gmail.com

The panels have the geometry of panels e, f and g of the paper's figure (page 16 of paper/Laquitaine_NatureNeuro_2025.pdf). Every element is drawn in
the pt coordinates of that page (y down), measured on the figure: axes, ticks, dots (1.95 pt, jitter +-1.52 pt), median markers (4.87 pt with a white edge of 1 pt) and
their 95% CI, red dashed line, bars (6.29 pt high), stars, texts. All text is Arial 5 pt (titles in bold). In g the axes keep the size of the paper's
(x axis 81.5 pt wide, ticks 26.69 pt on each side of 0): the tick values are 10, 0, 10 (or 20 if the weights exceed 15 in absolute value) since the weights of the
10-minute model are smaller than the paper's.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle

FONT = 5
BLACK = (0.06, 0.06, 0.05)
ORANGE = (0.97, 0.61, 0.05)
RED = (0.93, 0.12, 0.14)
PINK = (0.96, 0.40, 0.49)
LW = 0.5

# geometry measured on the paper's figure (pt of the page, y down)
PR = {
    "e": dict(region=(95.0, 490.0, 226.0, 625.0), title="Spontaneous regime", title_xy=(147.55, 498.3), y0=587.75, ppu=62.39, xs=(131.02, 169.11),
              axis=(127.49, 172.68, 590.87), yaxis=(125.23, 525.36, 587.75), dash_y=556.56, label_dy=0.0, with_y=True),
    "f": dict(region=(210.0, 490.0, 282.0, 625.0), title="Evoked regime", title_xy=(247.25, 498.3), y0=586.97, ppu=62.39, xs=(227.57, 265.65),
              axis=(224.0, 269.2, 590.09), yaxis=None, dash_y=555.77, label_dy=-0.78, with_y=False),
}
# 25-point-circle dots, jitter of seaborn's stripplot (0.04 of the category width, 38.09 pt)
DOT, JITTER = 1.95, 1.52
MARKER, MARKER_EDGE = 4.87, 1.0057
STAR = np.array([[-0.828, 1.994], [-1.699, 1.326], [-1.422, 1.021], [-1.133, 0.729], [-0.828, 0.443], [-0.719, 0.334], [-0.648, 0.26], [-0.617, 0.232],
                 [-0.707, 0.213], [-0.977, 0.146], [-1.422, 0.033], [-1.75, -0.045], [-1.965, -0.104], [-2.062, -0.135], [-1.727, -1.139], [-1.234, -0.932],
                 [-0.793, -0.713], [-0.406, -0.471], [-0.496, -1.096], [-0.539, -1.607], [-0.539, -1.994], [0.48, -1.994], [0.48, -1.721], [0.426, -1.205],
                 [0.324, -0.471], [0.406, -0.498], [0.574, -0.568], [0.816, -0.682], [1.16, -0.842], [1.477, -0.979], [1.773, -1.092], [2.062, -0.057],
                 [1.645, 0.045], [1.16, 0.139], [0.605, 0.232], [1.289, 0.99], [1.43, 1.154], [1.535, 1.279], [1.609, 1.373], [0.742, 1.951], [-0.039, 0.689],
                 [-0.262, 1.096], [-0.527, 1.533]])

W = dict(region=(308.0, 468.0, 508.0, 625.0), tick_y0=515.98, tick_dy=7.86, bar_h=6.29, zero=430.87, tick_dx=26.69, spine=(386.48, 510.86, 591.87),
         axis=(390.55, 472.07, 595.92), star_x0=474.49, star_pitch=4.545, star_dy=-0.65)


def _fig(region):
    x0, y0, x1, y1 = region
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    return fig, ax


def _text(ax, x, y, s, ha="center", rotation=0, bold=False):
    """Text of 5 pt; (x, y) is the anchor on the baseline"""
    ax.text(x, y, s, ha=ha, va="baseline", rotation=rotation, rotation_mode="anchor", fontsize=FONT, fontweight="bold" if bold else "normal", color="k")


def _line(ax, xs, ys, **kw):
    ax.plot(xs, ys, color=kw.pop("color", "k"), linewidth=LW, solid_capstyle="butt", **kw)


def plot_precision_recall(df, panel="e"):
    """Precision and recall of the model on the test units of each fold (black dots), median and 95% CI (orange), 0.5 (red dashed line)"""
    g = PR[panel]
    fig, ax = _fig(g["region"])
    y = lambda v: g["y0"] - g["ppu"] * np.asarray(v, dtype=float)
    rng = np.random.RandomState(0)

    # axes
    x0, x1, ya = g["axis"]
    _line(ax, [x0, x1], [ya, ya])
    for x in g["xs"]:
        _line(ax, [x, x], [ya, ya + 3.76])
    if g["with_y"]:
        xa, y_top, y_bottom = g["yaxis"]
        _line(ax, [xa, xa], [y_top, y_bottom])
        for k, v in enumerate(np.arange(0, 1.01, 0.2)):
            yt = g["y0"] - g["ppu"] * v
            _line(ax, [xa - 3.75, xa], [yt, yt])
            _text(ax, 117.7, yt + 1.79, f"{v:.1f}", ha="right")
        _text(ax, 103.7, 556.5, "Decoding performance (ratio)", rotation=90)
    ax.plot([x0, x1], [g["dash_y"]] * 2, color=RED, linewidth=LW, dashes=(1.85 / LW, 0.8 / LW), zorder=0)

    # dots, medians and 95% CI
    for x, col in zip(g["xs"], ["precision", "recall"]):
        v = df[col].dropna().values
        ax.scatter(x + rng.uniform(-JITTER, JITTER, len(v)), y(v), s=DOT ** 2, color=BLACK, linewidths=0, zorder=2)
        m = np.median(v)
        ci = 1.96 * np.std(v) / np.sqrt(len(v))
        _line(ax, [x, x], [y(m - ci), y(m + ci)], color=ORANGE, zorder=3)
        ax.plot([x], [y(m)], marker="o", markersize=MARKER, markerfacecolor=ORANGE, markeredgecolor="w", markeredgewidth=MARKER_EDGE, linestyle="none", zorder=4)

    # texts
    dy = g["label_dy"]
    _text(ax, *g["title_xy"], g["title"], bold=True)
    _text(ax, g["xs"][0], 602.6 + dy, "precision")
    _text(ax, g["xs"][1], 602.6 + dy, "recall")
    _text(ax, (g["xs"][0] + g["xs"][1]) / 2 - 0.1 + (0.0 if panel == "e" else 0.0), 611.4 + dy, "Metrics only")
    _text(ax, (g["xs"][0] + g["xs"][1]) / 2, 618.4 + dy, "(Regression model)")
    if panel == "e":                                     # key of the dots
        for yy, color, size in ((510.04, ORANGE, MARKER), (516.15, BLACK, MARKER)):
            ax.plot([160.04], [yy], marker="o", markersize=size, markerfacecolor=color, markeredgecolor="w", markeredgewidth=MARKER_EDGE, linestyle="none", zorder=4)
        _text(ax, 166.4, 511.6, "Median over 100 folds", ha="left")
        _text(ax, 165.3, 519.0, "A single fold", ha="left")
    return fig


def plot_weights(df):
    """Weights of the z-scored quality metrics (bars); the weights significantly different from 0 are pink with stars"""
    w = W
    fig, ax = _fig(w["region"])
    d = df.sort_values(by="weight", ascending=False).reset_index(drop=True)
    big = d["weight"].abs().max() > 15
    unit = 20.0 if big else 10.0
    pt_per_unit = w["tick_dx"] / unit

    # axes
    _line(ax, [w["spine"][0]] * 2, [w["spine"][1], w["spine"][2]])
    _line(ax, [w["axis"][0], w["axis"][1]], [w["axis"][2]] * 2)
    for v in (-unit, 0, unit):
        x = w["zero"] + v * pt_per_unit
        _line(ax, [x, x], [w["axis"][2], w["axis"][2] + 3.75])
        _text(ax, x, 607.6, f"{int(v)}".replace("-", "−"))
    _text(ax, 432.65, 617.3, "Weights of z-scored features")

    # bars, feature names and stars
    for i, r in d.iterrows():
        yc = w["tick_y0"] + i * w["tick_dy"]
        _line(ax, [w["spine"][0] - 3.75, w["spine"][0]], [yc, yc])
        sig = isinstance(r["significance"], str) and r["significance"] != ""
        x_end = w["zero"] + r["weight"] * pt_per_unit
        ax.add_patch(Rectangle((min(w["zero"], x_end), yc - w["bar_h"] / 2), abs(x_end - w["zero"]), w["bar_h"], facecolor=PINK if sig else BLACK, edgecolor="none"))
        _text(ax, 379.0, yc + 1.79, r["feature"], ha="right")
        if sig:
            for j in range(len(r["significance"])):
                ax.add_patch(Polygon(STAR + [w["star_x0"] + j * w["star_pitch"], yc + w["star_dy"]], closed=True, facecolor=BLACK, edgecolor="none"))
    _text(ax, 319.3, 551.0, "Spiking features", rotation=90)
    _text(ax, 431.15, 477.9, "Spiking features predictive of good units", bold=True)
    _text(ax, 431.25, 486.1, "(Regression model for the spontaneous regime)")
    return fig
