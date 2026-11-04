"""Panel of Fig. 6a: sorting accuracy of the ground truth neurons ranked by accuracy, per simulation and spike sorter, drawn from the source data

author: laquitainesteeve@gmail.com

The panel has the geometry of panel a of the paper's figure (page 19 of paper/Laquitaine_NatureNeuro_2025.pdf). Every element is drawn in the pt
coordinates of that page (y down), measured on the figure: the four axes (synthetic model, spontaneous and evoked Neuropixels simulations, dense probe),
their ticks, the curves of the six sorters (1 pt), the red dashed line at 0.8 and the legend. Text is Arial 5 pt and the titles 7 pt (bold).
"""

import numpy as np
from matplotlib import pyplot as plt

FONT, TITLE = 5, 7
REGION = (45.0, 68.0, 372.0, 205.0)
SORTERS = ["ks4", "ks3", "ks2.5", "ks2", "ks", "hs"]
COLORS = dict(zip(SORTERS, [(0.22, 0.46, 0.73), (0.96, 0.43, 0.09), (0.22, 0.65, 0.24), (0.85, 0.14, 0.15), (0.56, 0.34, 0.69), (0.53, 0.29, 0.27)]))
THR = 0.8
RED = (0.93, 0.12, 0.14)
PPU_Y = 56.59                      # pt per unit of accuracy
# per simulation: x of the first rank and pt per rank, y of the 0, x of the axis (left, right) and of its y, x ticks, x of the centre of the title, number format
AXES = {
    "S": dict(x0=79.92, ppr=36.74 / 200, y0=173.23, axis=(79.92, 127.95, 176.2), ticks=[0, 200], title="Synthetic", cx=102.1),
    "NS": dict(x0=138.6, ppr=32.98 / 1000, y0=173.23, axis=(138.6, 186.63, 176.2), ticks=[0, 1000], title="Biophy. Spont", cx=160.9),
    "E": dict(x0=198.05, ppr=29.5 / 1000, y0=173.09, axis=(198.05, 246.09, 176.06), ticks=[0, 1000], title="Biophy. Evoked", cx=222.3),
    "DS": dict(x0=256.94, ppr=41.98 / 2000, y0=173.23, axis=(256.94, 304.97, 176.2), ticks=[0, 2000], title="Biophy. Spont", cx=283.3),
}
PROP_X = {"NS": 181.95, "E": 241.35, "DS": 302.25}            # centre of the column of proportions
PROP_Y0 = {"NS": 119.0, "E": 120.1, "DS": 120.6}               # baseline of the first proportion
LEGEND_X = [(110.45, 118.6), (130.09, 138.2), (149.39, 157.5), (174.18, 182.3), (192.86, 201.0), (208.07, 216.2)]   # (start of the line, start of the label)
LEGEND_Y = [82.05, 81.9, 81.86, 81.95, 82.3, 82.58]


def _text(ax, x, y, s, ha="center", rotation=0, bold=False, size=FONT, color="k"):
    ax.text(x, y, s, ha=ha, va="baseline", rotation=rotation, rotation_mode="anchor", fontsize=size, fontweight="bold" if bold else "normal", color=color)


def _line(ax, xs, ys, **kw):
    ax.plot(xs, ys, color=kw.pop("color", "k"), linewidth=kw.pop("lw", 0.5), solid_capstyle="butt", **kw)


def plot_source_data(df):
    """Fig. 6a from the source data (columns simulation, sorter, rank, accuracy)"""
    x0, y0, x1, y1 = REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")

    for sim, g in AXES.items():
        d = df[df["simulation"] == sim]
        n = int(d[d["sorter"] == SORTERS[0]].shape[0])
        # axes
        xa0, xa1, ya = g["axis"]
        _line(ax, [xa0, xa1], [ya, ya])
        for t in g["ticks"]:
            xt = g["x0"] + t * g["ppr"]
            _line(ax, [xt, xt], [ya, ya + 3.75])
            _text(ax, xt, ya + 11.7, f"{t:d}")
        # curves, red dashed line
        for s in SORTERS:
            ds = d[d["sorter"] == s].sort_values("rank")
            ax.plot(g["x0"] + (ds["rank"].values - 1) * g["ppr"], g["y0"] - ds["accuracy"].values * PPU_Y, color=COLORS[s], linewidth=1.0, solid_joinstyle="round", zorder=2)
        ax.plot([xa0, xa1], [g["y0"] - THR * PPU_Y] * 2, color=RED, linewidth=0.5, dashes=(1.85 / 0.5, 0.8 / 0.5), zorder=3)
        # proportion of neurons sorted with at least 80% accuracy
        props = [(d[d["sorter"] == s]["accuracy"] >= THR).mean() for s in SORTERS]
        if sim == "S":
            for k, (s, p) in enumerate(zip(SORTERS, props)):
                _text(ax, 81.3, 138.7 + 5.85 * k, f"{s}: {round(p, 2):g}", ha="left")
            _text(ax, 79.5, 126.5, "0.80", ha="left", color="crimson")
        else:
            for k, p in enumerate(props):
                _text(ax, PROP_X[sim], PROP_Y0[sim] + 5.85 * k, f"{p:.2f}")
        # title of the axes
        _text(ax, g["cx"], 102.4, g["title"])
        _text(ax, g["cx"], 109.4, f"(n={n:,} model neurons)" if sim == "S" else f"(n={n:,})")

    # y axis of the first panel
    s = AXES["S"]
    _line(ax, [77.52, 77.52], [113.83, 173.23])
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        yt = 173.23 - v * PPU_Y
        _line(ax, [73.77, 77.52], [yt, yt])
        _text(ax, 70.0, yt + 1.79, f"{v:.2f}", ha="right")
    _text(ax, 52.2, 143.6, "Sorting accuracy (ratio)", rotation=90)
    _text(ax, 189.25, 197.8, "True units ranked by sorting accuracy")

    # legend and titles
    for k, s in enumerate(SORTERS):
        xs, xl = LEGEND_X[k]
        _line(ax, [xs, xs + 5.84], [LEGEND_Y[k]] * 2, color=COLORS[s], lw=1.0)
        _text(ax, xl, 84.2, s, ha="left")
    _text(ax, 167.4, 94.2, "Neuropixels (NPX)", bold=True, size=TITLE)
    _text(ax, 280.75, 94.2, "Dense probe", bold=True, size=TITLE)
    _text(ax, 303.0, 78.4, "Spike sorting accuracy on ground truths", bold=True, size=TITLE)
    return fig


# ---------------------------------------------------------------------------------------------------------------------------------
# Fig. 6b: proportion of the ground truth neurons sorted with at least 80% accuracy, per sorter, across simulations
# Geometry measured on panel b of the paper's figure (page 19): two bar charts (Neuropixels, dense probe), bars of 8.89 pt every 11.11 pt, 255.4 pt per unit.
GLOBAL_REGION = (322.0, 70.0, 545.0, 215.0)
GLOBAL_Y0, GLOBAL_PPU = 175.65, 255.4
GLOBAL = {
    "Neuropixels": dict(spine=367.26, x_axis=(370.8, 441.7), x_ticks=378.47, bar_x0=374.03, head="Neuropixels (NPX)", head_xy=(405.7, 95.3), n_xy=(405.7, 102.0), p_y=(114.47, 114.75), p_bar=(376.73, 436.64),
                        p_text_y=111.5, count_offsets=[3.35, 2.97, 1.4, 4.4, 1.7, 2.2], xlabel_cx=406.3, probes=["S", "NS", "E"]),
    "dense": dict(spine=463.32, x_axis=(466.86, 537.75), x_ticks=474.53, bar_x0=470.08, head="Dense probe", head_xy=(503.55, 94.2), n_xy=(503.55, 100.9), p_y=(138.37, 138.43), p_bar=(472.05, 532.41),
                  p_text_y=134.6, count_offsets=[0.75, 0.46, 2.2, 2.2, 2.9, 0.75], xlabel_cx=502.35, probes=["D1", "D2", "D3"]),
}
PITCH, BAR_W = 11.11, 8.89


def plot_global_accuracy(df):
    """Fig. 6b from the source data (columns probe, dataset, sorter, n_well_sorted, n_ground_truth)"""
    from scipy import stats
    x0, y0, x1, y1 = GLOBAL_REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    for probe, g in GLOBAL.items():
        d = df[df["probe"] == probe]
        good = d.groupby("sorter", sort=False)["n_well_sorted"].sum().loc[SORTERS].values
        total = int(d[d["sorter"] == SORTERS[0]]["n_ground_truth"].sum())
        prop = good / total
        p_value = stats.chisquare(f_obs=prop * 100).pvalue
        # axes
        _line(ax, [g["spine"]] * 2, [121.65, GLOBAL_Y0])
        for v in (0, 0.1, 0.2):
            yt = GLOBAL_Y0 - v * GLOBAL_PPU
            _line(ax, [g["spine"] - 3.75, g["spine"]], [yt, yt])
            _text(ax, g["spine"] + (359.7 - 367.26), yt + 1.79, f"{v:.1f}", ha="right")
        _line(ax, list(g["x_axis"]), [178.35, 178.35])
        # bars, counts, sorter labels
        for k, s in enumerate(SORTERS):
            xc = g["x_ticks"] + k * PITCH
            top = GLOBAL_Y0 - prop[k] * GLOBAL_PPU
            ax.add_patch(plt.Rectangle((g["bar_x0"] + k * PITCH, top), BAR_W, GLOBAL_Y0 - top, facecolor=COLORS[s], edgecolor="none"))
            _line(ax, [xc, xc], [178.35, 182.1])
            _text(ax, xc, top - g["count_offsets"][k], f"{good[k]}")
            ax.text(xc + 3.6, 188.4, s, ha="right", va="center", rotation=45, rotation_mode="anchor", fontsize=FONT)
        _text(ax, g["xlabel_cx"], 208.9, "Spike sorters")
        # chi-square p value
        _line(ax, list(g["p_bar"]), list(g["p_y"]))
        _text(ax, (g["p_bar"][0] + g["p_bar"][1]) / 2, g["p_text_y"], f"p={p_value:.1f}")
        _text(ax, *g["head_xy"], g["head"], bold=True, size=TITLE)
        _text(ax, *g["n_xy"], f"(n={total:,} model neurons)")
    _text(ax, 338.9, 149.45, "Proportion of neurons", rotation=90)
    _text(ax, 345.8, 148.6, "sorted with above 80% accuracy", rotation=90)
    _text(ax, 452.85, 84.0, "Global accuracy across simulations", bold=True, size=TITLE)
    return fig


# ---------------------------------------------------------------------------------------------------------------------------------
# Fig. 6c-d: sorting accuracy of the ground truth neurons (open circles) and prediction of the model (red line), ranked by predicted accuracy
# Geometry measured on panels c and d of the paper's figure (page 19)
GREY = (0.48, 0.49, 0.49)
MODEL_RED = (0.93, 0.11, 0.14)
MODEL_PPU = 51.43                                  # pt per unit of accuracy
MODEL = {
    "c": dict(region=(48.0, 205.0, 280.0, 305.0), y0=284.85, x0=85.44, ppr=28.2 / 200, spine_x=79.82, spine_y=(230.85, 284.85), axis=(85.44, 275.66, 287.55), x_max=1200,
              ytick_right=72.3, xtick_baseline=297.4, ylabel=(58.3, 258.95), r2=(230.7, 259.2), title="Model of sorting accuracy for ks4 (Biophy. Spont, NPX)", title_xy=(178.05, 216.4),
              xlabel=None, legend=True),
    "d": dict(region=(48.0, 308.0, 282.0, 415.0), y0=386.59, x0=86.93, ppr=22.35 / 200, spine_x=81.32, spine_y=(332.59, 386.59), axis=(86.93, 277.16, 389.29), x_max=1600,
              ytick_right=73.8, xtick_baseline=401.0, ylabel=(59.8, 359.65), r2=(232.4, 360.9), title="Model of sorting accuracy for ks4 (Biophy. Evoked, NPX)", title_xy=(180.9, 321.0),
              xlabel=("True units ranked by model sorting accuracy", 178.5, 410.3), legend=False),
}


def plot_model_accuracy(df, df_r2, panel="c"):
    """Fig. 6c or 6d from the source data (columns rank, unit_id, sorting_accuracy, prediction; df_r2: r2_median)"""
    g = MODEL[panel]
    x0, y0, x1, y1 = g["region"]
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    n = len(df)
    x = g["x0"] + np.arange(n) * g["ppr"]
    ax.scatter(x, g["y0"] - df["sorting_accuracy"].values * MODEL_PPU, s=1.95 ** 2, facecolors="w", edgecolors=GREY, linewidths=0.5, zorder=2)
    ax.plot(x, g["y0"] - df["prediction"].values * MODEL_PPU, color=MODEL_RED, linewidth=1.0, solid_joinstyle="round", zorder=3)

    # axes
    _line(ax, [g["spine_x"]] * 2, list(g["spine_y"]))
    for v in (0, 0.5, 1.0):
        yt = g["y0"] - v * MODEL_PPU
        _line(ax, [g["spine_x"] - 3.75, g["spine_x"]], [yt, yt])
        _text(ax, g["ytick_right"], yt + 1.79, f"{v:.1f}", ha="right")
    xa0, xa1, ya = g["axis"]
    _line(ax, [xa0, xa1], [ya, ya])
    for t in range(0, g["x_max"] + 1, 200):
        xt = g["x0"] + t * g["ppr"]
        _line(ax, [xt, xt], [ya, ya + 3.75])
        _text(ax, xt, g["xtick_baseline"], f"{t:d}")
    _text(ax, *g["ylabel"], "Sorting accuracy (ratio)", rotation=90)
    if g["xlabel"]:
        _text(ax, g["xlabel"][1], g["xlabel"][2], g["xlabel"][0])

    # legend (panel c), R-squared and title
    if g["legend"]:
        ax.scatter([213.15], [225.35], s=1.95 ** 2, facecolors="w", edgecolors=GREY, linewidths=0.5, zorder=2)
        _line(ax, [210.23, 216.07], [233.93, 233.93], color=MODEL_RED, lw=3.0)
        _text(ax, 222.1, 227.1, "Data", ha="left")
        _text(ax, 222.2, 235.6, "Regression model", ha="left")
    _text(ax, g["r2"][0], g["r2"][1], f"R\u00b2={df_r2['r2_median'].iloc[0]:.2f}", ha="left")
    _text(ax, *g["title_xy"], g["title"], bold=True, size=TITLE)
    return fig
