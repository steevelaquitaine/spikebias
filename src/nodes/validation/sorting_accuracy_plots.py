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


# ---------------------------------------------------------------------------------------------------------------------------------
# Fig. 6e-f: weights of the z-scored unit features of the model of sorting accuracy (bars), the weights and their stars written on the right
# Geometry measured on panels e and f of the paper's figure (page 19). Panel f has the rows (features) of panel e: it has no feature labels.
PINK_W = (0.95, 0.40, 0.49)
WEIGHTS = {
    "e": dict(region=(45.0, 420.0, 252.0, 652.0), spine=108.29, spine_y=(455.05, 617.05), tick_y0=460.0, tick_dy=7.607, bar_h=6.09, zero=148.72, ppu=22.58, axis=(112.36, 193.89, 625.15),
              label_right=100.8, value_x=171.6, star_x=189.5, ylabel=(54.4, 532.3), subtitle="Spontaneous model", subtitle_xy=(143.2, 448.2),
              title="ks4 sorting accuracy biases for unit features (Biophy, NPX)", title_xy=(156.05, 430.8), xlabel_xy=(207.4, 647.5), xtick_baseline=636.8, xticks=(-1, 0, 1, 2)),
    "f": dict(region=(205.0, 436.0, 300.0, 645.0), spine=None, spine_y=None, tick_y0=460.0, tick_dy=7.607, bar_h=6.09, zero=243.91, ppu=23.2, axis=(208.79, 290.31, 625.13),
              label_right=None, value_x=260.4, star_x=276.6, ylabel=None, subtitle="Evoked model", subtitle_xy=(238.4, 448.4), title=None, title_xy=None, xlabel_xy=None,
              xtick_baseline=636.75, xticks=(-1, 0, 1, 2)),
}


def plot_feature_weights(df, panel="e", order=None):
    """Fig. 6e or 6f from the source data (columns feature, weight, std_error, p_value, significance, n_neurons).
    The rows are the features sorted by weight (panel e); panel f has the order of panel e (`order`: list of features)."""
    g = WEIGHTS[panel]
    x0, y0, x1, y1 = g["region"]
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    if order is None:
        d = df.sort_values("weight").reset_index(drop=True)
    else:
        d = df.set_index("feature").loc[order].reset_index()
    if g["spine"] is not None:
        _line(ax, [g["spine"]] * 2, list(g["spine_y"]))
    for i, r in d.iterrows():
        yc = g["tick_y0"] + i * g["tick_dy"]
        sig = isinstance(r["significance"], str) and r["significance"] != ""
        xe = g["zero"] + r["weight"] * g["ppu"]
        ax.add_patch(plt.Rectangle((min(g["zero"], xe), yc - g["bar_h"] / 2), abs(xe - g["zero"]), g["bar_h"], facecolor=PINK_W if sig else (0.06, 0.06, 0.05), edgecolor="none"))
        if g["spine"] is not None:
            _line(ax, [g["spine"] - 3.75, g["spine"]], [yc, yc])
            _text(ax, g["label_right"], yc + 1.79, r["feature"], ha="right")
        _text(ax, g["value_x"], yc + 1.79, f"{round(r['weight'], 2):g}", ha="left")
        if sig:
            _text(ax, g["star_x"], yc + 2.6, r["significance"], ha="left")
    xa0, xa1, ya = g["axis"]
    _line(ax, [xa0, xa1], [ya, ya])
    for v in g["xticks"]:
        xt = g["zero"] + v * g["ppu"]
        _line(ax, [xt, xt], [ya, ya + 3.75])
        _text(ax, xt, g["xtick_baseline"], f"{v:d}".replace("-", "\u2212"))
    if g["xlabel_xy"]:
        _text(ax, *g["xlabel_xy"], "Weights of z-scored features")
    if g["ylabel"]:
        _text(ax, *g["ylabel"], "Unit features", rotation=90)
    _text(ax, *g["subtitle_xy"], g["subtitle"], bold=True, size=TITLE)
    if g["title"]:
        _text(ax, *g["title_xy"], g["title"], bold=True, size=TITLE)
    return fig


# ---------------------------------------------------------------------------------------------------------------------------------
# Fig. 6g: probability distribution of the model neuron types of the evoked biophysical ground truth (blue line) and number of neurons of each type
# Geometry measured on panel g of the paper's figure (page 19). Text is Arial 5 pt, the title 7 pt (bold).
# canvas: the bounding box of the panel of the paper (x from the left of 'EXC' to the right of the largest count, y from the top of the title to the bottom of the
# x label of panel h), so that panels g and h have the same height and the same vertical origin: their rows are aligned when they are placed at the same y
FIG6GH_Y = (233.8, 643.4)
TYPES_REGION = (299.5, FIG6GH_Y[0], 428.62, FIG6GH_Y[1])
TYPES = dict(spine_x=372.08, y0=300.28, dy=9.0545, x0=375.46, ppu=23.02 / 0.2, axis=(374.3, 418.77, 612.74), label_right=363.7, count_dx=1.6)
BLUE = (0.22, 0.46, 0.73)


def plot_neuron_type_distribution(df):
    """Fig. 6g from the source data (columns synapse_class, layer, etype, count, probability, n_neurons), rows in the order of the file"""
    g = TYPES
    x0, y0, x1, y1 = TYPES_REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    n = len(df)
    yr = g["y0"] + np.arange(n) * g["dy"]
    xr = g["x0"] + df["probability"].values * g["ppu"]
    ax.plot(xr, yr, color=BLUE, linewidth=1.0, solid_joinstyle="round", zorder=2)
    _line(ax, [g["spine_x"]] * 2, [yr[0], yr[-1]])
    for i in range(n):
        _line(ax, [g["spine_x"] - 1.46, g["spine_x"]], [yr[i]] * 2, lw=0.3)
        _text(ax, g["label_right"], yr[i] + 1.79, f"'{df['layer'].iloc[i]}', {df['etype'].iloc[i]}", ha="right")
        _text(ax, xr[i] + g["count_dx"], yr[i] + 0.1, str(df["count"].iloc[i]), ha="left")
    # probability axis
    xa0, xa1, ya = g["axis"]
    _line(ax, [xa0, xa1], [ya, ya])
    for v in (0.0, 0.2):
        xt = g["x0"] + v * g["ppu"]
        _line(ax, [xt, xt], [ya, ya + 1.46], lw=0.3)
        _text(ax, xt, 621.0, f"{v:.1f}")
    _text(ax, 395.8, 631.4, "Probability")
    _text(ax, 396.55, 637.7, "(ratio)")
    # synapse classes
    for syn, (xc, yc) in {"EXC": (308.95, 315.15), "INH": (309.55, 455.35)}.items():
        _text(ax, xc, yc + 1.79, syn)
    _line(ax, [323.17] * 2, [296.96, 330.91])
    _line(ax, [323.38, 324.08], [335.79, 607.12])
    _text(ax, 334.45, 286.3, "Model neuron type")
    # titles
    _text(ax, 372.95, 241.5, "Biophysical ground truth", bold=True, size=TITLE)
    _text(ax, 373.9, 249.7, "neuron type distribution", bold=True, size=TITLE)
    _text(ax, 374.0, 257.8, "(evoked model)")
    return fig


# Fig. 6h: same rows (y) and canvas height as Fig. 6g so that the two panels line up
BIAS_COLORS = {"increased": (0.93, 0.12, 0.14), "same": (0.0, 0.0, 0.0), "reduced": (0.23, 0.30, 0.66)}
BIAS_REGION = (432.69, FIG6GH_Y[0], 542.75, FIG6GH_Y[1])
BIAS = dict(spine_x=433.69, x_end=542.75, y_axis=612.69, col_x0=443.0, col_dx=18.12, tick=1.46, label_dx=1.5, label_y=620.8)


def plot_selection_biases(df):
    """Fig. 6h from the source data (columns type_index, sorter, ratio, bias): ratio of the probability of each neuron type after spike sorting to that in
    the ground truth, by sorter (columns) and neuron type (rows, same y positions as Fig. 6g); red/blue bold: increased/reduced probability (p < 0.05)"""
    g, gg = BIAS, TYPES
    x0, y0, x1, y1 = BIAS_REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    sorters = list(dict.fromkeys(df["sorter"]))
    n = df["type_index"].nunique()
    yr = gg["y0"] + np.arange(n) * gg["dy"]
    xc = g["col_x0"] + np.arange(len(sorters)) * g["col_dx"]
    for j, sorter in enumerate(sorters):
        d = df[df["sorter"] == sorter].sort_values("type_index")
        for i, (ratio, bias) in enumerate(zip(d["ratio"], d["bias"])):
            _text(ax, xc[j], yr[i] + 1.79, f"{ratio:.1f}", bold=bias != "same", color=BIAS_COLORS[bias])
        _line(ax, [xc[j]] * 2, [g["y_axis"], g["y_axis"] + g["tick"]], lw=0.3)
        _text(ax, xc[j] + g["label_dx"], g["label_y"], sorter, ha="right", rotation=45)
    _line(ax, [g["spine_x"]] * 2, [yr[0] - gg["dy"] / 2, g["y_axis"]])
    _line(ax, [g["spine_x"], g["x_end"]], [g["y_axis"]] * 2)
    for y in yr:
        _line(ax, [g["spine_x"] - g["tick"], g["spine_x"]], [y, y], lw=0.3)
    _text(ax, 492.8, 641.95, "Spike sorters")
    # title, subtitle and legend
    _text(ax, 490.05, 241.5, "Selection biases", bold=True, size=TITLE)
    _text(ax, 490.3, 252.2, "Probability after spike sorting divided")
    _text(ax, 491.1, 259.2, "by probability in ground truth (ratio)")
    for (xs, ys, label, key, base) in [(446.28, 266.36, "Increased probability (p<0.05)", "increased", 270.5),
                                       (446.28, 274.11, "Same probability (p>=0.05)", None, 278.1),
                                       (446.28, 281.68, "Reduced probability (p<0.05)", "reduced", 285.85)]:
        color = (0.69, 0.69, 0.68) if key is None else BIAS_COLORS[key]
        ax.add_patch(plt.Rectangle((xs, ys), 9.64, 3.58, color=color, linewidth=0))
        _text(ax, 459.2, base, label, ha="left", color="k" if key is None else color)
    return fig
