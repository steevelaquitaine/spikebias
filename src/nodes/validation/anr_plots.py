"""Panels of Fig. 2q: distribution of the trace amplitude-to-noise ratio (ANR) in each layer, drawn from the source data

author: laquitainesteeve@gmail.com

Each layer is one panel with the geometry of panel q in paper/figures/fig2.pdf (sizes in pt): the data area is 46.8 pt wide (from the
smallest to the largest ANR with data) and 3.8 pt per decade of probability (from 10^-7.6 to 10^0.3); the axes are detached (R style); the
x ticks are the smallest ANR, 0 and the largest ANR; layer label in bold.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

SOURCE = "figures/8_source_data/fig2q/fig2q_{layer}_{exp}.xlsx"
LAYER_LABELS = {"l1": "L1", "l23": "L2/3", "l4": "L4", "l5": "L5", "l6": "L6"}
# experiments in plotting order (as in the paper's notebook) and colours
EXPERIMENTS = {
    "npx_v": np.array([153, 153, 153]) / 255,   # Marques-Smith (light grey)
    "npx_s": [0.9, 0.14, 0.15],                  # biophysical spontaneous (red)
    "npx_e": [1, 0.49, 0],                       # biophysical evoked (orange)
    "npx_b": [0.22, 0.5, 0.72],                  # synthetic (blue, layer 5 only)
    "dense_v": [0.2, 0.2, 0.2],                  # Horvath (dark grey)
    "dense_s": np.array([26, 152, 80]) / 255,    # dense-probe biophysical (green)
}

# geometry (pt), measured on panel q of paper/figures/fig2.pdf
DATA_W = 46.8           # width of the data area
DECADE = 3.8            # height of one decade of probability
Y_MIN, Y_MAX = 10 ** -7.6, 10 ** 0.3
LEFT, RIGHT, TOP, BOTTOM = 22.0, 6.0, 3.0, 13.5          # margins around the data area
SPINE_OFFSET_X, SPINE_OFFSET_Y = 2.4, 4.0                # detached axes


def load(layer):
    """Source data of a layer: {experiment: dataframe with x, mean, ci}"""
    out = {}
    for exp in EXPERIMENTS:
        try:
            out[exp] = pd.read_excel(SOURCE.format(layer=layer, exp=exp))
        except FileNotFoundError:
            continue
    return out


def plot_layer(layer, data=None, ylabel=False, xlabel=False):
    """One panel of Fig. 2q, at its final size in the figure"""
    data = data if data is not None else load(layer)
    x_min = min(d.loc[d["mean"] > 0, "x"].min() for d in data.values())
    x_max = max(d.loc[d["mean"] > 0, "x"].max() for d in data.values())
    data_h = (np.log10(Y_MAX) - np.log10(Y_MIN)) * DECADE
    left = LEFT + (16.0 if ylabel else 0.0)
    bottom = BOTTOM + (8.0 if xlabel else 0.0) + (5.0 if ylabel else 0.0)   # the y label is taller than the data area
    right = RIGHT + (10.0 if xlabel else 0.0)              # room for the x label, wider than the data area
    top = TOP + (5.0 if ylabel else 0.0)
    W, H = left + DATA_W + right, bottom + data_h + top
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([left / W, bottom / H, DATA_W / W, data_h / H])
    for exp, d in data.items():
        color = EXPERIMENTS[exp]
        ax.plot(d["x"], d["mean"], color=color, linestyle="-", linewidth=1, marker="None")
        ax.fill_between(d["x"], d["mean"], d["mean"] + d["ci"], color=color, linewidth=0.1, alpha=0.4, rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([1, 1e-3, 1e-6], [r"$10^{0}$", r"$10^{-3}$", r"$10^{-6}$"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks([x_min, 0, x_max], [f"{x_min:.0f}", "0", f"{x_max:.0f}"])
    ax.tick_params(labelsize=5, length=2, width=0.5, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_position(("outward", SPINE_OFFSET_X))
    ax.spines["bottom"].set_position(("outward", SPINE_OFFSET_Y))
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.text(0.015, 0.97, LAYER_LABELS[layer], transform=ax.transAxes, fontsize=6, fontweight="bold", va="top", ha="left")
    if ylabel:
        ax.set_ylabel("Probability (ratio)", fontsize=5, labelpad=3)
    if xlabel:
        ax.set_xlabel("Amplitude-to-noise ratio", fontsize=6, labelpad=1)
    return fig
