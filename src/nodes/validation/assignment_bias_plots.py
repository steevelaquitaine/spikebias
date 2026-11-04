"""Panel of Fig. 5d: ratios of the spike assignment biases of the Kilosort4 single-units, by layer and simulation, drawn from the source data

author: laquitainesteeve@gmail.com

The panel has the geometry of panel d of the paper's figure (page 16 of paper/Laquitaine_NatureNeuro_2025.pdf): one row of bars per simulation (S, NS, DS, E),
one bar per layer (L1, L2/3, L4, L5, L6) with the ratio of the degraded units (merged + partial, white, on top) and of the other units (good + merged + partial,
pink, below) of the single-units, and their counts. Every element is drawn in the pt coordinates of the paper's page (y down), measured on the figure.
All text is Arial 5 pt (titles and the counts of the pink bars in bold).
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

FONT = 5
PINK = (0.96, 0.40, 0.49)
LW = 0.5
REGION = (300.0, 250.0, 530.0, 450.0)
LAYERS = ["L1", "L2/3", "L4", "L5", "L6"]
X_TICKS = [369.68, 380.08, 390.49, 400.89, 411.30]
# rows: experiment -> y of the 0 and of the 1 (top of the bars), axis spine y-range, label baseline, x of the left side of each bar, width of the bars
ROWS = {
    "S": dict(y0=320.84, y1=299.89, spine=(298.84, 320.84), label_y=311.2, x0={"L5": 396.0}, w=6.83),
    "NS": dict(y0=353.29, y1=332.34, spine=(331.29, 353.29), label_y=343.0, x0=dict(zip(LAYERS, [365.61, 376.01, 386.41, 396.82, 407.22])), w=7.8),
    "DS": dict(y0=385.73, y1=364.79, spine=(363.74, 385.73), label_y=375.8, x0=dict(zip(LAYERS, [365.61, 376.01, 386.41, 396.82, 407.22])), w=7.8),
    "E": dict(y0=418.18, y1=397.23, spine=(396.18, 418.18), label_y=410.8, x0={"L2/3": 376.39, "L4": 386.7, "L5": 397.0, "L6": 407.32}, w=7.73),
}
SPINE_X = 360.28


def _text(ax, x, y, s, ha="center", rotation=0, bold=False, va="baseline"):
    ax.text(x, y, s, ha=ha, va=va, rotation=rotation, rotation_mode="anchor", fontsize=FONT, fontweight="bold" if bold else "normal", color="k")


def _line(ax, xs, ys, **kw):
    ax.plot(xs, ys, color="k", linewidth=kw.pop("lw", LW), solid_capstyle="butt", **kw)


def counts(df, sorter="KS4"):
    """Number of degraded (white) and of good + merged + partial (pink) single-units of each simulation and layer"""
    d = df[df["sorter"] == sorter]
    out = {}
    for (exp, layer), g in d.groupby(["experiment", "layer"]):
        pink = g["quality"].str.contains("good").sum()
        out[(exp, layer)] = (int(len(g) - pink), int(pink))
    return out


def plot_source_data(df, sorter="KS4"):
    """Fig. 5d from the source data (columns sorted, quality, layer, sorter, experiment)"""
    x0, y0, x1, y1 = REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    n = counts(df, sorter)

    for exp, g in ROWS.items():
        H = g["y0"] - g["y1"]
        # axis with 5 ticks (0, 0.25, ..., 1) and the labels 0, 0.5, 1
        _line(ax, [SPINE_X] * 2, g["spine"])
        for k in range(5):
            yt = g["y0"] - k * H / 4
            _line(ax, [SPINE_X - 3.76, SPINE_X], [yt, yt])
        for v in (0, 0.5, 1):
            _text(ax, 352.6, g["y0"] - v * H + 1.79, f"{v:g}", ha="right")
        _text(ax, 325.2, g["label_y"], exp)
        # bars
        for layer, xl in g["x0"].items():
            if (exp, layer) not in n:
                continue
            white, pink = n[(exp, layer)]
            tot = white + pink
            hw, hp = H * white / tot, H * pink / tot
            if white:
                ax.add_patch(Rectangle((xl, g["y1"]), g["w"], hw, facecolor="white", edgecolor="k", linewidth=LW))
                _text(ax, xl + g["w"] / 2, g["y1"] + hw / 2 + 1.79, str(white))
            if pink:
                ax.add_patch(Rectangle((xl, g["y1"] + hw), g["w"], hp, facecolor=PINK, edgecolor="none"))
                _text(ax, xl + g["w"] / 2, g["y1"] + hw + hp / 2 + 1.79, str(pink), bold=True)

    # x ticks (below the last row) and labels
    for x, layer in zip(X_TICKS, LAYERS):
        _line(ax, [x, x], [419.73, 423.48])
        _text(ax, x, 431.3, layer)
    _text(ax, 389.6, 443.4, "Model cortical layer")
    _text(ax, 311.7, 357.45, "Ground truth simulation", rotation=90)
    _text(ax, 336.5, 358.65, "Single-unit ratio", rotation=90)
    _text(ax, 420.45, 259.7, "Predicted spike assignment biases among single-units", bold=True)

    # legend
    _text(ax, 474.5, 350.8, "Unit quality", bold=True)
    ax.add_patch(Rectangle((436.9, 363.78), 11.69, 4.09, facecolor="white", edgecolor="k", linewidth=LW))
    ax.add_patch(Rectangle((436.88, 373.42), 11.69, 4.1, facecolor=PINK, edgecolor="none"))
    _text(ax, 451.9, 367.9, "merged + partial (degraded)", ha="left")
    _text(ax, 451.5, 377.5, "good + merged + partial", ha="left")
    return fig
