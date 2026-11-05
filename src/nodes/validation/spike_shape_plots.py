"""Panels of Supplementary Fig. 3 drawn from the source data and rescaled to the height the journal guidelines allow

author: laquitainesteeve@gmail.com

The arrangement of the panels and of their subplots is that of paper/figures/supp_fig3.pdf (positions measured on the vector figure, see
spike_shape_layout.py), with every length multiplied by SCALE so that the whole figure is at most 185 mm high instead of the paper's 241 mm. Text is Arial 5 pt (titles
7 pt, bold) whatever the scale, and editable.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from src.nodes.validation.spike_shape_layout import A_SUBPLOTS, B_SUBPLOTS

FONT, TITLE = 5, 7
TARGET_HEIGHT_PT = 185.0 / 25.4 * 72     # 185 mm: figure height for a legend of less than 300 words
SCALE = 0.735                            # the stacked panels a, c and d-f (699.7 pt of the paper's layout, with their margins) are 514 pt = 181.5 mm high
LW = 0.5
RED, ORANGE = (0.8392, 0.1529, 0.1569), (1.0, 0.498, 0.0549)
BLUE, PURPLE = (0.1216, 0.4667, 0.7059), (0.5804, 0.4039, 0.7412)


def _ink(s, size=FONT, bold=False, rotation=0):
    path = TextPath((0, 0), s, size=size, prop=FontProperties(family="Arial", weight="bold" if bold else "normal"))
    return path.transformed(Affine2D().rotate_deg(rotation)).get_extents()


def _ctext(ax, cx, cy, s, size=FONT, bold=False, rotation=0, color="k"):
    """Text whose ink bounding box is centred on (cx, cy) (pt of the scaled figure, y down)"""
    e = _ink(s, size, bold, rotation)
    ax.text(cx - (e.x0 + e.x1) / 2, cy + (e.y0 + e.y1) / 2, s, ha="left", va="baseline", rotation=rotation, rotation_mode="anchor", fontsize=size,
            fontweight="bold" if bold else "normal", color=color)


def _rtext(ax, right, baseline, s, size=FONT, color="k"):
    e = _ink(s, size)
    ax.text(right - e.x1, baseline, s, ha="left", va="baseline", fontsize=size, color=color)


def _number(v, decimals):
    return f"{v:.{decimals}f}".replace("-", "−")


def _decimals(ticks):
    d = 0
    for v in ticks:
        while d < 4 and abs(v - round(v, d)) > 1e-9:
            d += 1
    return max(d, 1)


class _Canvas:
    """Canvas in the pt of the paper's figure multiplied by SCALE (y down); the coordinates given to the methods are those of the paper"""

    def __init__(self, region):
        x0, y0, x1, y1 = region
        self.x0, self.y0 = x0, y0
        self.fig = plt.figure(figsize=((x1 - x0) * SCALE / 72, (y1 - y0) * SCALE / 72))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, (x1 - x0) * SCALE)
        self.ax.set_ylim((y1 - y0) * SCALE, 0)
        self.ax.axis("off")

    def p(self, x, y):
        return (np.asarray(x) - self.x0) * SCALE, (np.asarray(y) - self.y0) * SCALE

    def line(self, xs, ys, lw=LW, color="k", **kw):
        X, Y = self.p(xs, ys)
        self.ax.plot(X, Y, color=color, linewidth=lw, solid_capstyle="butt", **kw)

    def text(self, cx, cy, s, **kw):
        X, Y = self.p(cx, cy)
        _ctext(self.ax, float(X), float(Y), s, **kw)

    def rtext(self, right, baseline, s, **kw):
        X, Y = self.p(right, baseline)
        _rtext(self.ax, float(X), float(Y), s, **kw)

    def rect(self, x0, y0, x1, y1, color):
        X, Y = self.p([x0, x1], [y0, y1])
        self.ax.add_patch(Rectangle((X[0], Y[0]), X[1] - X[0], Y[1] - Y[0], facecolor=color, edgecolor=color, linewidth=0.5))


def _waveforms(df, subplots, region, colors, title, legend, ylabel, xlabels, annotations=()):
    cv = _Canvas(region)
    for s in subplots:
        g = df[(df["column"] == s["col"]) & (df["row"] == s["row"])]
        Y = lambda v: s["y_top"] + (s["vmax"] - np.asarray(v)) / (s["vmax"] - s["vmin"]) * (s["y_bot"] - s["y_top"])
        X = lambda t: s["t0_x"] + (np.asarray(t) + 1.0) * s["ppm"]
        for electrode, color in zip(("above", "below"), colors):
            d = g[g["electrode"] == electrode].sort_values("time_ms")
            cv.line(X(d["time_ms"].values), Y(d["voltage_mV"].values), color=color, zorder=3)
        sx, ya = s["spine_x"], s["axis"][2]
        cv.line([sx, sx], [s["y_top"], s["y_bot"]])
        cv.line(s["axis"][:2], [ya, ya])
        dec = _decimals(s["yticks"])
        for v in s["yticks"]:
            y = float(Y(v))
            cv.line([sx - 3.86, sx], [y, y])
            cv.rtext(sx - 7.9, y + 1.8 / SCALE * 0.5, _number(v, dec))
        for xt, label in zip(s["xticks_pt"], ("−1", "0", "3")):
            cv.line([xt, xt], [ya, ya + 3.86])
            cv.text(xt, ya + 9.7, label)
        cv.rtext(s["axis"][1] - 2.2, s["y_bot"] + 0.2, s["mtype"])
    cv.text(title[0], title[1], title[2], size=TITLE, bold=True)
    for (hx0, hy0, hx1, hy1, color, tx, ty, label) in legend:
        cv.rect(hx0, hy0, hx1, hy1, color)
        X, Y = cv.p(tx, ty)
        e = _ink(label)
        cv.ax.text(float(X) - e.x0, float(Y) + (e.y0 + e.y1) / 2, label, ha="left", va="baseline", fontsize=FONT)
    cv.text(ylabel[0], ylabel[1], "Voltage (mV)", rotation=90) if ylabel else None
    for cx, cy in xlabels:
        cv.text(cx, cy, "Time (ms)")
    for cx, cy, s_, color in annotations:
        cv.text(cx, cy, s_, color=color)
    return cv.fig


def plot_excitatory(df):
    """Supplementary Fig. 3a: spike-triggered average of 16 excitatory neurons (columns synapse_class, column, row, mtype, electrode, time_ms, voltage_mV)"""
    legend = [(107.07, 12.55, 119.87, 14.52, RED, 122.8, 13.35, "Above somata"), (107.0, 20.8, 119.8, 22.77, ORANGE, 123.4, 21.35, "Below somata")]
    annotations = [(89.7, 32.2, "Positive spikes", RED), (94.8, 47.3, "Regular spike", ORANGE), (87.0, 284.6, "Triphasic spike", RED), (87.9, 311.5, "Regular spike", ORANGE)]
    return _waveforms(df, A_SUBPLOTS, (13.6, -5.0, 223.56, 444.2), (RED, ORANGE), (136.55, -0.5, "Excitatory model neurons"), legend, (16.4, 204.05),
                      [(91.3, 441.55), (196.1, 441.55)], annotations)


def plot_inhibitory(df):
    """Supplementary Fig. 3b: spike-triggered average of 20 inhibitory neurons"""
    legend = [(350.84, 13.84, 363.65, 15.81, BLUE, 366.6, 14.65, "Above somata"), (350.77, 22.08, 363.58, 24.05, PURPLE, 367.2, 22.65, "Below somata")]
    return _waveforms(df, B_SUBPLOTS, (266.0, -5.0, 477.5, 419.4), (BLUE, PURPLE), (385.7, -0.5, "Inhibitory model neurons"), legend, None, [(342.8, 416.7), (451.4, 416.7)])
