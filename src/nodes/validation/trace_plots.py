"""Panels c to g of Supplementary Fig. 2: first 4 seconds of the raw voltage trace of one site in each layer, drawn from the source data

author: laquitainesteeve@gmail.com

Each panel has the geometry of its panel in paper/figures/supp_fig2.pdf (sizes in pt, y down): the five traces (grey, 0.3 pt) span the minimum and maximum
of the 4 seconds (the ends of their y axes, which are labelled with these two values), time axis of 4 s with ticks at 0 and 4, layer labels (coloured),
title and axis labels. Text is Arial 5 pt (title 7 pt, bold), editable.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

FONT, TITLE = 5, 7
COLUMNS = ["L1_voltage_uV", "L2_3_voltage_uV", "L4_voltage_uV", "L5_voltage_uV", "L6_voltage_uV"]
LAYER_COLORS = {"L1": (0.7137, 0.6157, 0.3765), "L2/3": (0.7608, 0.4549, 0.5843), "L4": (0.6431, 0.5333, 0.6627), "L5": (0.3294, 0.4274, 0.498), "L6": (0.5647, 0.7294, 0.5098)}
TRACE_COLOR, TRACE_LW, AXIS_LW, TICK_LW = (0.298, 0.298, 0.298), 0.3004, 1.0013, 1.0414

# Fig. S2c: in vivo Neuropixels (Marques-Smith et al.): canvas 151.2 x 159.5 pt
IN_VIVO_NPX = dict(
    region=(15.95, 310.18, 167.19, 469.69), t0=54.24, t1=165.62, spine_x=53.13, tick_x0=49.28, label_right=45.38, x_axis_y=447.15,
    traces=[(331.14, 348.97), (355.46, 373.29), (379.79, 397.62), (404.11, 421.94), (428.42, 446.26)],
    title=("Anesthetized (Marques-Smith et al.)", 108.0, 316.56), ylabel=("Neuropixels Single-site voltage trace (µV)", 20.6, 377.8), xlabel=("Time (seconds)", 109.67, 467.0),
    layer_labels={"L1": (159.6, 332.68), "L2/3": (159.6, 354.57), "L4": (159.95, 378.97), "L5": (161.3, 402.72), "L6": (161.3, 424.32)}, x_tick_labels=((54.24, 456.9), (165.6, 456.88)))

# Fig. S2d: simulated Neuropixels, spontaneous regime: canvas 130.1 x 152.1 pt, widened to the 7 pt title; the y axes span the rounded minimum and maximum
SIM_NPX_SPONT = dict(
    region=(198.39, 313.93, 328.49, 466.0), rounded=True, t0=230.44, t1=323.05, spine_x=224.79, tick_x0=220.95, label_right=217.05, x_axis_y=443.46, x_axis=(225.81, 327.67),
    traces=[(327.36, 345.36), (351.84, 369.84), (376.33, 394.32), (400.81, 418.8), (425.29, 443.29)],
    title=("Biophy. model of the spontaneous regime", 269.2, 316.76), ylabel=None, xlabel=("Time (seconds)", 276.5, 463.3),
    layer_labels={"L1": (316.75, 327.1), "L2/3": (316.7, 348.99), "L4": (317.1, 373.4), "L5": (318.4, 397.1), "L6": (318.4, 418.75)}, x_tick_labels=((230.44, 453.2), (323.0, 453.2)))

# Fig. S2e: simulated Neuropixels, evoked regime: canvas 129.4 x 153.0 pt, widened to the 7 pt title if it is wider
SIM_NPX_EVOKED = dict(
    region=(355.4, 313.39, 484.77, 466.41), rounded=True, t0=387.46, t1=480.14, spine_x=381.8, tick_x0=377.95, label_right=374.05, x_axis_y=443.88, x_axis=(382.82, 484.77),
    traces=[(327.77, 345.77), (352.25, 370.25), (376.74, 394.73), (401.22, 419.21), (425.7, 443.7)],
    title=("Biophy. model of the evoked regime", 427.68, 316.2), ylabel=None, xlabel=("Time (seconds)", 433.5, 463.7),
    layer_labels={"L1": (474.0, 325.53), "L2/3": (473.96, 347.43), "L4": (474.35, 371.83), "L5": (475.7, 395.58), "L6": (475.7, 417.18)}, x_tick_labels=((387.45, 453.6), (480.1, 453.6)))

# Fig. S2f: in vivo dense probe (Horvath et al.): canvas 145.9 x 153.0 pt
IN_VIVO_DENSE = dict(
    region=(110.84, 481.72, 256.71, 634.68), t0=153.3, t1=255.14, spine_x=152.28, tick_x0=148.43, label_right=144.5, x_axis_y=612.14,
    traces=[(496.13, 513.96), (520.45, 538.29), (544.77, 562.61), (569.09, 586.93), (593.41, 611.25)],
    title=("Anesthetized (Horvath et al.)", 201.08, 484.55), ylabel=("Dense probe single-site voltage trace (\u00b5V)", 113.66, 554.07), xlabel=("Time (seconds)", 203.95, 632.0),
    layer_labels={"L1": (249.41, 495.21), "L2/3": (249.37, 517.1), "L4": (249.77, 542.5), "L5": (251.1, 566.25), "L6": (251.1, 590.87)}, x_tick_labels=((153.3, 621.88), (255.1, 621.87)))

# Fig. S2g: simulated dense probe, spontaneous regime: canvas 131.5 x 151.6 pt, widened to the 7 pt title if it is wider
SIM_DENSE_SPONT = dict(
    region=(276.37, 479.87, 407.9, 631.51), rounded=True, t0=303.77, t1=403.64, spine_x=302.77, tick_x0=298.92, label_right=295.02, x_axis_y=608.97,
    traces=[(492.96, 510.79), (517.28, 535.11), (541.6, 559.43), (565.92, 583.75), (590.25, 608.08)],
    title=("Biophy. model of the spontaneous regime", 348.62, 482.7), ylabel=None, xlabel=("Time (seconds)", 353.44, 628.83),
    layer_labels={"L1": (397.6, 491.6), "L2/3": (397.57, 513.49), "L4": (397.97, 538.89), "L5": (399.3, 562.64), "L6": (399.3, 587.25)}, x_tick_labels=((303.76, 618.71), (403.6, 618.7)))


def _ctext(ax, cx, cy, s, rotation=0, bold=False, size=FONT, color="k"):
    path = TextPath((0, 0), s, size=size, prop=FontProperties(family="Arial", weight="bold" if bold else "normal"))
    ext = path.transformed(Affine2D().rotate_deg(rotation)).get_extents()
    ax.text(cx - (ext.x0 + ext.x1) / 2, cy + (ext.y0 + ext.y1) / 2, s, ha="left", va="baseline", rotation=rotation, rotation_mode="anchor", fontsize=size,
            fontweight="bold" if bold else "normal", color=color)


def _number(v, decimals=0):
    return f"{v:.{decimals}f}".replace("-", "−")


def plot_traces(df, panel=IN_VIVO_NPX):
    """One panel from the source data (columns time_s, L1_voltage_uV, L2_3_voltage_uV, L4_voltage_uV, L5_voltage_uV, L6_voltage_uV)"""
    g = panel
    x0, y0, x1, y1 = g["region"]
    tw = TextPath((0, 0), g["title"][0], size=TITLE, prop=FontProperties(family="Arial", weight="bold")).get_extents().width
    x0, x1 = min(x0, g["title"][1] - tw / 2), max(x1, g["title"][1] + tw / 2)      # the 7 pt title is wider than the paper's (about 6 pt): the canvas is widened to it
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    t = df["time_s"].values
    X = lambda ts: g["t0"] + (ts - t.min()) / (t.max() - t.min()) * (g["t1"] - g["t0"])

    def line(xs, ys, lw):
        ax.plot(xs, ys, color="k", linewidth=lw, solid_capstyle="butt")

    for col, (yt, yb), (layer, color) in zip(COLUMNS, g["traces"], LAYER_COLORS.items()):
        v = df[col].values
        lo, hi = (np.floor(v.min()), np.ceil(v.max())) if g.get("rounded") else (v.min(), v.max())
        ys = yt + (hi - v) / (hi - lo) * (yb - yt)
        ax.plot(X(t), ys, color=TRACE_COLOR, linewidth=TRACE_LW, solid_joinstyle="miter")
        line([g["spine_x"]] * 2, [yt, yb], AXIS_LW)
        for y in (yt, yb):
            line([g["tick_x0"], g["spine_x"]], [y, y], TICK_LW)
        for val, y in ((hi, yt + 0.1), (lo, yb + 0.1)):
            s = _number(val, 1 if g.get("rounded") else 0)
            path = TextPath((0, 0), s, size=FONT, prop=FontProperties(family="Arial")).get_extents()
            ax.text(g["label_right"] - path.x1, y + (path.y0 + path.y1) / 2, s, ha="left", va="baseline", fontsize=FONT)
        _ctext(ax, *g["layer_labels"][layer], layer, bold=True, color=color)
    line(g.get("x_axis", (g["t0"], g["t1"])), [g["x_axis_y"]] * 2, AXIS_LW)
    for x in (g["t0"], g["t1"]):
        line([x, x], [g["x_axis_y"], g["x_axis_y"] + 3.85], TICK_LW)
    for (cx, cy), s in zip(g["x_tick_labels"], ("0", "4")):
        _ctext(ax, cx, cy, s)
    _ctext(ax, g["title"][1], g["title"][2], g["title"][0], bold=True, size=TITLE)
    if g["ylabel"]:
        _ctext(ax, g["ylabel"][1], g["ylabel"][2], g["ylabel"][0], rotation=90)
    _ctext(ax, g["xlabel"][1], g["xlabel"][2], g["xlabel"][0])
    return fig
