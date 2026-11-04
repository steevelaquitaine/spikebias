"""Panel of Fig. 3d: extracellular spike of a layer 5 pyramidal cell in the traces summed over all cells, drawn from the source data

author: laquitainesteeve@gmail.com

Geometry of the panel in the paper (page 12 of paper/Laquitaine_NatureNeuro_2025.pdf, measured at 600 dpi and drawn here in the same pixel units, 1 px = 0.12 pt):
5 rows x 4 columns of sites, each trace scaled to its own range; one grey per row (light, dark, darkest at the soma row, medium, light); black soma dot with a
blue edge, arrow and label "Somata"; "200 um" bar (vertical) and "75 um" bar (horizontal); title in bold.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

PX = 0.12                                               # pt per pixel
X0, X1, Y0, Y1 = -80, 840, -10, 905                     # drawing area (px)
GRID_X0, COL_W = 135, 130                               # left edge and width of a column
ROW_Y = [220, 345, 465, 590, 715]                       # centre of each row
HALF_H = 55                                             # half height of a trace
SOMA = (405, 442)
SOMA_R = 32
FONT = 7


def trace_xy(group, row, col, gap=0.0):
    """Pixel coordinates of one trace, scaled to fill its cell"""
    t, v = group["time"].values.astype(float), group["voltage"].values
    x = GRID_X0 + col * COL_W + (t - t.min()) / (t.max() - t.min()) * COL_W * (1 - gap)
    y = ROW_Y[row] - ((v - (v.max() + v.min()) / 2) / (v.max() - v.min()) * 2 * HALF_H)
    return x, y


def plot_source_data(df):
    """Fig. 3d from the source data (columns row, col, channel, time, voltage, r, g, b, a)"""
    fig = plt.figure(figsize=((X1 - X0) * PX / 72, (Y1 - Y0) * PX / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y1, Y0)
    ax.axis("off")

    for (row, col, ch), group in df.groupby(["row", "col", "channel"]):
        x, y = trace_xy(group, row, col)
        ax.plot(x, y, color=group[["r", "g", "b", "a"]].iloc[0].values.astype(float), linewidth=0.6, solid_joinstyle="round")

    # soma, arrow and label
    ax.add_patch(Circle(SOMA, SOMA_R, facecolor="black", edgecolor=(0, 0, 0.8), linewidth=0.4, zorder=5))
    ax.annotate("", xy=(392, 403), xytext=(58, 403), arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.13", color="black", linewidth=1.0, shrinkA=0, shrinkB=0))
    ax.text(100, 362, "Somata", ha="right", va="center", fontsize=5.2)

    # 200 um bar
    ax.plot([716, 716], [405, 462], color="black", linewidth=0.3, solid_capstyle="butt")
    ax.add_patch(FancyBboxPatch((696, 464), 18, 142, boxstyle="round,pad=0,rounding_size=9", facecolor="black", edgecolor="black", linewidth=0.5))
    ax.text(733, 437, "200", ha="left", va="center", fontsize=FONT)
    ax.text(782, 512, r"$\mathit{\mu}$m", ha="center", va="center", fontsize=FONT)

    # 75 um bar
    ax.add_patch(FancyBboxPatch((328, 811), 142, 20, boxstyle="round,pad=0,rounding_size=10", facecolor="black", edgecolor="black", linewidth=0.5))
    ax.text(441, 878, r"75 $\mathit{\mu}$m", ha="center", va="center", fontsize=FONT)

    ax.text(395, 14, "Traces summed over all cells", ha="center", va="center", fontsize=6.4, fontweight="bold")
    return fig
