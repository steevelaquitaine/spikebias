"""Panels a-f of Fig. 2: single-site voltage traces, drawn at the size of the panel in the figure

author: laquitainesteeve@gmail.com

Geometry measured on panel f of the paper's figure (pt, paper page 11): the data area is 40.8 pt wide for 0-1000 ms and 31.68 pt high for
-84 to 35 uV (the y axis spans the ticks); the axes are detached (the y axis 2.04 pt on the left of the trace, the x axis 1.56 pt below the
y axis). In the layout of Fig. 2 (paper/figures/fig2.pdf) the panels are 1.2414 times larger in both directions (y spine of 39.4 pt, x axis of
50.7 pt). Text 5 pt (guidelines of Nature journals: 5-7 pt).
"""

import numpy as np
from matplotlib import pyplot as plt

FIG2_SCALE = 34.2 / 27.55           # layout of Fig. 2 against the paper's page
TRACE_W, SPINE_H = 40.8, 31.68      # pt on the paper's page (0-1000 ms; -84 to 35 uV)
X_OFFSET, Y_OFFSET = 2.04, 1.56     # detached axes (pt on the paper's page)


def plot_single_site_trace(t_ms, v, color, yticks=(-84, 0, 35), scale=FIG2_SCALE, xlabel="Time (ms)", ylabel=None):
    """Voltage trace of one site (1 s): axes of the size of the paper's panel f; y limits and ticks are `yticks` (min, 0, max)"""
    s = scale
    w, h = TRACE_W * s, SPINE_H * s
    left, right, top, bottom = 24.0, 8.0, 6.0, 20.0
    W, H = left + w + right, bottom + h + top
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([left / W, bottom / H, w / W, h / H])
    ax.plot(t_ms, v, color=color, linewidth=0.5)
    ax.set_xlim(0, 1000)
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_yticks(list(yticks))
    ax.set_xticks([0, 500, 1000])
    ax.tick_params(labelsize=5, length=3, width=0.5, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_position(("outward", X_OFFSET * s))
    ax.spines["bottom"].set_position(("outward", Y_OFFSET * s))
    ax.set_xlabel(xlabel, fontsize=5, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=5, labelpad=1)
    return fig
