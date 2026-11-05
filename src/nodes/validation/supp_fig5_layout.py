"""Layout of the panels of Supplementary Fig. 5 (yield per site, spike assignment biases, CEBRA curation), drawn from the source data

author: laquitainesteeve@gmail.com

The axes have the position and size of the axes of paper/figures/supp_fig5.pdf (pt of the page, y down, measured on the figure; page 513 x 503 pt).
Text is Arial 5 pt, except the titles (7 pt, bold). The text of each panel is positioned with text_pt (centre at the page coordinates).
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle

from src.nodes.validation.supp_fig4_layout import add_axes_pt, new_figure, set_text_sizes, text_pt

FONT, TITLE = 5, 7

# panel a: yield per site (bars): axes left/top/width/height, legends (patch left x, tops of the patches, 12 x 4.2 pt)
A_AXES = (100.14, 26.68, 168.0, 55.55)
A_CANVAS = (0.0, 0.0, 513.0, 135.0)
A_LEGEND_NPX = dict(title_xy=(322.0, 30.0), patch_x=296.5, tops=[41.8, 50.6, 59.5, 68.1], text_x=313.0)
A_LEGEND_DENSE = dict(title_xy=(392.0, 30.0), patch_x=388.7, tops=[50.6, 59.5], text_x=405.0)

# panel b: spike assignment biases: 4 rows (S, NS, DS, E) x 6 sorters, axes of 55.6 x 22.6 pt
B_LEFT = [54.2, 116.5, 178.8, 241.1, 303.4, 365.7]
B_TOP = [164.3, 197.7, 231.1, 264.5]
B_W, B_H = 55.6, 22.6
B_CANVAS = (0.0, 128.0, 513.0, 225.0)
B_LEGEND = dict(patch_x=429.9, tops=[233.8, 243.5, 252.3, 262.2, 271.0], title_xy=(467.0, 224.0), text_x=445.0)

# panels c and d: decoding performance; panel e: training loss
C_AXES = (126.2, 394.4, 46.55, 64.2)
D_AXES = (240.6, 394.9, 46.6, 64.2)
E_AXES = (380.5, 400.3, 56.6, 55.5)
CDE_CANVAS = (0.0, 340.0, 513.0, 163.0)


def plot_biases_grid(df_counts, sorters, sorter_titles, experiments, layers, bias_order, bias_colors, bias_labels):
    """Panel b: ratios of 'single-unit' spike assignment biases by layer (x-axis), spike sorter (columns) and ground truth simulation (rows)"""
    fig = new_figure(B_CANVAS)
    axes = np.empty((len(experiments), len(sorters)), dtype=object)
    for r, exp in enumerate(experiments):
        for c, sorter in enumerate(sorters):
            ax = axes[r, c] = add_axes_pt(fig, B_LEFT[c], B_TOP[r], B_W, B_H)
            d = df_counts[(df_counts["sorter"] == sorter) & (df_counts["experiment"] == exp)]
            for x, layer in enumerate(layers):
                dl = d[d["layer"] == layer].set_index("quality")["count"]
                if dl.sum() == 0:
                    continue
                bottom = 0
                for quality in bias_order:
                    count = dl.get(quality, 0)
                    pct = count / dl.sum() * 100
                    if pct == 0:
                        continue
                    ax.bar(x, pct, bottom=bottom, width=0.75, color=bias_colors[quality], edgecolor="k", linewidth=0.3)
                    if pct > 5:        # only counts of segments above 5% are annotated
                        ax.text(x, bottom + pct / 2, str(int(count)), ha="center", va="center", fontsize=FONT, fontweight="bold" if "good" in quality else None)
                    bottom += pct
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["bottom"].set_visible(r == len(experiments) - 1)
            ax.spines["left"].set_visible(c == 0)
            ax.spines["bottom"].set_position(("axes", -0.05))
            ax.spines["left"].set_position(("axes", -0.05))
            ax.set_ylim([0, 100])
            ax.set_xlim([-0.6, len(layers) - 0.4])
            if c == 0:
                ax.set_yticks([0, 50, 100], [0, 0.5, 1])
            else:
                ax.set_yticks([])
            if r == len(experiments) - 1:
                ax.set_xticks(range(len(layers)), layers)
            else:
                ax.set_xticks([])
            if r == 0:
                ax.set_title(sorter_titles[c], fontsize=TITLE, fontweight="bold", pad=4.5)

    # labels and legend
    for r, exp in enumerate(experiments):
        text_pt(fig, 17.0, B_TOP[r] + B_H / 2, exp)
    text_pt(fig, 5.5, (B_TOP[0] + B_TOP[-1] + B_H) / 2, "Experiment", rotation=90)
    text_pt(fig, 31.0, (B_TOP[0] + B_TOP[-1] + B_H) / 2, "Single-unit ratio", rotation=90)
    text_pt(fig, 257.0, 138.0, "Predicted spike assignment biases among 'single-units'", fontsize=TITLE, fontweight="bold")
    lg = B_LEGEND
    text_pt(fig, lg["title_xy"][0], lg["title_xy"][1], "Spike assignment biases", fontsize=TITLE, fontweight="bold")
    for top, q in zip(lg["tops"], ["oversplitter", "mixed: good + oversplitter", "mixed: overmerger + oversplitter", "mixed: good + overmerger + oversplitter", "false positive"]):
        x0, y0, w, h = fig._origin
        fig.add_artist(Rectangle(((lg["patch_x"] - x0) / w, 1 - (top - y0 + 4.2) / h), 12.0 / w, 4.2 / h, transform=fig.transFigure, facecolor=bias_colors[q], edgecolor="k", linewidth=0.3))
        text_pt(fig, lg["text_x"], top + 2.1, bias_labels[q], ha="left")
    set_text_sizes(fig, titles=list(sorter_titles) + ["Predicted spike assignment biases among 'single-units'", "Spike assignment biases"])
    return fig, axes
