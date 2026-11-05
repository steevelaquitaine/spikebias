"""Layout of the panels of Supplementary Fig. 6 (delta time, features of the MEArec models, yield and accuracy), drawn from the source data

author: laquitainesteeve@gmail.com

The axes have the horizontal position and width of the axes of paper/figures/supp_fig6.pdf (pt of the page, y down, measured on the figure; page
508 x 599 pt). The vertical spacing of panel c (13 rows) is reduced and panels d and e are moved up accordingly, so that the whole figure is 184 mm high
instead of 211 mm (Nature: 185 mm at most for a figure with a legend of less than 300 words). Text is Arial 5 pt, except the titles (7 pt, bold).
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from src.nodes.validation.supp_fig4_layout import add_axes_pt, new_figure, set_text_sizes, text_pt

FONT, TITLE = 5, 7

A_AXES = (30.0, 15.3, 54.4, 58.3)
A_CANVAS = (0.0, 0.0, 140.0, 95.0)
B_LEFT = [111.0, 168.6, 226.1, 283.7, 343.4, 401.0, 458.6]
B_TOP = [27.9] * 4 + [26.2] * 3
B_W, B_H = 46.5, 36.6
B_CANVAS = (60.0, 0.0, 460.0, 100.0)
C_LEFT = [82.2, 161.8, 245.2, 328.5, 411.9]
C_TOP0, C_PITCH, C_W, C_H = 126.6, 22.0, 60.6, 12.0
C_CANVAS = (0.0, 90.0, 508.0, 340.0)
D_AXES = (120.4, 443.0, 120.0, 49.3)
E_AXES = (327.0, 438.0, 125.0, 53.7)
DE_CANVAS = (60.0, 414.0, 450.0, 130.0)


def plain_log_labels(ax):
    """Labels of a log y axis written as plain numbers (1, 100, 1000): a mathtext exponent would be smaller than 5 pt"""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_features_grid(features, features_ns, plots):
    """Panel c: distributions of the features of each synthetic model (blue) and of the biophysical model (red); 13 rows x 5 models"""
    models = list(plots.MODEL_TITLES)
    fig = new_figure(C_CANVAS)
    n_rows = len(plots.FEATURE_ROWS)
    axes = np.empty((n_rows, len(models)), dtype=object)
    for r in range(n_rows):
        for c in range(len(models)):
            axes[r, c] = add_axes_pt(fig, C_LEFT[c], C_TOP0 + r * C_PITCH, C_W, C_H)
    for col, model in enumerate(models):
        plots.plot_feature_distributions(axes, col, features[model], features_ns, plots.MODEL_TITLES[model], n_matched=plots.N_MATCHED[model])
    for ax in axes.ravel():
        ax.tick_params(labelsize=FONT, length=2.5, pad=1)
        plain_log_labels(ax)
    for col, model in enumerate(models):
        axes[0, col].set_title(plots.MODEL_TITLES[model], fontsize=TITLE, fontweight="bold", pad=5.5)
    fig.canvas.draw()
    for col, model in enumerate(models):
        for row in range(plots.N_MATCHED[model]):
            plots.annotate_matched(axes[row, col], fontsize=FONT)
    for row, (_, _, label, _, _, _) in enumerate(plots.FEATURE_ROWS):
        text_pt(fig, 60.0, C_TOP0 + row * C_PITCH + C_H / 2, label, ha="right", fontsize=FONT, fontweight="bold")
    text_pt(fig, 6.0, C_TOP0 + (n_rows - 1) * C_PITCH / 2 + C_H / 2, "Unit count", rotation=90)
    text_pt(fig, 290.0, 100.0, "Increasingly detailed MEAREC models vs our model", fontsize=TITLE, fontweight="bold")
    set_text_sizes(fig, titles=list(plots.MODEL_TITLES.values()) + ["Increasingly detailed MEAREC models vs our model"])
    return fig, axes
