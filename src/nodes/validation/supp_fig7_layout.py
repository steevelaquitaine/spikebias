"""Layout of the panels of Supplementary Fig. 7 (firing rates of the sorted single-units and of the ground truth neurons by layer), drawn from the source data

author: laquitainesteeve@gmail.com

The axes have the position and size measured on paper/figures/supp_fig7.pdf (pt of the page, y down; page 540 x 396 pt, the artwork is about 168 x 139 mm,
within the 180 mm width of Nature Communications). a, b: violins (5 layers 43.2 pt apart), c: ratio of the counts, d, e: mean ratios (5 layers 36.8 pt apart).
Text is Arial 5 pt, except the titles (7 pt, bold).
"""

import seaborn as sns

from src.nodes.validation.supp_fig4_layout import add_axes_pt, new_figure, set_text_sizes, text_pt

A_CANVAS, A_AXES, A_YLIM = (0.0, 0.0, 300.0, 175.0), (60.6, 17.2, 216.0, 145.9), (-2.97, 1.3)
B_CANVAS, B_AXES, B_YLIM = (0.0, 205.0, 300.0, 170.0), (60.6, 217.2, 216.0, 143.6), (-3.11, 1.35)
C_CANVAS, C_AXES, C_YLIM, C_XLIM = (300.0, 0.0, 240.0, 140.0), (340.8, 14.3, 147.8, 104.8), (0.0, 3.0), (-3.0, 0.75)
D_CANVAS, D_AXES, D_YLIM, D_XLIM = (300.0, 150.0, 240.0, 115.0), (336.0, 160.0, 162.1, 82.0), (0.0, 2.5), (-0.2, 4.2)
E_CANVAS, E_AXES, E_YLIM, E_XLIM = (300.0, 265.0, 240.0, 115.0), (336.0, 281.6, 162.1, 79.2), (-1.545, 0.0), (-0.2, 4.2)


def move_violin_legend(ax):
    """Legend of the violins (experiment: GT, NS) at the bottom left of the axes"""
    if ax.get_legend() is not None:
        sns.move_legend(ax, "lower left", bbox_to_anchor=(0.0, 0.02), frameon=True)
