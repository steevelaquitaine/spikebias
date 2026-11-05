"""Panel of Fig. 7b: manifold information capacity of the ground truth and of the sorted populations, drawn from the source data

author: laquitainesteeve@gmail.com

The panel has the geometry of panel b of the paper's figure (page 22 of paper/Laquitaine_NatureNeuro_2025.pdf): canvas 160.5 x 134.5 pt (bounding
box of the panel with its title, panel letter excluded). Every element is drawn in the pt coordinates of that page (y down): the y axis (153.2 pt per
unit of capacity / lower bound, 1.0 at y = 371.71 pt), the eight notched box plots (11.76 pt apart, 5.76 pt wide, filled black), the red dashed
mean of the ground truth and the blue dashed chance level, and the labels. Text is Arial 5 pt and the title 7 pt (bold), editable.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

FONT, TITLE = 5, 7
REGION = (49.63, 271.7, 210.1, 406.16)
POPULATIONS = ["Ground\ntruth", "KS4", "KS3", "KS2.5", "KS2", "KS", "HS", "TC"]
Y1, PPU = 371.71, 153.2                 # y (pt) of a capacity of 1.0, pt per unit of capacity
X0, DX, BOX_W = 95.72, 11.76, 5.76      # x of the first box, spacing and width
AXIS = dict(spine_x=78.96, y=(318.07, 373.24), x_axis_y=376.0, x=(83.96, 183.9), tick=1.49)
YTICKS = [1.3, 1.2, 1.1, 1.0]
LINES = (89.84, 183.9)
RED, BLUE = (1.0, 0.0, 0.0), (0.0, 0.698, 1.0)
BLUE_TEXT = (0.0, 0.498, 0.8235)
# centres (pt) of the tick labels (30 deg) of the populations: 'Ground' and 'truth' are two lines
XLABELS = {"Ground\ntruth": [("Ground", 94.14, 387.5), ("truth", 97.14, 393.66)], "KS4": [("KS4", 107.0, 385.76)], "KS3": [("KS3", 118.8, 385.86)],
           "KS2.5": [("KS2.5", 130.55, 386.92)], "KS2": [("KS2", 142.51, 385.81)], "KS": [("KS", 154.05, 385.03)], "HS": [("HS", 165.82, 385.1)],
           "TC": [("TC", 177.42, 384.75)]}


def _ctext(ax, cx, cy, s, size=FONT, bold=False, rotation=0, color="k"):
    """Text whose ink bounding box (after the rotation, in degrees counter-clockwise) is centred on (cx, cy)"""
    path = TextPath((0, 0), s, size=size, prop=FontProperties(family="Arial", weight="bold" if bold else "normal"))
    ext = path.transformed(Affine2D().rotate_deg(rotation)).get_extents()
    ax.text(cx - (ext.x0 + ext.x1) / 2, cy + (ext.y0 + ext.y1) / 2, s, ha="left", va="baseline", rotation=rotation, rotation_mode="anchor",
            fontsize=size, fontweight="bold" if bold else "normal", color=color)


def _line(ax, xs, ys, **kw):
    ax.plot(xs, ys, color=kw.pop("color", "k"), linewidth=kw.pop("lw", 0.5), solid_capstyle="butt", **kw)


def plot_information_capacity(df):
    """Fig. 7b from the source data (columns population, bootstrap, capacity_over_lower_bound)"""
    x0, y0, x1, y1 = REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    to_y = lambda v: Y1 - (np.asarray(v) - 1.0) * PPU

    gt_mean = df[df["population"] == POPULATIONS[0]]["capacity_over_lower_bound"].mean()    # the paper's dashed line is the mean of the 200 bootstraps
    for y, color in [(to_y(gt_mean), RED), (Y1, BLUE)]:
        ax.plot(LINES, [y, y], color=color, linewidth=1.0, linestyle=(0, (3.7, 1.6)), zorder=1)
    values = [to_y(df[df["population"] == p]["capacity_over_lower_bound"].values) for p in POPULATIONS]
    xs = X0 + DX * np.arange(len(POPULATIONS))
    bp = ax.boxplot(values, positions=xs, widths=BOX_W, notch=True, patch_artist=True, manage_ticks=False, zorder=3,
               boxprops=dict(facecolor="k", edgecolor="k", linewidth=0.5), medianprops=dict(color="k", linewidth=0.5),
               whiskerprops=dict(color="k", linewidth=0.5), capprops=dict(color="k", linewidth=0.5), capwidths=BOX_W / 2,
               flierprops=dict(marker="o", markerfacecolor="k", markeredgecolor="k", markersize=1.5, markeredgewidth=0.5))

    # the axis is limited to capacities from 0.99 to 1.35: what is beyond is clipped
    clip = Rectangle((x0, 317.5), x1 - x0, 373.3 - 317.5, transform=ax.transData)
    for artists in bp.values():
        for a in artists:
            a.set_clip_path(clip)

    # axes
    _line(ax, [AXIS["spine_x"]] * 2, AXIS["y"])
    for v in YTICKS:
        _line(ax, [AXIS["spine_x"] - AXIS["tick"], AXIS["spine_x"]], [to_y(v)] * 2, lw=0.3)
        _ctext(ax, 70.03, float(to_y(v)), f"{v:.1f}")
    _line(ax, AXIS["x"], [AXIS["x_axis_y"]] * 2)
    for x in xs:
        _line(ax, [x, x], [AXIS["x_axis_y"], AXIS["x_axis_y"] + AXIS["tick"]], lw=0.3)
    for p in POPULATIONS:
        for s, cx, cy in XLABELS[p]:
            _ctext(ax, cx, cy, s, rotation=30)

    # labels and annotations
    _ctext(ax, 52.45, 346.5, "Manifold capacity / lower bound", rotation=90)
    _ctext(ax, 58.9, 345.6, "(ratio)", rotation=90)
    _ctext(ax, 133.98, 403.4, "Ground truth and sorted unit populations")
    _ctext(ax, 129.05, 294.9, "200 bootstraps")
    _ctext(ax, 150.28, 317.48, "Ground truth", color=RED)
    _ctext(ax, 151.32, 325.64, "mean capacity", color=RED)
    _ctext(ax, 99.55, 365.83, "Chance", color=BLUE_TEXT)
    # title on two lines (at 7 pt the paper's single line, 148 pt wide at 6 pt, would be wider than the panel)
    _ctext(ax, 135.95, 274.5, "Spike sorting reduces stimulus", size=TITLE, bold=True)
    _ctext(ax, 135.95, 282.9, "information capacity", size=TITLE, bold=True)
    return fig


# Fig. 7f: capacity of the ground truth under the undersampling and the selection bias of each sorter (page 22): canvas 116.8 x 120.5 pt
SAMPLING_REGION = (421.31, 288.01, 538.09, 408.54)
SAMPLING = dict(spine_x=450.64, y=(318.76, 373.93), x_axis_y=376.69, x=(454.8, 538.09), tick=1.49, y1=372.88, ppu=105.35, x_gt=456.33, tick_x0=458.48, tick_dx=12.245,
                offset=2.145, box_w=3.0, lines=(452.35, 538.09), yticks=[1.4, 1.2, 1.0])
SAMPLING_COLORS = {"undersampling": (0.8745, 0.1255, 0.1255), "selection bias": (1.0, 1.0, 1.0)}
GREY = (0.298, 0.298, 0.298)
SAMPLING_SORTERS = ["KS4", "KS3", "KS2.5", "KS2", "KS", "HS"]


def plot_sampling_capacity(df):
    """Fig. 7f from the source data (columns population, sampling_scheme, bootstrap, capacity_over_lower_bound)"""
    g = SAMPLING
    x0, y0, x1, y1 = SAMPLING_REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    to_y = lambda v: g["y1"] - (np.asarray(v) - 1.0) * g["ppu"]
    cap = "capacity_over_lower_bound"

    gt = df[df["population"] == "ground truth"][cap].values
    for y, color in [(to_y(np.mean(gt)), RED), (g["y1"], BLUE)]:
        ax.plot(g["lines"], [y, y], color=color, linewidth=1.0, linestyle=(0, (3.7, 1.6)), zorder=1)

    style = dict(notch=True, patch_artist=True, widths=g["box_w"], manage_ticks=False, zorder=3, capwidths=g["box_w"] / 2,
                 boxprops=dict(edgecolor=GREY, linewidth=0.5), whiskerprops=dict(color=GREY, linewidth=0.5), capprops=dict(color=GREY, linewidth=0.5),
                 flierprops=dict(marker="o", markerfacecolor="k", markeredgecolor=GREY, markersize=1.5, markeredgewidth=0.5))
    clip = Rectangle((x0, 317.5), x1 - x0, 373.95 - 317.5, transform=ax.transData)
    boxes = [([to_y(gt)], g["x_gt"], "k", "k")]
    for i, sorter in enumerate(SAMPLING_SORTERS):
        for scheme, sign in [("undersampling", -1), ("selection bias", 1)]:
            c = df[(df["population"] == sorter) & (df["sampling_scheme"] == scheme)][cap].values
            boxes.append(([to_y(c)], g["tick_x0"] + g["tick_dx"] * (i + 1) + sign * g["offset"], SAMPLING_COLORS[scheme], GREY))
    for vals, x, face, median in boxes:
        bp = ax.boxplot(vals, positions=[x], medianprops=dict(color=median, linewidth=0.5), **style)
        bp["boxes"][0].set_facecolor(face)
        for artists in bp.values():
            for a in artists:
                a.set_clip_path(clip)

    # axes
    _line(ax, [g["spine_x"]] * 2, g["y"])
    for v in g["yticks"]:
        _line(ax, [g["spine_x"] - g["tick"], g["spine_x"]], [to_y(v)] * 2, lw=0.3)
        _ctext(ax, 441.7, float(to_y(v)), f"{v:.1f}")
    _line(ax, g["x"], [g["x_axis_y"]] * 2)
    labels = ["ground\ntruth"] + SAMPLING_SORTERS
    for i, lab in enumerate(labels):
        xt = g["tick_x0"] + g["tick_dx"] * i
        _line(ax, [xt, xt], [g["x_axis_y"], g["x_axis_y"] + g["tick"]], lw=0.3)
        if i == 0:
            _ctext(ax, xt - 1.58, g["x_axis_y"] + 11.5, "ground", rotation=30)
            _ctext(ax, xt + 1.42, g["x_axis_y"] + 17.66, "truth", rotation=30)
        else:       # the top right corner of the rotated label is 3.6 pt right of the tick and 5.6 pt below the axis
            path = TextPath((0, 0), lab, size=FONT, prop=FontProperties(family="Arial")).transformed(Affine2D().rotate_deg(30)).get_extents()
            ax.text(xt + g.get("label_dx", 3.6) - path.x1, g["x_axis_y"] + 5.6 + path.y1, lab, ha="left", va="baseline", rotation=30, rotation_mode="anchor", fontsize=FONT)

    # labels and legend
    _ctext(ax, 423.5, 347.17, "Manifold capacity / lower bound", rotation=90)
    _ctext(ax, 430.7, 346.4, "(ratio)", rotation=90)
    _ctext(ax, 494.82, 398.63, "Sampling scheme")
    _ctext(ax, 494.76, 405.78, "applied to ground truth")
    _ctext(ax, 493.8, 290.75, "200 bootstraps")
    for (scheme, ys, label, cy) in [("undersampling", 297.32, "undersampling", 299.99), ("selection bias", 305.83, "selection bias", 307.91)]:
        ax.add_patch(Rectangle((465.63, ys), 11.94, 4.18, facecolor=SAMPLING_COLORS[scheme], edgecolor=GREY, linewidth=0.5))
        _ctext(ax, 501.92 if scheme == "undersampling" else 500.41, cy, label)
    return fig


# Fig. 7g: capacity of the ground truth, of the 'good + merged + partial' units and of the degraded units of each sorter (page 22): canvas 124.6 x 117.0 pt
QUALITY_REGION = (61.45, 456.52, 186.04, 573.53)
QUALITY = dict(spine_x=88.2, y=(481.56, 536.73), x_axis_y=539.49, x=(92.36, 175.64), tick=1.49, y1=514.65, ppu=47.34, x_gt=95.52, tick_x0=98.31, tick_dx=11.9,
               white_dx=2.78, box_w=1.95, red_line=(92.36, 181.59), blue_line=(92.36, 175.64), yticks=[1.5, 1.0])
QUALITY_CLASSES = [("good + merged + partial", SAMPLING_COLORS["undersampling"], "k"), ("degraded (merged + partial)", (1.0, 1.0, 1.0), GREY)]


def _ltext(ax, x, cy, s, size=FONT):
    """Text whose ink starts at x and whose ink is centred vertically on cy"""
    ext = TextPath((0, 0), s, size=size, prop=FontProperties(family="Arial")).get_extents()
    ax.text(x - ext.x0, cy + (ext.y0 + ext.y1) / 2, s, ha="left", va="baseline", fontsize=size)


def plot_quality_capacity(df):
    """Fig. 7g from the source data (columns population, unit_class, bootstrap, capacity_over_lower_bound)"""
    g = QUALITY
    x0, y0, x1, y1 = QUALITY_REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    to_y = lambda v: g["y1"] - (np.asarray(v) - 1.0) * g["ppu"]
    cap = "capacity_over_lower_bound"

    gt = df[df["population"] == "ground truth"][cap].values
    ax.plot(g["red_line"], [to_y(np.mean(gt))] * 2, color=RED, linewidth=1.0, linestyle=(0, (3.7, 1.6)), zorder=1)
    ax.plot(g["blue_line"], [g["y1"]] * 2, color=BLUE, linewidth=0.5, linestyle=(0, (1.85, 0.8)), zorder=1)

    style = dict(notch=True, patch_artist=True, widths=g["box_w"], manage_ticks=False, zorder=3, capwidths=g["box_w"] / 2,
                 boxprops=dict(edgecolor=GREY, linewidth=0.5), whiskerprops=dict(color=GREY, linewidth=0.5), capprops=dict(color=GREY, linewidth=0.5),
                 flierprops=dict(marker="o", markerfacecolor="w", markeredgecolor=GREY, markersize=1.5, markeredgewidth=0.5))
    clip = Rectangle((x0, 481.4), x1 - x0, 536.8 - 481.4, transform=ax.transData)
    boxes = [(to_y(gt), g["x_gt"], "k", "k")]
    for i, sorter in enumerate(SAMPLING_SORTERS):
        for j, (unit_class, face, median) in enumerate(QUALITY_CLASSES):
            c = df[(df["population"] == sorter) & (df["unit_class"] == unit_class)][cap].values
            boxes.append((to_y(c), g["tick_x0"] + g["tick_dx"] * (i + 1) + j * g["white_dx"], face, median))
    for vals, x, face, median in boxes:
        bp = ax.boxplot([vals], positions=[x], medianprops=dict(color=median, linewidth=0.5), **style)
        bp["boxes"][0].set_facecolor(face)
        for artists in bp.values():
            for a in artists:
                a.set_clip_path(clip)

    # axes
    _line(ax, [g["spine_x"]] * 2, g["y"])
    for v in g["yticks"]:
        _line(ax, [g["spine_x"] - 1.5, g["spine_x"]], [to_y(v)] * 2, lw=0.3)
        _ctext(ax, 79.27, float(to_y(v)), f"{v:.1f}")
    _line(ax, g["x"], [g["x_axis_y"]] * 2)
    labels = ["ground\ntruth"] + SAMPLING_SORTERS
    for i, lab in enumerate(labels):
        xt = g["tick_x0"] + g["tick_dx"] * i
        _line(ax, [xt, xt], [g["x_axis_y"], g["x_axis_y"] + g["tick"]], lw=0.3)
        if i == 0:
            _ctext(ax, xt - 1.58, g["x_axis_y"] + 11.5, "ground", rotation=30)
            _ctext(ax, xt + 1.42, g["x_axis_y"] + 17.66, "truth", rotation=30)
        else:
            path = TextPath((0, 0), lab, size=FONT, prop=FontProperties(family="Arial")).transformed(Affine2D().rotate_deg(30)).get_extents()
            ax.text(xt + g.get("label_dx", 3.6) - path.x1, g["x_axis_y"] + 5.6 + path.y1, lab, ha="left", va="baseline", rotation=30, rotation_mode="anchor", fontsize=FONT)

    # labels and legend
    _ctext(ax, 63.66, 507.55, "Manifold capacity / lower bound", rotation=90)
    _ctext(ax, 70.5, 507.55, "(ratio)", rotation=90)
    _ctext(ax, 134.18, 570.77, "Spike sorter")
    _ctext(ax, 134.76, 459.26, "200 bootstraps")
    for (face, ys, label) in [(RED, 469.64, "good + merged + partial units"), ((1, 1, 1), 476.18, "degraded (merged + partial units)")]:
        ax.add_patch(Rectangle((90.3, ys), 5.79, 4.32, facecolor=face, edgecolor=GREY, linewidth=0.354))
        _ltext(ax, 97.77, ys + 2.16, label)
    return fig


# Fig. 7h, 7i: manifold radius and dimension of the ground truth and of the sorted populations (page 22): black notched box plots, red dashed mean of the
# ground truth. Canvas 110.25 x 92.57 pt (7h) and 108.58 x 113.97 pt (7i, with the title "200 bootstraps")
RADIUS = dict(
    region=(195.52, 480.21, 305.77, 572.78), column="manifold_radius", spine_x=218.31, y=(480.21, 535.38), x_axis_y=538.14, x=(222.48, 305.77), tick=1.49,
    y0=500.01, v0=0.8, ppu=232.9, x_box=[227.68, 238.09, 248.5, 258.91, 269.32, 279.74, 290.15, 300.56], tick_x=[226.19, 238.09, 248.5, 258.91, 269.32, 279.74, 290.15, 300.56],
    box_w=5.1, red_line=(222.48, 305.77), yticks=[(0.8, "0.8", 209.19), (0.7, "0.7", 209.18)], ylabel=("Manifold radius", 197.73, 507.69),
    gt_label=((218.64, 549.64), (221.64, 555.8)), xlabel=("Spike sorter", 264.3, 570.0), title=None)
DIMENSION = dict(
    region=(310.88, 458.86, 419.46, 572.83), column="manifold_dimension", spine_x=332.01, y=(480.27, 535.43), x_axis_y=538.19, x=(336.17, 419.46), tick=1.49,
    y0=487.41, v0=14.0, ppu=7.13, x_box=[341.38 + 10.41 * i for i in range(8)], tick_x=[341.38 + 10.41 * i for i in range(8)], box_w=5.1, red_line=(336.17, 419.46),
    yticks=[(14, "14", 323.9), (12, "12", 323.88), (10, "10", 323.9), (8, "8", 325.36)], ylabel=("Manifold dimension", 313.1, 507.83),
    gt_label=((333.83, 549.69), (336.83, 555.85)), xlabel=("Spike sorter", 378.0, 570.07), title=("200 bootstraps", 375.03, 461.6))
CORRELATION = dict(
    region=(425.61, 479.01, 539.16, 571.58), column="center_correlation", spine_x=451.71, y=(479.01, 534.18), x_axis_y=536.93, x=(455.88, 539.16), tick=1.49,
    y0=517.17, v0=0.14, ppu=1449.0, x_box=[461.08 + 10.41 * i for i in range(8)], tick_x=[461.08 + 10.41 * i for i in range(8)], box_w=5.1, red_line=(455.88, 539.16),
    yticks=[(0.16, "0.16", 440.91), (0.15, "0.15", 440.93), (0.14, "0.14", 440.9), (0.13, "0.13", 440.91)], ylabel=("Center correlation", 427.8, 506.65),
    gt_label=((454.98, 548.43), (457.98, 554.59)), xlabel=("Spike sorter", 497.7, 568.8), title=None, label_dx=5.05)


def _population_panel(df, g):
    x0, y0, x1, y1 = g["region"]
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    to_y = lambda v: g["y0"] - (np.asarray(v) - g["v0"]) * g["ppu"]
    col = g["column"]

    ax.plot(g["red_line"], [to_y(df[df["population"] == POPULATIONS[0]][col].mean())] * 2, color=RED, linewidth=1.0, linestyle=(0, (3.7, 1.6)), zorder=1)
    values = [to_y(df[df["population"] == p][col].values) for p in POPULATIONS]
    bp = ax.boxplot(values, positions=g["x_box"], widths=g["box_w"], notch=True, patch_artist=True, manage_ticks=False, zorder=3, capwidths=g["box_w"] / 2,
                    boxprops=dict(facecolor="k", edgecolor="k", linewidth=0.5), medianprops=dict(color="k", linewidth=0.5),
                    whiskerprops=dict(color="k", linewidth=0.5), capprops=dict(color="k", linewidth=0.5),
                    flierprops=dict(marker="o", markerfacecolor="k", markeredgecolor="k", markersize=1.5, markeredgewidth=0.5))
    clip = Rectangle((x0, g["y"][0] - 0.3), x1 - x0, g["y"][1] - g["y"][0] + 0.6, transform=ax.transData)
    for artists in bp.values():
        for a in artists:
            a.set_clip_path(clip)

    # axes
    _line(ax, [g["spine_x"]] * 2, g["y"])
    for v, label, cx in g["yticks"]:
        _line(ax, [g["spine_x"] - g["tick"], g["spine_x"]], [to_y(v)] * 2, lw=0.3)
        _ctext(ax, cx, float(to_y(v)), label)
    _line(ax, g["x"], [g["x_axis_y"]] * 2)
    for xt, pop in zip(g["tick_x"], POPULATIONS):
        _line(ax, [xt, xt], [g["x_axis_y"], g["x_axis_y"] + g["tick"]], lw=0.3)
        if pop.startswith("Ground"):
            _ctext(ax, *g["gt_label"][0], "Ground", rotation=30)
            _ctext(ax, *g["gt_label"][1], "truth", rotation=30)
        else:       # the top right corner of the rotated label is 3.6 pt right of the tick and 5.6 pt below the axis
            path = TextPath((0, 0), pop, size=FONT, prop=FontProperties(family="Arial")).transformed(Affine2D().rotate_deg(30)).get_extents()
            ax.text(xt + g.get("label_dx", 3.6) - path.x1, g["x_axis_y"] + 5.6 + path.y1, pop, ha="left", va="baseline", rotation=30, rotation_mode="anchor", fontsize=FONT)
    _ctext(ax, g["ylabel"][1], g["ylabel"][2], g["ylabel"][0], rotation=90)
    _ctext(ax, g["xlabel"][1], g["xlabel"][2], g["xlabel"][0])
    if g["title"]:
        _ctext(ax, g["title"][1], g["title"][2], g["title"][0])
    return fig


def plot_radius(df):
    """Fig. 7h from the source data (columns population, bootstrap, manifold_radius)"""
    return _population_panel(df, RADIUS)


def plot_dimension(df):
    """Fig. 7i from the source data (columns population, bootstrap, manifold_dimension)"""
    return _population_panel(df, DIMENSION)


def plot_correlation(df):
    """Fig. 7j from the source data (columns population, bootstrap, center_correlation)"""
    return _population_panel(df, CORRELATION)
