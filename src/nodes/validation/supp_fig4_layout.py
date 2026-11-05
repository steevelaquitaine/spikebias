"""Layout of the panels of Supplementary Fig. 4 (firing rates of the single-units sorted by each spike sorter, by layer)

author: laquitainesteeve@gmail.com

The axes have the width and horizontal positions of the axes of paper/figures/supp_fig4.pdf (pt of the page, y down, measured on the figure); their height and the vertical spacing are reduced (the whole figure is 184 mm high instead of 214 mm, Nature: 185 mm at most for a figure with a legend of less than 300 words). Detached axes
are drawn by the notebooks (spines moved by 5% of the axes size). Text is Arial 5 pt, except the titles (7 pt, bold).
Panel a: 5 sorters (columns) x 6 rows (the whole column and the five layers), axes of 55.78 x 30.5 pt, 36.5 pt between rows.
Panels b, c: 5 sorters, axes of 33.47 x 150 pt (b) and 27.89 x 150 pt (c).
"""

from matplotlib import pyplot as plt
from matplotlib.text import Text

FONT, TITLE = 5, 7

# panel a: left of the axes of each sorter, y of the top of the first row of each sorter, size and pitch of the rows
A_LEFT = [34.57, 135.23, 237.88, 335.40, 430.06]
A_TOP = [50.0, 47.8, 47.9, 49.4, 48.8]
A_W, A_H, A_PITCH = 55.78, 30.5, 36.5
A_CANVAS = (0.0, 0.0, 504.0, 300.0)                      # x0, y0, width and height (pt) of the figure in the page

# panels b and c
B_LEFT = [37.24, 84.39, 128.02, 172.44, 216.62]
B_TOP = [409.25, 409.45, 409.28, 409.04, 408.18]
B_W, B_H = 33.47, 150.0
B_CANVAS = (0.0, 372.0, 262.0, 242.0)
C_LEFT = [303.89, 347.02, 387.55, 429.26, 471.45]
C_TOP = [405.18, 404.54, 403.96, 403.53, 403.16]
C_W, C_H = 27.89, 150.0
C_CANVAS = (262.0, 372.0, 242.0, 242.0)


def new_figure(canvas):
    """Figure with the size (pt) of the canvas; fig._origin keeps the position of the canvas in the page"""
    x0, y0, w, h = canvas
    fig = plt.figure(figsize=(w / 72, h / 72))
    fig._origin = canvas
    return fig


def add_axes_pt(fig, left, top, width, height):
    """Axes at (left, top) of the page, y down, in pt"""
    x0, y0, w, h = fig._origin
    return fig.add_axes([(left - x0) / w, 1 - (top - y0 + height) / h, width / w, height / h])


def text_pt(fig, x, y, s, **kw):
    """Text centred at (x, y) of the page, y down, in pt"""
    x0, y0, w, h = fig._origin
    kw = {"ha": "center", "va": "center", **kw}
    return fig.text((x - x0) / w, 1 - (y - y0) / h, s, **kw)


def axes_a(fig):
    """[sorter][row] axes of panel a"""
    return [[add_axes_pt(fig, A_LEFT[c], A_TOP[c] + r * A_PITCH, A_W, A_H) for r in range(6)] for c in range(5)]


def axes_bc(fig, left, top, width, height):
    return [add_axes_pt(fig, left[c], top[c], width, height) for c in range(5)]


def set_text_sizes(fig, titles=()):
    """5 pt for all the text, 7 pt (bold) for the titles: the texts in `titles` and the bold texts of 7 pt or more"""
    fig.canvas.draw()
    for ax in fig.axes:
        ax.tick_params(which="both", labelsize=FONT)
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if not t.get_text().strip():
            continue
        is_title = t.get_text() in titles or (t.get_fontweight() in ("bold", 700) and t.get_fontsize() >= 6.9)
        t.set_fontsize(TITLE if is_title else FONT)
        if is_title:
            t.set_fontweight("bold")
    fig.canvas.draw()
