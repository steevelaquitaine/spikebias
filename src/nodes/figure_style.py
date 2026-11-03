"""Nature figure guidelines applied to every saved figure

author: laquitainesteeve@gmail.com

Guide to preparing final artwork (Nature research journals, https://www.nature.com/documents/NRJs-guide-to-preparing-final-artwork.pdf):
sans-serif text, preferably Helvetica or Arial, from 5 to 7 pt; graphs as vector files with editable text (AI, EPS, PDF); images of at least
300 dpi; RGB colours.

`apply()` wraps `matplotlib.figure.Figure.savefig` so that, whatever the notebook's own settings, every saved figure:
* uses Arial for all its text (and regular Arial for math text),
* has its text sizes clipped to 5-7 pt,
* keeps its text as text (svg: `svg.fonttype = "none"`; pdf and eps: TrueType fonts, `fonttype = 42`),
* rasterizes its images at no less than 300 dpi,
* is also saved as a .pdf next to each .svg.

Usage, at the top of a notebook (after moving to the repository):

    from src.nodes import figure_style
    figure_style.apply()
"""

import os

import matplotlib
from matplotlib.figure import Figure
from matplotlib.text import Text

FONT = "Arial"
MIN_SIZE, MAX_SIZE = 5.0, 7.0       # pt
MIN_DPI = 300

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.default": "regular",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
}

_original_savefig = Figure.savefig


def _clip(size: float) -> float:
    return min(max(size, MIN_SIZE), MAX_SIZE)


def format_text(fig: Figure):
    """Arial and 5-7 pt for every text of the figure, including the tick labels"""
    fig.canvas.draw()                                        # create the tick labels
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = axis.get_ticklabels(which="both")
            if labels:
                axis.set_tick_params(which="both", labelsize=_clip(labels[0].get_fontsize()), labelfontfamily="sans-serif")
    for text in fig.findobj(Text):
        if text.get_text():
            text.set_fontfamily("sans-serif")
            text.set_fontsize(_clip(text.get_fontsize()))
            weight = text.get_fontweight()
            if isinstance(weight, (int, float)):                   # numeric weights (e.g. 1000) cannot be written as svg text
                text.set_fontweight("bold" if weight >= 600 else "normal")


def savefig(self, fname, *args, **kwargs):
    """Figure.savefig with the guidelines; a .svg is also saved as .pdf"""
    with matplotlib.rc_context(RC):
        format_text(self)
        dpi = kwargs.get("dpi", matplotlib.rcParams["savefig.dpi"])
        if not isinstance(dpi, (int, float)) or dpi < MIN_DPI:
            kwargs["dpi"] = MIN_DPI
        _original_savefig(self, fname, *args, **kwargs)
        if isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".svg"):
            kwargs.pop("format", None)
            _original_savefig(self, str(fname)[:-4] + ".pdf", *args, **kwargs)


def apply():
    """Apply the guidelines to all the figures saved from now on"""
    matplotlib.rcParams.update(RC)
    Figure.savefig = savefig
