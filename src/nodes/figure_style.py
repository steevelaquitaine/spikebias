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
* is also saved as a .pdf next to each .svg,
* writes the svg text with explicit font attributes (font-family="Arial", font-size, font-weight) instead of the CSS shorthand
  "font: 5px 'Arial', ..." that Adobe Illustrator does not read (see fix_svg_fonts).

Usage, at the top of a notebook (after moving to the repository):

    from src.nodes import figure_style
    figure_style.apply()
"""

import os
import re

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


_FONT_STYLE = re.compile(r'style="font:\s*((?:(?:italic|oblique|bold|normal|\d00)\s+)*)([\d.]+)px\s+([^;"]+?)\s*(;[^"]*)?"')


def _font_attributes(match):
    flags, size, rest = match.group(1).split(), match.group(2), match.group(4) or ""
    attrs = [f'font-family="{FONT}"', f'font-size="{size}"']
    if "bold" in flags or any(f.isdigit() and int(f) >= 600 for f in flags):
        attrs.append('font-weight="bold"')
    if "italic" in flags or "oblique" in flags:
        attrs.append('font-style="italic"')
    for decl in rest.strip("; ").split(";"):                # keep the other declarations (e.g. text-anchor) as attributes
        if ":" in decl:
            key, value = (x.strip() for x in decl.split(":", 1))
            attrs.append(f'{key}="{value}"')
    return " ".join(attrs)


def fix_svg_fonts(path):
    """Replace the CSS font shorthand of matplotlib's svg text by explicit attributes, which Adobe Illustrator reads"""
    with open(path, encoding="utf-8") as f:
        svg = f.read()
    fixed, n = _FONT_STYLE.subn(_font_attributes, svg)
    if n:
        with open(path, "w", encoding="utf-8") as f:
            f.write(fixed)
    return n


def savefig(self, fname, *args, **kwargs):
    """Figure.savefig with the guidelines; a .svg is also saved as .pdf"""
    with matplotlib.rc_context(RC):
        format_text(self)
        dpi = kwargs.get("dpi", matplotlib.rcParams["savefig.dpi"])
        if not isinstance(dpi, (int, float)) or dpi < MIN_DPI:
            kwargs["dpi"] = MIN_DPI
        _original_savefig(self, fname, *args, **kwargs)
        if isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".svg"):
            fix_svg_fonts(fname)
            kwargs.pop("format", None)
            _original_savefig(self, str(fname)[:-4] + ".pdf", *args, **kwargs)


def apply():
    """Apply the guidelines to all the figures saved from now on"""
    matplotlib.rcParams.update(RC)
    Figure.savefig = savefig
