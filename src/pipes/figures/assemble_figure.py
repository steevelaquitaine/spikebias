"""Assemble the panels of a figure into one vector pdf with the layout of the paper

author: laquitainesteeve@gmail.com

The layout (position of each panel, panel letter and title) is measured on the figure of the paper (paper/Laquitaine_NatureNeuro_2025.pdf, in
pt of the paper's page) and scaled so that the panels keep about their native size (their text stays at 5-7 pt). Each panel pdf of
figures/8_source_data/<panel>/ is fitted in its box (same scale in x and y, centred); panels made of several files (e.g. one per layer) are
stacked in their box. Panel letters (Arial bold 8 pt) and titles (Arial bold 7 pt) are drawn as editable text.

Output: paper/figures/<figure>.pdf and a report of the scale of each panel (a scale far from 1 changes the text size).

Usage:

    PYTHONPATH=$PWD:<pypdf> envs/spikebias/bin/python src/pipes/figures/assemble_figure.py fig2
"""

import argparse
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

SOURCE = "figures/8_source_data"
OUTPUT = "paper/figures"               # the figures of the manuscript

# Fig. 2: boxes [x0, y0, x1, y1] in pt of the paper's page (y downwards), letters (x, y of the letter's top-left) and titles (centre x, top y)
LAYOUTS = {
    "fig2": {
        "page": [91.0, 72.0, 506.0, 656.0],
        "panels": {
            "a": ["fig2a/fig2a.pdf"], "b": ["fig2b/fig2b.pdf"], "c": ["fig2c/fig2c.pdf"], "d": ["fig2d/fig2d.pdf"],
            "e": ["fig2e/fig2e.pdf"], "f": ["fig2f/fig2f.pdf"], "g": ["fig2g/fig2g.pdf"], "h": ["fig2h/fig2h.pdf"],
            "i": ["fig2i/fig2i.pdf"], "j": ["fig2j/fig2j.pdf"],
            "k": ["fig2k/fig2k_depth_1.pdf", "fig2k/fig2k_depth_2.pdf", "fig2k/fig2k_depth_3.pdf"],
            "l": ["fig2l/fig2l_depth_1.pdf", "fig2l/fig2l_depth_2.pdf", "fig2l/fig2l_depth_3.pdf"],
            "m": ["fig2m/fig2m.pdf"], "n": ["fig2n/fig2n.pdf"], "o": ["fig2o/fig2o.pdf"], "p": ["fig2p/fig2p.pdf"],
            "q": ["fig2q/fig2q_l1.pdf", "fig2q/fig2q_l23.pdf", "fig2q/fig2q_l4.pdf", "fig2q/fig2q_l5.pdf", "fig2q/fig2q_l6.pdf"],
            "r": ["fig2r/fig2r.pdf"], "s": ["fig2s/fig2s.pdf"], "t": ["fig2t/fig2t.pdf"], "u": ["fig2u/fig2u.pdf"],
        },
        "boxes": {
            "a": [92.4, 83.0, 170.0, 150.0], "b": [170.0, 83.0, 238.0, 150.0], "c": [238.0, 83.0, 308.0, 150.0],
            "d": [308.0, 83.0, 377.0, 150.0], "e": [377.0, 83.0, 441.0, 150.0], "f": [441.0, 83.0, 504.0, 150.0],
            "g": [91.2, 155.0, 172.0, 320.0], "h": [172.0, 155.0, 240.0, 320.0], "i": [240.0, 155.0, 305.0, 320.0],
            "j": [305.0, 155.0, 372.0, 320.0], "k": [372.0, 155.0, 438.0, 320.0], "l": [438.0, 155.0, 504.0, 320.0],
            "m": [92.0, 336.0, 252.0, 412.0], "n": [92.0, 418.0, 252.0, 490.0], "o": [92.0, 502.0, 252.0, 573.0],
            "p": [92.0, 576.0, 252.0, 655.0], "q": [255.0, 336.0, 335.0, 655.0],
            "r": [335.0, 336.0, 504.0, 415.0], "s": [335.0, 418.0, 504.0, 495.0], "t": [335.0, 502.0, 504.0, 575.0],
            "u": [335.0, 578.0, 504.0, 655.0],
        },
        "letters": {
            "a": (94.7, 82.0), "b": (172.0, 82.0), "c": (240.0, 82.0), "d": (310.0, 82.0), "e": (379.0, 82.0), "f": (443.6, 82.0),
            "g": (97.0, 152.5), "h": (174.0, 152.5), "i": (240.0, 152.5), "j": (305.0, 152.5), "k": (373.0, 152.5), "l": (440.0, 152.5),
            "m": (96.5, 334.0), "n": (97.0, 417.0), "o": (97.0, 491.5), "p": (97.0, 570.0), "q": (267.0, 334.0),
            "r": (355.0, 334.0), "s": (357.0, 417.0), "t": (356.0, 493.0), "u": (357.0, 570.0),
        },
        "titles": [
            ("Neuropixels 1.0 probe (NPX)", 240.5, 73.6), ("Dense probe", 433.5, 73.6),
            ("NPX", 181.6, 327.5), ("NPX", 433.4, 325.0), ("Dense probe", 182.9, 494.0), ("Dense probe", 435.6, 495.0),
        ],
    },
}


def text_overlay(width, height, letters, titles, to_page):
    """A page with the panel letters and titles as editable Arial text"""
    with matplotlib.rc_context({"pdf.fonttype": 42, "font.family": "Arial"}):
        fig = plt.figure(figsize=(width / 72, height / 72))
        for letter, (x, y) in letters.items():
            px, py = to_page(x, y)
            fig.text(px / width, 1 - py / height, letter, fontsize=8, fontweight="bold", va="top", ha="left")
        for title, x, y in titles:
            px, py = to_page(x, y)
            fig.text(px / width, 1 - py / height, title, fontsize=7, fontweight="bold", va="top", ha="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", transparent=True)
        plt.close(fig)
    return PdfReader(buf).pages[0]


def assemble(name, scale):
    lay = LAYOUTS[name]
    x0p, y0p, x1p, y1p = lay["page"]
    W, H = (x1p - x0p) * scale, (y1p - y0p) * scale

    def to_page(x, y):
        return (x - x0p) * scale, (y - y0p) * scale

    page = PageObject.create_blank_page(width=W, height=H)
    report = []
    for letter, files in lay["panels"].items():
        bx0, by0, bx1, by1 = lay["boxes"][letter]
        (px0, py0), (px1, py1) = to_page(bx0, by0), to_page(bx1, by1)
        bw, bh = px1 - px0, py1 - py0
        parts = [PdfReader(os.path.join(SOURCE, f)).pages[0] for f in files]
        sizes = [(float(p.mediabox.width), float(p.mediabox.height)) for p in parts]
        natural_w = max(w for w, h in sizes)
        natural_h = sum(h for w, h in sizes)                       # parts are stacked from top to bottom
        s = min(bw / natural_w, bh / natural_h)
        y_top = H - py0 - (bh - natural_h * s) / 2                # centred in the box (pdf y upwards)
        x_left = px0 + (bw - natural_w * s) / 2
        for part, (w, h) in zip(parts, sizes):
            y_top -= h * s
            page.merge_transformed_page(part, Transformation().scale(s, s).translate(x_left + (natural_w - w) * s / 2, y_top))
        report.append((letter, round(s, 3), round(natural_w / 72 * 25.4, 1), round(natural_h / 72 * 25.4, 1), round(bw / 72 * 25.4, 1), round(bh / 72 * 25.4, 1)))
    page.merge_page(text_overlay(W, H, lay["letters"], lay["titles"], to_page))
    out = PdfWriter()
    out.add_page(page)
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, f"{name}.pdf")
    with open(path, "wb") as f:
        out.write(f)
    return path, (W / 72 * 25.4, H / 72 * 25.4), report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=list(LAYOUTS))
    parser.add_argument("--scale", type=float, default=None, help="scale of the paper's layout (default: the median native size of the panels)")
    args = parser.parse_args()
    lay = LAYOUTS[args.figure]
    if args.scale is None:
        ratios = []
        for letter, files in lay["panels"].items():
            bx0, by0, bx1, by1 = lay["boxes"][letter]
            sizes = [(float(p.mediabox.width), float(p.mediabox.height)) for p in (PdfReader(os.path.join(SOURCE, f)).pages[0] for f in files)]
            ratios.append(min(max(w for w, h in sizes) / (bx1 - bx0), sum(h for w, h in sizes) / (by1 - by0)))
        args.scale = sorted(ratios)[len(ratios) // 2]
    path, size, report = assemble(args.figure, args.scale)
    print(f"{path}: {size[0]:.1f} x {size[1]:.1f} mm (layout scale {args.scale:.3f})")
    print("panel  scale  native WxH (mm)  box WxH (mm)")
    for r in report:
        print(f"{r[0]:>5}  {r[1]:5.2f}  {r[2]:5.1f} x {r[3]:5.1f}   {r[4]:5.1f} x {r[5]:5.1f}")
