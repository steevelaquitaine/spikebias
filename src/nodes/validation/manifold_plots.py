"""Panel of Fig. 7a: neural manifolds of the ground truth and of the sorted populations, drawn from the source data

author: laquitainesteeve@gmail.com

The panel has the geometry of panel a of the paper's figure (page 22 of paper/Laquitaine_NatureNeuro_2025.pdf): canvas 488.5 x 160.5 pt (the
bounding box of the panel on the page, panel letter excluded). Every element is placed at its position on that page (pt, y down): the eight 2D
manifolds (top row, points averaged over the trials of each direction) and the eight 3D manifolds (bottom row, one point per trial), each fitted in
the rectangle of the paper's image, the latent axes, the titles and the colour bar of the stimulus direction (359 to 0 deg). Text is Arial 5 pt and
the figure title 7 pt (bold), editable. The 3D manifolds are rendered as images (300 dpi or more), as in the paper.
"""

import io

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

FONT, TITLE = 5, 7
REGION = (49.4, 87.9, 537.93, 248.4)
POPULATIONS = ["Ground truth", "KS4", "KS3", "KS2.5", "KS2", "KS", "HS", "TC"]
# rectangles (x0, y0, x1, y1) of the paper's images, 2D (top row) and 3D (bottom row) manifolds
RECTS_2D = [(61.68, 115.56, 107.19, 157.93), (123.49, 115.56, 165.49, 154.68), (181.15, 120.32, 229.8, 158.94), (242.96, 119.32, 288.6, 154.55),
            (310.74, 116.19, 359.76, 158.94), (376.93, 118.32, 418.68, 153.68), (432.21, 119.2, 477.97, 156.06), (496.23, 116.06, 536.73, 159.95)]
RECTS_3D = [(55.63, 173.31, 108.82, 214.33), (119.87, 177.49, 180.95, 213.85), (186.26, 173.19, 233.73, 216.72), (242.56, 171.94, 299.88, 215.11),
            (308.07, 175.04, 372.55, 215.52), (384.79, 178.27, 416.67, 214.45), (431.66, 178.09, 477.15, 221.07), (495.13, 174.5, 537.93, 226.45)]
# origins of the latent axes: 2D (right and up arrows) and 3D (up, up-right and down-right arrows)
ORIGINS_2D = [(59.8, 164.54), (115.73, 164.77), (170.96, 164.02), (234.39, 163.28), (301.56, 163.28), (369.48, 164.02), (423.21, 164.77), (485.16, 163.28)]
ORIGINS_3D = [(56.22, 215.38), (116.33, 217.42), (172.3, 215.93), (237.98, 215.18), (303.65, 214.44), (370.82, 215.18), (421.57, 215.18), (485.0, 214.44)]
# centres of the titles of the populations (one or two lines)
TITLES = {
    "Ground truth": [("Ground truth", 81.45, 102.6)],
    "KS4": [("Kilosort 4", 140.9, 102.6), ("(KS4)", 141.75, 111.9)],
    "KS3": [("Kilosort 3", 201.4, 103.5), ("(KS3)", 202.25, 112.75)],
    "KS2.5": [("Kilosort 2.5", 264.3, 103.2), ("(KS2.5)", 264.25, 112.5)],
    "KS2": [("Kilosort 2", 333.0, 102.8), ("(KS2)", 333.8, 112.1)],
    "KS": [("Kilosort", 396.7, 102.9), ("(KS)", 397.4, 112.2)],
    "HS": [("Herdingspikes", 452.05, 103.0), ("(HS)", 452.85, 111.7)],
    "TC": [("Multiunit threshold", 512.7, 102.05), ("crossing (TC)", 513.6, 111.3)],
}
BAR = dict(x=(275.97, 336.66), y=(246.0, 248.37), ticks=[276.0, 291.1, 306.3, 321.5, 336.7], labels=["359", "270", "180", "90", "0"])
MARKER = 5.2          # marker area (pt^2), diameter 2.3 pt
DPI_3D = 600


def _ctext(ax, cx, cy, s, size=FONT, bold=False, rotation=0):
    """Text whose bounding box is centred on (cx, cy)"""
    if rotation:
        ax.text(cx, cy, s, ha="center", va="center", rotation=rotation, fontsize=size, fontweight="bold" if bold else "normal")
        return
    ext = TextPath((0, 0), s, size=size, prop=FontProperties(family="Arial", weight="bold" if bold else "normal")).get_extents()
    ax.text(cx - (ext.x0 + ext.x1) / 2, cy + (ext.y0 + ext.y1) / 2, s, ha="left", va="baseline", fontsize=size, fontweight="bold" if bold else "normal")


def _arrow(ax, p0, p1, head, lw):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="k", linewidth=lw, solid_capstyle="butt")
    ax.fill([h[0] for h in head], [h[1] for h in head], color="k", linewidth=0.3)


def _latent_axes_2d(ax, o):
    x, y = o
    _arrow(ax, (x, y - 0.27), (x + 0.05, y - 5.35), [(x + 0.1, y - 6.8), (x + 1.31, y - 4.61), (x - 1.15, y - 4.64)], 1.0)
    _arrow(ax, (x, y), (x + 7.11, y - 0.05), [(x + 8.47, y - 0.06), (x + 6.44, y + 1.16), (x + 6.42, y - 1.26)], 0.97)


def _latent_axes_3d(ax, o):
    x, y = o
    _arrow(ax, (x - 0.1, y - 0.34), (x - 0.05, y - 5.43), [(x - 0.04, y - 6.87), (x + 1.17, y - 4.69), (x - 1.29, y - 4.72)], 1.0)
    _arrow(ax, (x + 0.49, y - 0.21), (x + 7.98, y - 3.32), [(x + 9.3, y - 3.87), (x + 7.8, y - 1.89), (x + 6.85, y - 4.2)], 1.0)
    _arrow(ax, (x, y), (x + 8.15, y + 3.46), [(x + 9.45, y + 4.02), (x + 7.03, y + 4.35), (x + 7.96, y + 2.02)], 1.0)


def _render_3d(d3, width, height):
    """3D manifold of the trials, rendered at DPI_3D with the extent of the point cloud fitted in a (width x height) pt rectangle: (rgba, w, h in pt)"""
    xyz = d3[["latent_1", "latent_2", "latent_3"]].values
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    zoom = 1.0
    for _ in range(4):
        fig = plt.figure(figsize=(2.5 * width / 72, 2.5 * height / 72))
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.view_init(20, 45, 0)
        ax.scatter(*xyz.T, c=d3["direction_deg"], cmap="hsv", edgecolors="w", linewidths=0.1, s=MARKER)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_axis_off(); ax.set_box_aspect((1, 1, 0.75), zoom=zoom)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI_3D, transparent=True)
        plt.close(fig)
        buf.seek(0)
        img = plt.imread(buf)
        ys, xs = np.where(img[..., 3] > 0.02)
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        w, h = img.shape[1] / DPI_3D * 72, img.shape[0] / DPI_3D * 72
        zoom *= min(width / w, height / h)
    return img, w, h


def plot_manifolds(df_3d, df_2d):
    """Fig. 7a from the source data: df_3d (population, trial, direction_deg, latent_1-3) and df_2d (population, direction_deg, latent_1, latent_2)"""
    x0, y0, x1, y1 = REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")

    r = np.sqrt(MARKER) / 2                     # marker radius (pt)
    for j, pop in enumerate(POPULATIONS):
        # 2D manifold: the point cloud fitted (equal scales) in the rectangle of the paper's image
        d2 = df_2d[df_2d["population"] == pop]
        rx0, ry0, rx1, ry1 = RECTS_2D[j]
        xs, ys = d2["latent_1"].values, d2["latent_2"].values
        scale = min((rx1 - rx0 - 2 * r) / np.ptp(xs), (ry1 - ry0 - 2 * r) / np.ptp(ys))
        cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
        px = cx + (xs - (xs.min() + xs.max()) / 2) * scale
        py = cy - (ys - (ys.min() + ys.max()) / 2) * scale
        ax.scatter(px, py, c=d2["direction_deg"], cmap="hsv", edgecolors="w", linewidths=0.2, s=MARKER, rasterized=True, zorder=2)
        _latent_axes_2d(ax, ORIGINS_2D[j])

        # 3D manifold: rendered image fitted in the rectangle of the paper's image
        d3 = df_3d[df_3d["population"] == pop]
        bx0, by0, bx1, by1 = RECTS_3D[j]
        img, w, h = _render_3d(d3, bx1 - bx0, by1 - by0)
        bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
        ax.imshow(img, extent=(bcx - w / 2, bcx + w / 2, bcy + h / 2, bcy - h / 2), aspect="auto", interpolation="antialiased", zorder=2)
        _latent_axes_3d(ax, ORIGINS_3D[j])

        for s, cx_, cy_ in TITLES[pop]:
            _ctext(ax, cx_, cy_, s)

    # latent axes labels (first column)
    _ctext(ax, 54.9, 152.0, "Latent 2", rotation=90)
    _ctext(ax, 71.3, 168.65, "Latent 1")
    _ctext(ax, 51.6, 207.1, "Latent 3", rotation=90)
    _ctext(ax, 77.3, 213.55, "Latent 1")
    _ctext(ax, 65.3, 222.95, "Latent 2")

    # colour bar of the stimulus direction (359 to 0 deg, left to right)
    bx0, bx1 = BAR["x"]
    by0, by1 = BAR["y"]
    ax.imshow(np.linspace(1, 0, 256)[None, :], cmap="hsv", vmin=0, vmax=1, extent=(bx0, bx1, by1, by0), aspect="auto", zorder=2)
    ax.plot([276.0, 336.7, 336.7, 276.0, 276.0], [245.1, 245.1, 248.4, 248.4, 245.1], color="k", linewidth=0.22)
    for xt, lab in zip(BAR["ticks"], BAR["labels"]):
        ax.plot([xt, xt], [245.1, 246.0], color="k", linewidth=0.22)
    for xt, lab in zip([276.5, 291.35, 306.6, 321.65, 336.0], BAR["labels"]):
        _ctext(ax, xt, 240.85, lab)
    _ctext(ax, 306.1, 234.4, "Stimulus direction (deg)")
    _ctext(ax, 293.0, 90.7, "Spike sorting distorts the neural manifolds evoked by 360 stimulus directions", size=TITLE, bold=True)
    return fig


# Fig. 7c: 3D manifold of the ground truth (page 22): canvas 48.8 x 67.8 pt (panel letter excluded, the title shared by panels c-f is not part of it)
GT_PANEL = dict(region=(211.06, 307.87, 259.82, 375.7), rect=(211.1, 334.5, 259.82, 372.12), origin=(212.35, 371.25), titles=[("Ground truth", 234.36, 310.19)])
# Fig. 7d: ground truth subjected to the undersampling of Kilosort 4: canvas 52.5 x 76.9 pt
UNDERSAMPLED_PANEL = dict(region=(273.07, 298.8, 325.56, 375.7), rect=(280.6, 333.43, 325.56, 373.91), origin=(274.37, 371.34),
                          titles=[("Ground truth", 299.64, 301.12), ("subjected to KS4", 299.75, 309.22), ("undersampling", 300.77, 316.76)])


def plot_single_manifold(df_3d, panel=GT_PANEL):
    """One 3D manifold of the trials (columns direction_deg, latent_1-3), title and latent axes at the paper's positions"""
    x0, y0, x1, y1 = panel["region"]
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    bx0, by0, bx1, by1 = panel["rect"]
    img, w, h = _render_3d(df_3d, bx1 - bx0, by1 - by0)
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    ax.imshow(img, extent=(bcx - w / 2, bcx + w / 2, bcy + h / 2, bcy - h / 2), aspect="auto", interpolation="antialiased", zorder=2)
    _latent_axes_3d(ax, panel["origin"])
    for s, cx, cy in panel["titles"]:
        _ctext(ax, cx, cy, s)
    return fig


# Fig. 7e: ground truth subjected to the undersampling and the selection biases of Kilosort 4: canvas 59.6 x 75.8 pt
SELECTED_PANEL = dict(region=(341.58, 299.29, 401.22, 375.1), rect=(349.68, 333.97, 391.59, 371.76), origin=(342.87, 370.7),
                      titles=[("Ground truth", 372.89, 301.6), ("subjected to KS4", 374.05, 309.7), ("undersampling and", 372.97, 317.24),
                              ("selection bias", 374.07, 324.2)])
