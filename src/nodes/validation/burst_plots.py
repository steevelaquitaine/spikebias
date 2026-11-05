"""Panels c to f of Supplementary Fig. 3 drawn from the source data and rescaled to the height the journal guidelines allow

author: laquitainesteeve@gmail.com

c: mean waveform (and 95% CI) of the first three spikes of bursts of three bursting units; d, e, f: box plots (notched) over units of the trough amplitude
(d bursting units, e non-bursting units) and of the trough width (f, bursting units) of the first and second spike of a burst, for ranges of inter-spike intervals.
The arrangement is that of paper/figures/supp_fig3.pdf (positions measured on the vector figure) with every length multiplied by SCALE (spike_shape_plots) so that the
whole figure is 185 mm high; text is Arial 5 pt (titles 7 pt), editable.
"""

import numpy as np
from matplotlib import cbook

from src.nodes.validation.spike_shape_plots import FONT, TITLE, SCALE, _Canvas, _ink

BLACK, RED = (0, 0, 0), (0.8745, 0.1255, 0.1255)
SPIKES = {"1st spike": ((0, 0, 0), (0.5020, 0.5020, 0.5020)), "2nd spike": ((1.0, 0, 0), (1.0, 0.7529, 0.7961)), "3rd spike": ((1.0, 0.6471, 0), (1.0, 0.6471, 0))}

# Fig. S3c: three bursting units; per plot: unit, x (pt) of -1.25 ms, pt per ms, y (pt) = a + b * voltage (uV), spine x, y extent of the spine, x axis (x0, x1, y), y ticks (uV),
# centre of the title and of the y label
C_PLOTS = [
    dict(unit=2281009, x0=41.2, a=497.257, b=-3.4879, sx=37.1, ys=(476.4, 535.5), ax=(37.1, 126.4, 535.5), yticks=[4, 2, 0, -2, -4, -6, -8, -10], title=(82.2, 471.9), ylabel=(19.6, 506.0)),
    dict(unit=2873667, x0=216.0, a=495.851, b=-1.2347, sx=212.0, ys=(476.7, 535.8), ax=(212.0, 301.3, 535.8), yticks=[10, 0, -10, -20, -30], title=(259.4, 473.4), ylabel=(194.5, 506.35)),
    dict(unit=506676, x0=392.8, a=490.771, b=-12.7189, sx=388.7, ys=(475.4, 534.5), ax=(388.7, 478.0, 534.5), yticks=[1, 0, -1, -2, -3], title=(435.25, 470.3), ylabel=(373.0, 506.5)),
]
PPM_C = 33.1
C_LEGEND = [("1st spike", 484.4, "line"), ("95% CI", 491.9, "patch"), ("2nd spike", 499.4, "line"), ("95% CI", 507.0, "patch"), ("3rd spike", 514.5, "line"), ("95% CI", 522.05, "patch")]
C_REGION = (11.0, 446.0, 482.0, 556.0)


def plot_bursts(df):
    """Supplementary Fig. 3c from the source data (columns unit, mtype, etype, n_bursts, spike, time_ms, voltage_mean_uV, voltage_ci95_uV)"""
    cv = _Canvas(C_REGION)
    for pl in C_PLOTS:
        Yp = lambda v: pl["a"] + pl["b"] * np.asarray(v)
        Xp = lambda t: pl["x0"] + (np.asarray(t) + 1.25) * PPM_C
        for spike, (color, band) in SPIKES.items():
            g = df[(df["unit"] == pl["unit"]) & (df["spike"] == spike)].sort_values("time_ms")
            m, c, t = g["voltage_mean_uV"].values, g["voltage_ci95_uV"].values, g["time_ms"].values
            X, Yu = cv.p(Xp(t), Yp(m + c))
            _, Yl = cv.p(Xp(t), Yp(m - c))
            cv.ax.fill_between(X, Yu, Yl, facecolor=band, edgecolor=band, alpha=0.3, linewidth=0.53, zorder=2)
            cv.line(Xp(t), Yp(m), lw=1.0, color=color, zorder=3)
        sx, (yt, yb), (xa0, xa1, ya) = pl["sx"], pl["ys"], pl["ax"]
        cv.line([sx, sx], [yt, yb], lw=0.27)
        cv.line([xa0, xa1], [ya, ya], lw=0.27)
        for v in pl["yticks"]:
            y = float(Yp(v))
            cv.line([sx - 2.0, sx], [y, y], lw=0.27)
            cv.rtext(sx - 3.2, y + 1.7, str(v).replace("-", "−"))
        for tv, label in ((-1.25, "−1.25"), (0.0, "0"), (1.25, "1.25")):
            x = float(Xp(tv))
            cv.line([x, x], [ya, ya + 2.0], lw=0.27)
            cv.text(x, ya + 6.4, label)
        cv.text(pl["title"][0], pl["title"][1] - 2.5, f"Bursting unit {pl['unit']}", size=TITLE)
        cv.text(pl["ylabel"][0], pl["ylabel"][1], "Voltage (µV)", rotation=90)
        cv.text((xa0 + xa1) / 2, ya + 13.4, "Time (ms)")
    # legend between the first two plots
    handles = {"1st spike": SPIKES["1st spike"], "2nd spike": SPIKES["2nd spike"], "3rd spike": SPIKES["3rd spike"]}
    order = ["1st spike", "1st spike", "2nd spike", "2nd spike", "3rd spike", "3rd spike"]
    for (label, y, kind), key in zip(C_LEGEND, order):
        color, band = handles[key]
        if kind == "line":
            cv.line([142.2, 151.6], [y, y], lw=1.5, color=color)
        else:
            X, Y = cv.p([142.2, 151.6], [y - 1.6, y + 1.6])
            from matplotlib.patches import Rectangle
            cv.ax.add_patch(Rectangle((X[0], Y[0]), X[1] - X[0], Y[1] - Y[0], facecolor=band, edgecolor=band, alpha=0.3, linewidth=0.5))
        X, Y = cv.p(154.7, y)
        e = _ink(label)
        cv.ax.text(float(X) - e.x0, float(Y) + (e.y0 + e.y1) / 2, label, ha="left", va="baseline", fontsize=FONT)
    cv.text(256.45, 455.7, "Modulation of bursting units' spike amplitudes and widths", size=TITLE, bold=True)
    return cv.fig


# Fig. S3d-f: box plots by ISI range. Per panel: groups (one axes each) with the x of the spine, y extent of the spine, bottom axis (x0, x1, y), x of the two boxes, centre of the
# range label, of the significance mark; y calibration (pt of the value `v0` and pt per unit), y ticks, geometry of the labels
def _groups(spines, y, axis_y, axis_w0, axis_w1, boxes, label_xy, sig_xy):
    return [dict(sx=s, ys=y, ax=(s + axis_w0, s + axis_w1, axis_y), boxes=b, label=l, sig=g) for s, b, l, g in zip(spines, boxes, label_xy, sig_xy)]


D_PANEL = dict(
    column="trough_amplitude_uV", ranges=["0-21", "21-61", "61-101", "101-141", "141-181"], stats="statistics",
    groups=_groups([24.9, 49.2, 73.4, 101.0, 128.5], (585.1, 660.8), 664.6, 0.9, 18.2, [(28.6, 39.9), (52.9, 64.1), (77.1, 88.4), (104.7, 115.9), (132.2, 143.5)],
                   [(33.15, 568.8), (57.35, 568.8), (83.1, 568.8), (110.85, 568.8), (136.95, 568.8)], [(34.4, 575.8), (59.7, 576.35), (82.95, 576.35), (111.85, 576.0), (138.85, 575.45)]),
    box_w=5.6, v0=0.0, y0=585.64, ppu=-1.002, yticks=[0, -10, -20, -30, -40, -50, -60, -70], decimals=0, label_right=21.2, title=("Bursting units' ISI (ms)", 85.5, 560.6),
    ylabel=("Trough amplitudes (µV)", 5.3, 623.0), xlabel=(87.1, 679.95), tick_label_dy=6.5, legend=(133.3, 139.2, 638.1, 644.2, 141.4, 142.0),
    region=(2.5, 550.5, 176.0, 691.0), all_ns=None)
E_PANEL = dict(
    column="trough_amplitude_uV", ranges=["21-61", "61-101", "101-141", "141-181"], stats="statistics",
    groups=_groups([194.7, 230.6, 266.5, 302.4], (588.2, 663.9), 667.6, 1.3, 27.0, [(200.2, 216.9), (236.1, 252.8), (272.0, 288.7), (307.9, 324.6)],
                   [(208.75, 571.0), (242.05, 571.0), (278.8, 571.0), (315.55, 571.0)], [None] * 4),
    box_w=6.7, v0=0.0, y0=589.1, ppu=-1.742, yticks=[0, -5, -10, -15, -20, -25, -30, -35, -40], decimals=0, label_right=189.6, title=("Non-bursting units' ISI (ms)", 261.95, 563.4),
    ylabel=None, xlabel=(264.65, 681.05), tick_label_dy=6.55, legend=None, region=(180.0, 552.5, 331.0, 691.0), all_ns=(260.25, 579.8))
F_PANEL = dict(
    column="trough_width_ms", ranges=["0-21", "21-61", "61-101", "101-141", "141-181"], stats="statistics",
    groups=_groups([379.2, 403.4, 427.7, 455.2, 482.8], (586.6, 662.2), 666.0, 0.8, 18.1, [(384.2, 392.9), (408.5, 417.1), (432.7, 441.4), (460.3, 468.9), (487.8, 496.5)],
                   [(388.0, 568.65), (412.2, 568.65), (438.0, 568.65), (464.25, 568.65), (493.25, 568.65)], [None] * 5),
    box_w=3.5, v0=0.55, y0=590.0, ppu=-229.4, yticks=[0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25], decimals=2, label_right=372.8, title=("Bursting units' ISI (ms)", 438.2, 559.55),
    ylabel=("Trough widths (ms)", 354.0, 624.4), xlabel=(441.7, 680.8), tick_label_dy=6.5, legend=None, region=(351.2, 550.5, 505.5, 691.0), all_ns=(437.75, 579.05))


def _box_panel(df, cfg):
    cv = _Canvas(cfg["region"])
    Y = lambda v: cfg["y0"] + (np.asarray(v) - cfg["v0"]) * cfg["ppu"]
    stats_df = df[1] if isinstance(df, tuple) else None
    amp = df[0] if isinstance(df, tuple) else df
    sig = dict(zip(stats_df["isi_range_ms"], stats_df["significance"])) if stats_df is not None else {}
    for gi, (rng, grp) in enumerate(zip(cfg["ranges"], cfg["groups"])):
        sx, (yt, yb), (xa0, xa1, ya) = grp["sx"], grp["ys"], grp["ax"]
        cv.line([sx, sx], [yt, yb], lw=0.5)
        cv.line([xa0, xa1], [ya, ya], lw=0.5)
        for spike, xc, color in zip(("1st", "2nd"), grp["boxes"], (BLACK, RED)):
            v = amp[(amp["isi_range_ms"] == rng) & (amp["spike_rank"] == spike)][cfg["column"]].values
            st = cbook.boxplot_stats(Y(v), bootstrap=None)[0]
            _bxp(cv, st, xc, cfg["box_w"], color, clip=(float(cv.p(0, yt)[1]), float(cv.p(0, yb)[1])))
            cv.line([xc, xc], [ya, ya + 2.8], lw=0.5)
            cv.text(xc + (-1.3 if spike == "1st" else 1.8), ya + cfg["tick_label_dy"], spike)
        cv.text(*grp["label"], rng)
        mark = sig.get(rng)
        if grp["sig"] is not None and mark is not None:
            cv.text(*grp["sig"], str(mark))
    # y axis ticks and labels (first group)
    sx = cfg["groups"][0]["sx"]
    for v in cfg["yticks"]:
        y = float(Y(v))
        cv.line([sx - 2.8, sx], [y, y], lw=0.5)
        s = f"{v:.{cfg['decimals']}f}".replace("-", "−")
        cv.rtext(cfg["label_right"], y + 1.7, s)
    cv.text(cfg["title"][1], cfg["title"][2] - 2.5, cfg["title"][0], size=TITLE, bold=True)
    if cfg["ylabel"]:
        cv.text(cfg["ylabel"][1], cfg["ylabel"][2], cfg["ylabel"][0], rotation=90)
    cv.text(cfg["xlabel"][0], cfg["xlabel"][1] + 3.2, "Spike order within burst")
    if cfg["all_ns"]:
        cv.text(*cfg["all_ns"], "All ns")
    if cfg["legend"]:
        x0, x1, y1, y2, tx, _ = cfg["legend"]
        cv.line([x0, x1], [y1, y1], lw=2.48, color=BLACK)
        cv.line([x0, x1], [y2, y2], lw=2.48, color=RED)
        for label, y in (("1st spike", y1), ("2nd spike", y2)):
            X, Yy = cv.p(tx + 0.6, y)
            e = _ink(label)
            cv.ax.text(float(X) - e.x0, float(Yy) + (e.y0 + e.y1) / 2, label, ha="left", va="baseline", fontsize=FONT)
    return cv.fig


def _bxp(cv, st, xc, width, color, clip=None):
    """Notched box plot at x = xc (pt of the paper) from the statistics of the values in pt of the paper (y down)"""
    X0 = float(cv.p(xc, 0)[0])
    w = width * SCALE
    notch_w = 0.5
    # coordinates in canvas pt
    qy = lambda v: float(cv.p(0, v)[1])
    q1, q3, med, lo, hi = qy(st["q1"]), qy(st["q3"]), qy(st["med"]), qy(st["cilo"]), qy(st["cihi"])
    ys, xs = [], []
    hw, nw = w / 2, w / 4
    pts = [(X0 - hw, q1), (X0 - hw, lo), (X0 - nw, med), (X0 - hw, hi), (X0 - hw, q3), (X0 + hw, q3), (X0 + hw, hi), (X0 + nw, med), (X0 + hw, lo), (X0 + hw, q1)]
    from matplotlib.patches import Polygon
    patch = Polygon(pts, closed=True, facecolor=color, edgecolor=color, linewidth=0.5, zorder=3)
    cv.ax.add_patch(patch)
    artists = [patch]
    artists += cv.ax.plot([X0 - nw, X0 + nw], [med, med], color=(1, 1, 1) if color == BLACK else BLACK, linewidth=0.5, zorder=4)
    wl, wh = qy(st["whislo"]), qy(st["whishi"])
    artists += cv.ax.plot([X0, X0], [q3, wh], color=color, linewidth=0.5, zorder=2, solid_capstyle="butt")
    artists += cv.ax.plot([X0, X0], [q1, wl], color=color, linewidth=0.5, zorder=2, solid_capstyle="butt")
    for yy in (wl, wh):
        artists += cv.ax.plot([X0 - w / 4, X0 + w / 4], [yy, yy], color=color, linewidth=0.5, zorder=2, solid_capstyle="butt")
    fl = [qy(f) for f in st["fliers"]]
    artists += cv.ax.plot([X0] * len(fl), fl, linestyle="None", marker="o", markersize=1.5, markerfacecolor="k", markeredgecolor="k", markeredgewidth=0.3, zorder=4)
    if clip is not None:       # the axis ends where the spine ends: what is beyond is clipped
        from matplotlib.patches import Rectangle
        r = Rectangle((X0 - w, clip[0]), 2 * w, clip[1] - clip[0], transform=cv.ax.transData)
        for a in artists:
            a.set_clip_path(r)


def plot_trough_amplitudes_bursting(sheets):
    """Supplementary Fig. 3d: `sheets` = (trough_amplitudes, statistics) dataframes of supp_fig3d.xlsx"""
    return _box_panel(sheets, D_PANEL)


def plot_trough_amplitudes_non_bursting(sheets):
    """Supplementary Fig. 3e"""
    return _box_panel(sheets, E_PANEL)


def plot_trough_widths_bursting(sheets):
    """Supplementary Fig. 3f"""
    return _box_panel(sheets, F_PANEL)
