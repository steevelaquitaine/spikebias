"""Panel of Supplementary Fig. 2a: power spectral density of the raw traces of the in vivo and simulated experiments, drawn from the source data

author: laquitainesteeve@gmail.com

The panel has the geometry of panel a of paper/figures/supp_fig2.pdf (sizes in pt, y down): canvas 120.1 x 191.1 pt. The axes are drawn in the
page coordinates of the figure: frequency on a log scale (17.13 pt per decade, 1 Hz at x = 82.05 pt) and normalized power on a log scale (8.21 pt per
decade, 10^5 at y = 55.09 pt), detached axes, ticks of the paper (major ticks at 10^1 and 10^3 Hz, minor ticks at 0.2, 0.4, 0.6 and 0.8 of each decade),
dots of the mean over the sites, power laws fitted to 1-90 Hz (black lines) and the legend below the axes. Text is Arial 5 pt, editable.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

FONT, TITLE = 5, 7
REGION = (39.28, 50.08, 159.39, 241.18)
FX0, FXPD = 82.05, 17.13                 # x (pt) of 1 Hz and pt per decade of frequency
PY0, PYPD = 55.09, 8.21                  # y (pt) of a power of 10^5 and pt per decade of power
AXES = dict(spine_x=74.32, y=(50.1, 130.7), x_axis_y=134.73, x=(78.37, 159.39), y_tick=4.66, x_major=4.66, x_minor=2.67)
LW = 0.6058
EXPERIMENTS = {      # label, colour of the dots and of the legend, legend row (dot centre y, label centre y) and left edge of the label
    "M : Marques-Smith": ((0.8, 0.8, 0.8), 195.67, 196.76, 104.93),
    "NS : Biophy. spont.": ((0.8745, 0.1216, 0.1216), 204.34, 204.66, 104.43),
    "E: Biophy. evoked": ((0.8745, 0.4941, 0.1216), 212.84, 212.65, 105.27),
    "S: Synthetic": ((0.4118, 0.6392, 0.7882), 221.93, 221.98, 104.92),
    "H: Horvath": ((0.298, 0.298, 0.298), 230.23, 229.83, 104.92),
    "DS: Biophy. spont": ((0.1647, 0.5333, 0.3216), 238.91, 238.83, 105.28),
}
DRAW_ORDER = ["H: Horvath", "NS : Biophy. spont.", "E: Biophy. evoked", "DS: Biophy. spont", "M : Marques-Smith"]
LEGEND_DOT_X = {"M : Marques-Smith": 98.81, "NS : Biophy. spont.": 98.79, "E: Biophy. evoked": 98.95, "S: Synthetic": 99.26, "H: Horvath": 99.38, "DS: Biophy. spont": 99.56}
DOT, EDGE = 1.8, 0.1                     # diameter (pt) of the dots and width (pt) of their white edge


def _width(s):
    return TextPath((0, 0), s, size=FONT, prop=FontProperties(family="Arial")).get_extents().width


def _ctext(ax, cx, cy, s, rotation=0, bold=False, size=FONT):
    path = TextPath((0, 0), s, size=size, prop=FontProperties(family="Arial", weight="bold" if bold else "normal"))
    from matplotlib.transforms import Affine2D
    ext = path.transformed(Affine2D().rotate_deg(rotation)).get_extents()
    ax.text(cx - (ext.x0 + ext.x1) / 2, cy + (ext.y0 + ext.y1) / 2, s, ha="left", va="baseline", rotation=rotation, rotation_mode="anchor", fontsize=size,
            fontweight="bold" if bold else "normal")


def _pow10(ax, x, y_base, exponent, align):
    """10 to the power of an integer as two 5 pt texts (the exponent raised); align: 'right' (x is the right edge) or 'center'"""
    base, exp_s = "10", str(exponent).replace("-", "−")
    total = _width(base) + 0.4 + _width(exp_s)
    x0 = x - total if align == "right" else x - total / 2
    ax.text(x0, y_base, base, ha="left", va="baseline", fontsize=FONT)
    ax.text(x0 + _width(base) + 0.4, y_base - 3.7, exp_s, ha="left", va="baseline", fontsize=FONT)


def plot_psd(df_points, df_fits):
    """Fig. S2a from the source data: df_points (experiment, frequency_Hz, log10_power_mean_over_sites), df_fits (experiment, slope_mean, intercept_mean, band_*_Hz)"""
    g = AXES
    x0, y0, x1, y1 = REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    X = lambda f: FX0 + FXPD * np.log10(f)
    Y = lambda lp: PY0 - (np.asarray(lp) - 5.0) * PYPD

    for name in DRAW_ORDER:
        d = df_points[df_points["experiment"] == name]
        ax.plot(X(d["frequency_Hz"].values), Y(d["log10_power_mean_over_sites"].values), linestyle="None", marker="o", markersize=DOT, markeredgewidth=EDGE,
                markerfacecolor=EXPERIMENTS[name][0], markeredgecolor=(1, 1, 1), rasterized=True, zorder=2)
    for name in DRAW_ORDER:
        f = df_fits[df_fits["experiment"] == name].iloc[0]
        lf = np.linspace(np.log10(max(f["band_low_Hz"], 1)), np.log10(f["band_high_Hz"]), 79)
        ax.plot(FX0 + FXPD * lf, Y(f["slope_mean"] * lf + f["intercept_mean"]), color="k", linewidth=LW, solid_capstyle="butt", zorder=3)

    def line(xs, ys, lw=LW):
        ax.plot(xs, ys, color="k", linewidth=lw, solid_capstyle="butt")
    # axes: y (powers of 10^5, 10^3, ... 10^-3) and x (log frequency), detached
    line([g["spine_x"]] * 2, g["y"])
    for k, p in enumerate([5, 3, 1, -1, -3]):
        y = PY0 + k * 2 * PYPD
        line([g["spine_x"] - g["y_tick"], g["spine_x"]], [y, y])
        _pow10(ax, 64.86, y + 3.74, p, "right")
    line(g["x"], [g["x_axis_y"]] * 2)
    for decade in (-1, 0, 1, 2, 3, 4):
        for sub in (1, 2, 4, 6, 8):
            xt = X(sub * 10.0 ** decade)
            if xt < g["x"][0] + 0.5 or xt > g["x"][1] + 0.01:
                continue
            major = sub == 1 and decade in (1, 3)
            if sub == 1 and not major:
                continue
            line([xt, xt], [g["x_axis_y"], g["x_axis_y"] + (g["x_major"] if major else g["x_minor"])])
    for xt, e in [(98.9, 1), (133.5, 3)]:
        _pow10(ax, xt, 151.15, e, "center")
    _ctext(ax, 115.9, 156.93, "Frequency (Hz)")
    _ctext(ax, 42.27, 86.5, "Normalized power (ratio)", rotation=90)

    # legend below the axes
    for name, (color, cy_dot, cy_label, xl) in EXPERIMENTS.items():
        text = name.replace(" :", ":")           # the paper writes "M: Marques-Smith", the source data "M : Marques-Smith"
        ax.plot([LEGEND_DOT_X[name]], [cy_dot], marker="o", markersize=4.0, markeredgewidth=0, color=color, linestyle="None")
        path = TextPath((0, 0), text, size=FONT, prop=FontProperties(family="Arial")).get_extents()
        ax.text(xl - path.x0, cy_label + (path.y0 + path.y1) / 2, text, ha="left", va="baseline", fontsize=FONT)
    return fig


# Supplementary Fig. 2b: normalised PSD of the high-pass filtered traces in each layer (page of paper/figures/supp_fig2.pdf): canvas 277.9 x 296.8 pt
LAYERS_REGION = (199.91, 2.55, 477.76, 299.37)
# per layer: x of the y spine, y of the x axis, y of the top and bottom of the y spine, y (pt) and exponent of the first y tick, pt per decade of power,
# number of y ticks (two decades apart), centre of the title, centre of the y label (None: no label) and of the x label (None: no label)
LAYERS = {
    "L1": dict(sx=241.15, ay=123.54, ys=(55.88, 120.32), yt0=70.62, e0=-3, ppd=12.96, nt=2, title=(292.7, 42.64), ylabel=(203.78, 88.55), xlabel=None),
    "L2/3": dict(sx=381.34, ay=82.48, ys=(14.82, 79.26), yt0=26.06, e0=-3, ppd=13.475, nt=2, title=(431.4, 6.2), ylabel=None, xlabel=None),
    "L4": dict(sx=241.44, ay=235.96, ys=(168.29, 232.74), yt0=182.34, e0=-3, ppd=12.85, nt=2, title=(293.1, 156.75), ylabel=(202.9, 203.4), xlabel=(297.1, 266.9)),
    "L5": dict(sx=380.85, ay=178.42, ys=(110.76, 175.2), yt0=124.23, e0=-3, ppd=13.015, nt=2, title=(433.7, 101.3), ylabel=None, xlabel=None),
    "L6": dict(sx=380.09, ay=267.14, ys=(199.47, 263.91), yt0=199.95, e0=-2, ppd=12.15, nt=3, title=(432.4, 195.4), ylabel=None, xlabel=(430.4, 296.55)),
}
XTICK_LABELS = ("L1", "L4", "L6")            # the other panels have the ticks only
LAYER_EXPERIMENTS = ["H: Horvath", "M : Marques-Smith", "NS : Biophy. spont.", "E: Biophy. evoked", "DS: Biophy. spont", "S: Synthetic"]
XPD, X1HZ, AXW, AXH = 19.41, 8.77, 91.83, 64.44       # pt per decade of frequency, offset of 1 Hz from the y spine, width and height of the axes


def plot_psd_layers(df_points, df_fits):
    """Fig. S2b from the source data: df_points (layer, experiment, frequency_Hz, log10_normalised_power_mean_over_sites), df_fits (layer, experiment, slope_mean, intercept_mean, band_*_Hz)"""
    x0, y0, x1, y1 = LAYERS_REGION
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")

    def line(xs, ys, lw=0.721):
        ax.plot(xs, ys, color="k", linewidth=lw, solid_capstyle="butt")

    for layer, g in LAYERS.items():
        sx, ay = g["sx"], g["ay"]
        X = lambda f: sx + X1HZ + XPD * np.log10(f)
        Y = lambda lp: g["yt0"] + (g["e0"] - np.asarray(lp)) * g["ppd"]
        clip = plt.Rectangle((sx + 4.59, g["ys"][0]), AXW, g["ys"][1] - g["ys"][0], transform=ax.transData)
        names = [e for e in LAYER_EXPERIMENTS if e in set(df_points[df_points["layer"] == layer]["experiment"])]
        for name in names:
            d = df_points[(df_points["layer"] == layer) & (df_points["experiment"] == name)]
            artist, = ax.plot(X(d["frequency_Hz"].values), Y(d["log10_normalised_power_mean_over_sites"].values), linestyle="None", marker="o", markersize=DOT,
                              markeredgewidth=EDGE, markerfacecolor=EXPERIMENTS[name][0], markeredgecolor=(1, 1, 1), rasterized=True, zorder=2)
            artist.set_clip_path(clip)
        for name in names:
            for _, f in df_fits[(df_fits["layer"] == layer) & (df_fits["experiment"] == name)].iterrows():
                lf = np.log10(np.array([f["band_low_Hz"], f["band_high_Hz"]], dtype=float))
                ax.plot(X(10 ** lf), Y(f["slope_mean"] * lf + f["intercept_mean"]), color="k", linewidth=0.721, solid_capstyle="butt", zorder=3)

        # y axis (ticks two decades apart) and x axis (log frequency), detached
        line([sx] * 2, g["ys"])
        for k in range(g["nt"]):
            y = g["yt0"] + 2 * k * g["ppd"]
            line([sx - 5.54, sx], [y, y])
            _pow10_center(ax, sx - 19.03, y - 0.75, g["e0"] - 2 * k)
        line([sx + 4.59, sx + 4.59 + AXW], [ay, ay])
        for decade in range(-1, 5):
            xt = X(10.0 ** decade)
            if decade >= 0:
                line([xt, xt], [ay, ay + 5.56])
                if layer in XTICK_LABELS:
                    _pow10_center(ax, xt, ay + 14.3, decade)
            for sub in (2, 4, 6, 8):
                xm = X(sub * 10.0 ** decade)
                if sx + 4.59 <= xm <= sx + 4.59 + AXW:
                    line([xm, xm], [ay, ay + 3.18])
        _ctext(ax, *g["title"], layer, bold=True, size=TITLE)
        if g["ylabel"]:
            _ctext(ax, *g["ylabel"], "Normalized power (ratio)", rotation=90)
        if g["xlabel"]:
            _ctext(ax, *g["xlabel"], "Frequency (Hz)")
    return fig


def _pow10_center(ax, cx, cy, exponent):
    """10 to an integer power as two 5 pt texts, the bounding box of the pair centred on (cx, cy)"""
    base, exp_s = "10", str(exponent).replace("-", "−")
    total = _width(base) + 0.4 + _width(exp_s)
    x0 = cx - total / 2
    baseline = cy + 3.65
    ax.text(x0, baseline, base, ha="left", va="baseline", fontsize=FONT)
    ax.text(x0 + _width(base) + 0.4, baseline - 3.7, exp_s, ha="left", va="baseline", fontsize=FONT)
