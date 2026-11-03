"""Data and panels of Supplementary Fig. 7: firing rates of the sorted single-units (NS, Kilosort4) and of the ground truth neurons (GT)
of the spontaneous biophysical simulation of Neuropixels recordings, by layer

author: laquitainesteeve@gmail.com
"""

import numpy as np
import pandas as pd
import seaborn as sns
import spikeinterface as si
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.stats import norm

from src.nodes.analysis.npx_probe import true_fr
from src.nodes.postpro import metadata as meta
from src.nodes.validation import firing_rate as fr

LAYERS = ["L1", "L2/3", "L4", "L5", "L6"]
SORTED_PATH = "dataset/01_intermediate/sorting/npx_spont/SortingKS4_10m"
GT_PATH = "dataset/00_raw/ground_truth_npx_spont"
DURATION = 600  # seconds (first 10 minutes of the sorting and of the ground truth)
N_BINS, KERNEL_W = 31, 0.04
LOG_BINS = np.linspace(-3, 1, N_BINS)

# axes size (inches) of each panel in the paper: the figure is 7.5 x 5.5 in and its axes are [left, bottom, width, height] in fractions
PANEL_SIZE = {"a": (3.0, 1.925), "b": (3.0, 1.925), "c": (2.25, 1.5125), "d": (2.25, 1.1), "e": (2.25, 1.1)}


def get_firing_rates() -> pd.DataFrame:
    """Firing rates (spikes/s) of the sorted single-units of Kilosort4 (NS) and of all the ground truth neurons (GT) in each layer"""
    df_ns = fr.get_sorted_unit_meta(SORTED_PATH)
    df_ns = df_ns[np.isin(df_ns["layer"], LAYERS) & (df_ns["kslabel"] == "good")]
    gt = si.load_extractor(GT_PATH)
    gt = gt.frame_slice(start_frame=0, end_frame=DURATION * gt.sampling_frequency)
    gt = meta.set_firing_rates(gt, DURATION, "", save=False)
    df_gt = meta.get_gt_unit_meta_df_from_extractor(gt)
    df_gt = df_gt[np.isin(df_gt["layer"], LAYERS)]
    df = true_fr.get_log10fr_df(df_ns.iloc[:0], df_ns, df_gt, LAYERS, log=False)
    df = df[df["experiment"].isin(["GT", "NS"])]
    return df.reset_index(drop=True)[["experiment", "layer", "firing rate"]]


def to_log10(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["firing rate"] = np.log10(out["firing rate"])
    return out


def _gaussian_kernel_estimate(bins, rel_width):
    kernel = norm(0, rel_width * (bins[-1] - bins[0]))
    bin_c = 0.5 * (bins[:-1] + bins[1:])

    def func(pts):
        ret = [kernel.pdf(bin_c - _pt) for _pt in pts]
        ret = np.nansum(np.vstack([_ret / _ret.sum() for _ret in ret]), axis=0)
        return pd.Series(ret, index=bin_c)

    return func


def get_count_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted / true neuron count in each logarithmic firing rate bin (rows: bin centers, columns: layers), panel c"""
    flt_log = to_log10(df)
    log_df = flt_log.groupby(["experiment", "layer"])["firing rate"].apply(_gaussian_kernel_estimate(LOG_BINS, KERNEL_W))
    p_df = log_df.unstack("experiment").apply(lambda _row: _row["NS"] / (1e-1 + _row["GT"]), axis=1).unstack("layer")
    mn, mx = flt_log.groupby("layer")["firing rate"].min(), flt_log.groupby("layer")["firing rate"].max()
    for col in mn.index:
        p_df.loc[p_df.index < mn[col], col] = np.nan
        p_df.loc[p_df.index > mx[col], col] = np.nan
    return p_df[LAYERS].rename_axis("log firing rate")


def _bootstrap(smpls, n=100, m=100, func_cmp=lambda x, y: x / y):
    a, b = np.array(smpls["GT"]), np.array(smpls["NS"])
    a, b = a[~np.isinf(a)], b[~np.isinf(b)]
    est = np.array([func_cmp(np.nanmean(np.random.choice(a, m)), np.nanmean(np.random.choice(b, m))) for _ in range(n)])
    est = est[~np.isnan(est)]
    return pd.Series({"5th": np.percentile(est, 5), "mean": np.mean(est), "95th": np.percentile(est, 95)})


def get_mean_ratio(df: pd.DataFrame, log: bool, seed: int = 0) -> pd.DataFrame:
    """Mean firing rate of the ground truth divided by that of the sorted units (log=False, panel d), or difference of the means of the
    log10 firing rates (log=True, panel e), by layer and pooled, with a 90% confidence interval (5th and 95th percentiles of 100 bootstraps)"""
    np.random.seed(seed)
    d = to_log10(df) if log else df
    cmp = (lambda x, y: x - y) if log else (lambda x, y: x / y)
    by_layer = d.groupby(["experiment", "layer"])["firing rate"].apply(list).unstack("experiment")
    out = by_layer.apply(_bootstrap, axis=1, func_cmp=cmp).loc[LAYERS]
    out.loc["pooled"] = _bootstrap(d.set_index("experiment")["firing rate"], func_cmp=cmp)
    return out.rename_axis("layer")


def plot_violins(ax, df: pd.DataFrame, normalized: bool):
    """Panel a (normalized widths) and b (widths proportional to the neuron counts): violins of the log10 firing rates, GT (blue) and NS (orange)"""
    d = to_log10(df)
    sns.violinplot(d, x="layer", y="firing rate", split=True, hue="experiment", hue_order=["GT", "NS"], gap=0.1, width=1.0, linewidth=0.75, common_norm=False,
                   density_norm="width" if normalized else "count", cut=0 if normalized else 0.5, bw_adjust=0.75, legend=True if normalized else None, ax=ax)
    ax.set_yticks([-2, -1, 0, 1] if normalized else [-3, -2, -1, 0, 1])
    ax.set_ylabel("log firing rate")
    ax.set_title("Normalized distributions" if normalized else "Neuron counts")
    ax.set_frame_on(False)


def plot_count_ratio(ax, p_df: pd.DataFrame):
    for col in p_df.columns:
        ax.plot(p_df.index, p_df[col].values, label=col, linewidth=1)
    ax.legend()
    ax.set_xticks([-3, -2, -1, 0])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_frame_on(False)
    ax.set_xlabel("log firing rate")
    ax.set_ylabel("Sorted / true count")


def plot_mean_ratio(ax, stats: pd.DataFrame, log: bool):
    """Panel d (log=False) and e (log=True): mean (squares), 90% confidence interval (grey) by layer, and pooled mean (dashed)"""
    by_layer = stats.loc[LAYERS]
    x = np.arange(len(LAYERS))
    polygon = Polygon(np.vstack([np.hstack([x, x[::-1]]), np.hstack([by_layer["5th"].values, by_layer["95th"].values[::-1]])]).T, closed=True)
    ax.add_collection(PatchCollection([polygon], alpha=0.5, color="grey"))
    ax.plot(x, by_layer["mean"], marker="s", color="black", linewidth=1 if not log else None)
    ax.plot(x, stats.loc["pooled", "mean"] * np.ones_like(x), color="black", ls="--", linewidth=1 if not log else None)
    if log:
        ax.set_ylim([ax.get_ylim()[0], 0])
    if not log:
        ax.set_yticks([0, 1, 2])
    ax.set_xticks(x)
    ax.set_xticklabels(LAYERS)
    ax.set_frame_on(False)
    ax.set_ylabel("Mean log fr: True - sorted" if log else "Mean fr: True / sorted")
    ax.set_xlabel("layer")
