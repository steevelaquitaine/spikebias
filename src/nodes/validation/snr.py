"""Signal-to-noise ratio node
author: steeve.laquitaine@epfl.ch

Returns:
    _type_: _description_
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib
from concurrent.futures import ProcessPoolExecutor

from src.nodes.validation import amplitude as amp
import numpy as np

# SETUP PARAMETERS
RND_SEED = 0                # random seed
SFREQ_VIVO = 30000          # sampling frequency
SFREQ_SILICO = 40000        # sampling frequency

# FIGURE SETTINGS
COLOR_V = np.array([153,153,153]) / 255
COLOR_S = (0.84, 0.27, 0.2)
N_MAJOR_TICKS = 4
N_MINOR_TICKS = 12
YLIM = [1e-7, 1e8]


def plot_layer_snr_npx(
    axis,
    layer: str,
    layers_sili: list,
    layers_vivo: list,
    layers_e: list,
    snr_sili: list,
    snr_vivo: list,
    snr_e: list,
    n_bins: int,
    color_s: tuple,
    color_v: tuple,
    color_e: tuple,
    label_s: str,
    label_v: str,
    label_e: str,
    pm: dict,
):
    """plot snr distribution for specified layer

    Args:
        layer (str): _description_
        snr_sili (list): sites x timepoints
        snr_vivo (list): _description_
        snr_e (list): evoked
        n_bins (int): _description_
        pm (dict): of plot parameters
    """
    # 1 - get this layer
    snr_sili_layer_i = snr_sili[layers_sili == layer, :]
    snr_vivo_layer_i = snr_vivo[layers_vivo == layer, :]
    if len(snr_e) > 1:
        snr_e_layer_i = snr_e[layers_e == layer, :]

    # 2 - calculate common bins (2 mins)
    snr_max_layer_i = np.max(
        [np.array(snr_vivo_layer_i).max(),
         np.array(snr_sili_layer_i).max(),
         np.array(snr_e_layer_i).max()]
    )
    snr_min_layer_i = np.min(
        [np.array(snr_vivo_layer_i).min(),
         np.array(snr_sili_layer_i).min(),
         np.array(snr_e_layer_i).min()]
    )
    step_layer_i = (snr_max_layer_i - snr_min_layer_i) / n_bins
    bins = np.arange(
        snr_min_layer_i, snr_max_layer_i + step_layer_i / 2, step_layer_i
    )

    # 3 - Compute the snr pdf stats over sites (1 min)
    # vivo
    mean_vivo_layer_i, ci_vivo_layer_i, _ = amp.get_snr_pdfs(
        snr_vivo_layer_i, bins
    )
    # silico
    (
        mean_sili_layer_i,
        ci_sili_layer_i, 
        _
    ) = amp.get_snr_pdfs(snr_sili_layer_i, bins)
    # evoked
    (
        mean_e_layer_i,
        ci_e_layer_i, 
        _
    ) = amp.get_snr_pdfs(snr_e_layer_i, bins)

    # vivo
    amp.plot_proba_dist_stats(
        axis,
        mean_vivo_layer_i[mean_vivo_layer_i > 0],
        ci_vivo_layer_i[mean_vivo_layer_i > 0],
        bins[:-1][mean_vivo_layer_i > 0],
        color=color_v,
        ci_color=color_v,
        label=label_v,
        pm=pm
    )
    # silico
    amp.plot_proba_dist_stats(
        axis,
        mean_sili_layer_i[mean_sili_layer_i > 0],
        ci_sili_layer_i[mean_sili_layer_i > 0],
        bins[:-1][mean_sili_layer_i > 0],
        color=color_s,
        ci_color=color_s,
        label=label_s,
        pm=pm
    )    
    # evoked
    amp.plot_proba_dist_stats(
        axis,
        mean_e_layer_i[mean_e_layer_i > 0],
        ci_e_layer_i[mean_e_layer_i > 0],
        bins[:-1][mean_e_layer_i > 0],
        color=color_e,
        ci_color=color_e,
        label=label_e,
        pm=pm
    )        
    # legend
    axis.set_yscale("log")
    axis.spines[["right", "top"]].set_visible(False)
    axis.tick_params(which="both")

    # show minor ticks
    axis.tick_params(which="major")
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0, subs=(0.5, 1), numticks=2
    )    
    axis.yaxis.set_major_locator(locmaj)
    axis.yaxis.set_minor_locator(locmin)
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.spines[["right", "top"]].set_visible(False)

    # disconnect axes (R style)
    axis.spines["bottom"].set_position(("axes", -0.05))
    axis.yaxis.set_ticks_position("left")
    axis.spines["left"].set_position(("axes", -0.05))
    axis.set_box_aspect(1)  # square axis
    return axis, bins


def plot_layer_snr_horv(
    axis,
    layer: str,
    layers_sili: list,
    layers_vivo: list,
    snr_sili: list,
    snr_vivo: list,
    n_bins: int,
    color_s: tuple,
    color_v: tuple,
    label_s: str,
    label_v: str,
    pm: dict,
):
    """plot snr distribution for specified layer

    Args:
        layer (str): _description_
        layers_sili (list): site layers in silico
        layers_vivo (list): site layers in vivo
        snr_sili (list): sites x timepoints
        snr_vivo (list): _description_
        n_bins (int): _description_
        xlim (tuple): x-axis limits
        pm (dict): of plot parameters
    """
    # 1 - get this layer
    snr_sili_layer_i = snr_sili[layers_sili == layer, :]
    snr_vivo_layer_i = snr_vivo[layers_vivo == layer, :]

    # 2 - calculate common bins (2 mins)
    snr_max_layer_i = np.max(
        [np.array(snr_vivo_layer_i).max(), np.array(snr_sili_layer_i).max()]
    )
    snr_min_layer_i = np.min(
        [np.array(snr_vivo_layer_i).min(), np.array(snr_sili_layer_i).min()]
    )
    step_layer_i = (snr_max_layer_i - snr_min_layer_i) / n_bins
    bins = np.arange(
        snr_min_layer_i, snr_max_layer_i + step_layer_i / 2, step_layer_i
    )

    # 3 - Compute the snr pdf stats over sites (1 min)
    # vivo
    mean_vivo_layer_i, ci_vivo_layer_i, _ = amp.get_snr_pdfs(
        snr_vivo_layer_i, bins
    )
    # silico
    (
        mean_sili_layer_i,
        ci_sili_layer_i, 
        _
    ) = amp.get_snr_pdfs(snr_sili_layer_i, bins)

    # vivo
    amp.plot_proba_dist_stats(
        axis,
        mean_vivo_layer_i[mean_vivo_layer_i > 0],
        ci_vivo_layer_i[mean_vivo_layer_i > 0],
        bins[:-1][mean_vivo_layer_i > 0],
        color=color_v,
        ci_color=color_v,
        label=label_v,
        pm=pm
    )
    # silico
    amp.plot_proba_dist_stats(
        axis,
        mean_sili_layer_i[mean_sili_layer_i > 0],
        ci_sili_layer_i[mean_sili_layer_i > 0],
        bins[:-1][mean_sili_layer_i > 0],
        color=color_s,
        ci_color=color_s,
        label=label_s,
        pm=pm
    )    
    # legend
    axis.set_yscale("log")
    axis.spines[["right", "top"]].set_visible(False)
    axis.tick_params(which="both")

    # show minor ticks
    axis.tick_params(which="major")
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0, subs=(0.5, 1), numticks=2
    )    
    axis.yaxis.set_major_locator(locmaj)
    axis.yaxis.set_minor_locator(locmin)
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.spines[["right", "top"]].set_visible(False)

    # disconnect axes (R style)
    axis.spines["bottom"].set_position(("axes", -0.05))
    axis.yaxis.set_ticks_position("left")
    axis.spines["left"].set_position(("axes", -0.05))
    axis.set_box_aspect(1)  # square axis
    return axis, bins


def plot_snr_for_layer_5_npx(
    axis,
    layers_s: list,
    layers_v: list,
    layers_e: list,
    snr_s: list,
    snr_v: list,
    snr_e: list,
    snr_b: list,
    n_bins: int,
    color_s: tuple,
    color_v: tuple,
    color_e: tuple,
    color_b: tuple,
    label_s: str,
    label_v: str,  
    label_e: str,
    label_b: str, 
    pm: dict,
):
    """plot distribution of signal-to-noise ratio
    of the recording sites in layer 5

    Args:
        layer (str): _description_
        snr_sili (list): sites x timepoints
        snr_vivo (list): _description_
        n_bins (int): _description_
    
    note: lasts 3 mins because the Synthetic model 
    has 384 sites
    """
    # 1 - get this layer
    snr_s_layer_i = snr_s[layers_s == "L5", :]
    snr_v_layer_i = snr_v[layers_v == "L5", :]
    snr_e_layer_i = snr_e[layers_e == "L5", :]

    # 2 - calculate common bins (2 mins)
    snr_max_layer_i = np.nanmax(
        [np.array(snr_v_layer_i).max(), np.array(snr_s_layer_i).max(), np.max(snr_e_layer_i), np.max(snr_b)]
    )
    snr_min_layer_i = np.nanmin(
        [np.array(snr_v_layer_i).min(), np.array(snr_s_layer_i).min(), np.min(snr_e_layer_i), np.min(snr_b)]
    )
    step_layer_i = (snr_max_layer_i - snr_min_layer_i) / n_bins
    bins = np.arange(
        snr_min_layer_i, snr_max_layer_i + step_layer_i / 2, step_layer_i
    )

    # 3 - Compute the snr pdf stats over sites (1 min)
    # vivo
    mean_v, ci_v,_ = amp.get_snr_pdfs(
        snr_v_layer_i, bins
    )
    # buccino
    (
        mean_b,
        ci_b,
        _
    ) = amp.get_snr_pdfs(snr_b, bins)
    # silico
    (
        mean_s_layer_i,
        ci_s_layer_i,
        _
    ) = amp.get_snr_pdfs(snr_s_layer_i, bins)
    # evoked
    (
        mean_e_layer_i,
        ci_e_layer_i,
        _
    ) = amp.get_snr_pdfs(snr_e_layer_i, bins)
    
    # plot *****************************
    # vivo
    amp.plot_proba_dist_stats(
        axis,
        mean_v[mean_v > 0],
        ci_v[mean_v > 0],
        bins[:-1][mean_v > 0],
        color=color_v,
        ci_color=color_v,
        label=label_v,
        pm=pm, 
    )
    # silico
    amp.plot_proba_dist_stats(
        axis,
        mean_s_layer_i[mean_s_layer_i > 0],
        ci_s_layer_i[mean_s_layer_i > 0],
        bins[:-1][mean_s_layer_i > 0],
        color=color_s,
        ci_color=color_s,
        label=label_s,
        pm=pm
    )
    # evoked
    amp.plot_proba_dist_stats(
        axis,
        mean_e_layer_i[mean_e_layer_i > 0],
        ci_e_layer_i[mean_e_layer_i > 0],
        bins[:-1][mean_e_layer_i > 0],
        color=color_e,
        ci_color=color_e,
        label=label_e,
        pm=pm
    )    
    # buccino
    amp.plot_proba_dist_stats(
        axis,
        mean_b[mean_b > 0],
        ci_b[mean_b > 0],
        bins[:-1][mean_b > 0],
        color=color_b,
        ci_color=color_b,
        label=label_b,
        pm=pm
    )

    # legend
    axis.set_yscale("log")
    axis.spines[["right", "top"]].set_visible(False)
    axis.tick_params(which="both")
    axis.set_box_aspect(1)  # square axis

    # show minor ticks
    axis.tick_params(which="major")
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0, subs=(0.5, 1), numticks=2
    )    
    axis.yaxis.set_major_locator(locmaj)
    axis.yaxis.set_minor_locator(locmin)
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.spines[["right", "top"]].set_visible(False)

    # disconnect axes (R style)
    axis.spines["bottom"].set_position(("axes", -0.05))
    axis.yaxis.set_ticks_position("left")
    axis.spines["left"].set_position(("axes", -0.05))
    return axis, bins


def plot_snr_for_layer_5_horv(
    axis,
    layers_s: list,
    layers_v: list,
    snr_s: list,
    snr_v: list,
    n_bins: int,
    color_s: tuple,
    color_v: tuple,
    label_s: str,
    label_v: str,  
    pm: dict,
):
    """plot snr distribution for specified layer

    Args:
        layer (str): _description_
        snr_sili (list): sites x timepoints
        snr_vivo (list): _description_
        n_bins (int): _description_
    """
    # 1 - get this layer
    snr_s_layer_i = snr_s[layers_s == "L5", :]
    snr_v_layer_i = snr_v[layers_v == "L5", :]

    # 2 - calculate common bins (2 mins)
    snr_max_layer_i = np.nanmax(
        [np.array(snr_v_layer_i).max(), np.array(snr_s_layer_i).max()]
    )
    snr_min_layer_i = np.nanmin(
        [np.array(snr_v_layer_i).min(), np.array(snr_s_layer_i).min()]
    )
    step_layer_i = (snr_max_layer_i - snr_min_layer_i) / n_bins
    bins = np.arange(
        snr_min_layer_i, snr_max_layer_i + step_layer_i / 2, step_layer_i
    )

    # 3 - Compute the snr pdf stats over sites (1 min)
    # vivo
    mean_v_layer_i, ci_v_layer_i,_ = amp.get_snr_pdfs(
        snr_v_layer_i, bins
    )
    # silico
    (
        mean_s_layer_i,
        ci_s_layer_i,
        _
    ) = amp.get_snr_pdfs(snr_s_layer_i, bins)

    # vivo
    amp.plot_proba_dist_stats(
        axis,
        mean_v_layer_i[mean_v_layer_i > 0],
        ci_v_layer_i[mean_v_layer_i > 0],
        bins[:-1][mean_v_layer_i > 0],
        color=color_v,
        ci_color=color_v,
        label=label_v,
        pm=pm, 
    )
    # silico
    amp.plot_proba_dist_stats(
        axis,
        mean_s_layer_i[mean_s_layer_i > 0],
        ci_s_layer_i[mean_s_layer_i > 0],
        bins[:-1][mean_s_layer_i > 0],
        color=color_s,
        ci_color=color_s,
        label=label_s,
        pm=pm
    )

    # legend
    axis.set_yscale("log")
    axis.spines[["right", "top"]].set_visible(False)
    axis.tick_params(which="both")
    axis.set_box_aspect(1)  # square axis

    # show minor ticks
    axis.tick_params(which="major")
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0, subs=(0.5, 1), numticks=2
    )    
    axis.yaxis.set_major_locator(locmaj)
    axis.yaxis.set_minor_locator(locmin)
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.spines[["right", "top"]].set_visible(False)

    # disconnect axes (R style)
    axis.spines["bottom"].set_position(("axes", -0.05))
    axis.yaxis.set_ticks_position("left")
    axis.spines["left"].set_position(("axes", -0.05))
    return axis, bins


def get_site_snr(trace: np.array, site: int):
    """calculate signal-to-noise ratio
    of a single site
    
    Noise is the mean absolute deviation
    of the entire trace
    
    Args:
        trace (np.array):
        site: silent argument used by 
        ProcessPoolExecutor()
    """
    mad = pd.DataFrame(trace).mad().values
    return trace / mad


def get_snrs_parallel(traces: np.ndarray):
    """calculate signal-to-noise ratios
    of all sites (parallelized)

    Args:
        traces (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    # takes 2 min (instead of 70 min w/o multiprocessing)
    nsites = traces.shape[1]

    # compute power for each site trace
    # in parallel with a pool of workers
    with ProcessPoolExecutor() as executor:
        site_snrs = executor.map(
            get_site_snr,
            traces.T,
            np.arange(0, nsites, 1),
        )
    snr_by_site = list(site_snrs)
    return np.array(snr_by_site)


def get_pdf_median_ci(pdfs):
    dist_mean = np.median(np.array(pdfs), axis=0)
    dist_ci = 1.96 * np.std(pdfs, axis=0) / np.sqrt(len(pdfs[0]))
    return dist_mean, dist_ci