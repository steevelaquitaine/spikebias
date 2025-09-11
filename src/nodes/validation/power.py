
import copy
from sklearn.linear_model import LinearRegression
import pandas as pd
import scipy 
import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
from src.nodes import utils
import seaborn as sns




def get_power_old(traces, n_contacts_reyes, samp_frq):    
    """calculate the power spectrum of each trace

    Args:
        traces (_type_): _description_
        n_contacts_reyes (_type_): _description_
        samp_frq (_type_): _description_

    Returns:
        _type_: _description_
    """
    contact_ids = np.arange(0, n_contacts_reyes, 1)
    powers = []
    freqs = []
    traces = traces.get_traces()
    for c_i in contact_ids:
        (freq, S) = scipy.signal.periodogram(traces[:,c_i], samp_frq, scaling='density') 
        powers.append(S)
        freqs.append(freq)
    powers = np.array(powers)
    freqs = np.array(freqs)
    return powers, freqs


def get_power(traces, sfreq):    

    """calculate the power spectrum of each trace

    Args:
        traces (np.ndarray): timepoints x sites voltage trace
        sfreq (_type_): voltage trace sampling frequency

    Returns:
        _type_: _description_
    """

    nsites = traces.shape[1]
    powers = []
    freqs = []
    traces = traces.get_traces()
    for site,_ in enumerate(nsites):
        (freq, power) = scipy.signal.periodogram(traces[:,site], sfreq, scaling='density')         
        powers.append(power)
        freqs.append(freq)
    powers = np.array(powers)
    freqs = np.array(freqs)
    return {
        "powers": powers, 
        "freqs": freqs
        }


def get_psd_plot_mean_and_ci_old(
    raw_hv, raw_mv, raw_hs, raw_ms, raw_b, pre_hv, pre_mv, pre_hs, pre_ms, pre_b
):
    """calculate PSD means and confidence intervals"""
    # (11s) average over sites
    psd_mean_hv = np.median(raw_hv["power"], axis=0)
    psd_mean_mv = np.median(raw_mv["power"], axis=0)
    psd_mean_hs = np.median(raw_hs["power"], axis=0)
    psd_mean_ms = np.median(raw_ms["power"], axis=0)
    psd_mean_b = np.median(raw_b["power"], axis=0)
    psd_mean_pre_hv = np.median(pre_hv["power"], axis=0)
    psd_mean_pre_hs = np.median(pre_hs["power"], axis=0)
    psd_mean_pre_mv = np.median(pre_mv["power"], axis=0)
    psd_mean_pre_ms = np.median(pre_ms["power"], axis=0)
    psd_mean_pre_b = np.median(pre_b["power"], axis=0)

    # confidence intervals
    # vivo ---------------------
    # horvath
    n_samples = raw_hv["power"].shape[0]
    ci_raw_hv = 1.96 * np.std(raw_hv["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_hv = 1.96 * np.std(pre_hv["power"], axis=0) / np.sqrt(n_samples)
    # marques
    n_samples = raw_mv["power"].shape[0]
    ci_raw_mv = 1.96 * np.std(raw_mv["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_mv = 1.96 * np.std(pre_mv["power"], axis=0) / np.sqrt(n_samples)
    # silico ---------------------
    # horvath
    n_samples = raw_hs["power"].shape[0]
    ci_raw_hs = 1.96 * np.std(raw_hs["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_hs = 1.96 * np.std(pre_hs["power"], axis=0) / np.sqrt(n_samples)
    # marques
    n_samples = raw_ms["power"].shape[0]
    ci_raw_ms = 1.96 * np.std(raw_ms["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_ms = 1.96 * np.std(pre_ms["power"], axis=0) / np.sqrt(n_samples)
    # buccino
    n_samples = raw_b["power"].shape[0]
    ci_raw_b = 1.96 * np.std(raw_b["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_b = 1.96 * np.std(pre_b["power"], axis=0) / np.sqrt(n_samples)

    return (
        psd_mean_hv,
        psd_mean_mv,
        psd_mean_hs,
        psd_mean_ms,
        psd_mean_b,
        psd_mean_pre_hv,
        psd_mean_pre_mv,
        psd_mean_pre_hs,
        psd_mean_pre_ms,
        psd_mean_pre_b,
        ci_raw_hv,
        ci_raw_mv,
        ci_raw_hs,
        ci_raw_ms,
        ci_raw_b,
        ci_pre_hv,
        ci_pre_mv,
        ci_pre_hs,
        ci_pre_ms,
        ci_pre_b,
    )
    
    
def get_psd_plot_mean_and_ci(raw_hv1,
    raw_hv2,
    raw_hv3,
    raw_hs1,
    raw_hs2,
    raw_hs3,
    raw_nv,
    raw_ns,
    raw_ne,
    raw_nb,
    pre_hv1,
    pre_hv2,
    pre_hv3,
    pre_hs1,
    pre_hs2,
    pre_hs3,
    pre_nv,
    pre_ns,
    pre_ne,
    pre_nb,
):
    """calculate PSD means and confidence intervals"""
    
    # (11s) average over sites
    # horvath
    # in vivo
    psd_mean_raw_hv1 = np.mean(raw_hv1["power"], axis=0)
    psd_mean_raw_hv2 = np.mean(raw_hv2["power"], axis=0)
    psd_mean_raw_hv3 = np.mean(raw_hv3["power"], axis=0)
    # biophy
    psd_mean_raw_hs1 = np.mean(raw_hs1["power"], axis=0)
    psd_mean_raw_hs2 = np.mean(raw_hs2["power"], axis=0)
    psd_mean_raw_hs3 = np.mean(raw_hs3["power"], axis=0)    
    
    # preprocessed
    psd_mean_pre_hv1 = np.mean(pre_hv1["power"], axis=0)
    psd_mean_pre_hv2 = np.mean(pre_hv2["power"], axis=0)
    psd_mean_pre_hv3 = np.mean(pre_hv3["power"], axis=0)
    psd_mean_pre_hs1 = np.mean(pre_hs1["power"], axis=0)
    psd_mean_pre_hs2 = np.mean(pre_hs2["power"], axis=0)
    psd_mean_pre_hs3 = np.mean(pre_hs3["power"], axis=0)    
    
    # neuropixels
    # raw
    psd_mean_raw_nv = np.mean(raw_nv["power"], axis=0)
    psd_mean_raw_ns = np.mean(raw_ns["power"], axis=0)
    psd_mean_raw_ne = np.mean(raw_ne["power"], axis=0)
    psd_mean_raw_nb = np.mean(raw_nb["power"], axis=0)
    # preprocessed 
    psd_mean_pre_nv = np.mean(pre_nv["power"], axis=0)
    psd_mean_pre_ns = np.mean(pre_ns["power"], axis=0)
    psd_mean_pre_ne = np.mean(pre_ne["power"], axis=0)
    psd_mean_pre_nb = np.mean(pre_nb["power"], axis=0)
    
    # confidence intervals
    
    # horvath ******************
    # vivo
    # probe 1
    n_samples = raw_hv1["power"].shape[0]
    ci_raw_hv1 = 1.96 * np.std(raw_hv1["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_hv1 = 1.96 * np.std(pre_hv1["power"], axis=0) / np.sqrt(n_samples)
    # probe 2
    n_samples = raw_hv2["power"].shape[0]
    ci_raw_hv2 = 1.96 * np.std(raw_hv2["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_hv2 = 1.96 * np.std(pre_hv2["power"], axis=0) / np.sqrt(n_samples)
    # probe 3
    n_samples = raw_hv3["power"].shape[0]
    ci_raw_hv3 = 1.96 * np.std(raw_hv3["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_hv3 = 1.96 * np.std(pre_hv3["power"], axis=0) / np.sqrt(n_samples)
    
    # biophy.
    # probe 1
    n_samples = raw_hs1["power"].shape[0]
    ci_raw_hs1 = 1.96 * np.std(raw_hs1["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_hs1 = 1.96 * np.std(pre_hs1["power"], axis=0) / np.sqrt(n_samples)
    # probe 2
    n_samples = raw_hs2["power"].shape[0]
    ci_raw_hs2 = 1.96 * np.std(raw_hs2["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_hs2 = 1.96 * np.std(pre_hs2["power"], axis=0) / np.sqrt(n_samples)
    # probe 3
    n_samples = raw_hs3["power"].shape[0]
    ci_raw_hs3 = 1.96 * np.std(raw_hs3["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_hs3 = 1.96 * np.std(pre_hs3["power"], axis=0) / np.sqrt(n_samples)
    
    # neuropixels
    # vivo
    n_samples = raw_nv["power"].shape[0]
    ci_raw_nv = 1.96 * np.std(raw_nv["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_nv = 1.96 * np.std(pre_nv["power"], axis=0) / np.sqrt(n_samples)
    # biophy. spont.
    n_samples = raw_ns["power"].shape[0]
    ci_raw_ns = 1.96 * np.std(raw_ns["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_ns = 1.96 * np.std(pre_ns["power"], axis=0) / np.sqrt(n_samples)    
    # biophy. evoked
    n_samples = raw_ne["power"].shape[0]
    ci_raw_ne = 1.96 * np.std(raw_ne["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_ne = 1.96 * np.std(pre_ne["power"], axis=0) / np.sqrt(n_samples)
    # synth. (Buccino)
    n_samples = raw_nb["power"].shape[0]
    ci_raw_nb = 1.96 * np.std(raw_nb["power"], axis=0) / np.sqrt(n_samples)
    ci_pre_nb = 1.96 * np.std(pre_nb["power"], axis=0) / np.sqrt(n_samples)

    return (
        psd_mean_raw_hv1,
        psd_mean_raw_hv2,
        psd_mean_raw_hv3,
        psd_mean_raw_hs1,
        psd_mean_raw_hs2,
        psd_mean_raw_hs3,
        psd_mean_raw_nv,
        psd_mean_raw_ns,
        psd_mean_raw_ne,
        psd_mean_raw_nb,
        psd_mean_pre_hv1,
        psd_mean_pre_hv2,
        psd_mean_pre_hv3,
        psd_mean_pre_hs1,
        psd_mean_pre_hs2,
        psd_mean_pre_hs3,
        psd_mean_pre_nv,
        psd_mean_pre_ns,
        psd_mean_pre_ne,
        psd_mean_pre_nb,
        ci_raw_hv2,
        ci_raw_hv3,
        ci_raw_hs1,
        ci_raw_hs2,
        ci_raw_hs3,
        ci_raw_nv,
        ci_raw_ns,
        ci_raw_ne,
        ci_raw_nb,
        ci_pre_hv1,
        ci_pre_hv2,
        ci_pre_hv3,
        ci_pre_hs1,
        ci_pre_hs2,
        ci_pre_hs3,
        ci_pre_nv,
        ci_pre_ns,
        ci_pre_ne,
        ci_pre_nb,        
    )
            

def get_psd_plot_mean_and_ci_for_layer(
    layer: str,
    lyrs_v: list,
    lyrs_s: list,
    out_raw_vivo: dict,
    out_raw_sili: dict,
    out_prep_vivo: dict,
    out_prep_sili: dict,
):
    # get sites in layer
    power_v = out_raw_vivo["power"][lyrs_v == layer, :]
    power_s = out_raw_sili["power"][lyrs_s == layer, :]
    pre_power_v = out_prep_vivo["power"][lyrs_v == layer, :]
    pre_power_s = out_prep_sili["power"][lyrs_s == layer, :]

    # (11s) average over sites
    psd_mean_raw_vivo = np.median(power_v, axis=0)
    psd_mean_raw_sili = np.median(power_s, axis=0)
    psd_mean_prep_vivo = np.median(pre_power_v, axis=0)
    psd_mean_prep_sili = np.median(pre_power_s, axis=0)

    # confidence intervals
    # vivo
    n_samples = power_v.shape[0]
    ci_raw_vivo = 1.96 * np.std(power_v, axis=0) / np.sqrt(n_samples)
    ci_prep_vivo = 1.96 * np.std(pre_power_v, axis=0) / np.sqrt(n_samples)

    # sili
    n_samples = power_s.shape[0]
    ci_raw_sili = 1.96 * np.std(power_s, axis=0) / np.sqrt(n_samples)
    ci_prep_sili = 1.96 * np.std(pre_power_s, axis=0) / np.sqrt(n_samples)

    return (
        psd_mean_raw_vivo,
        psd_mean_raw_sili,
        psd_mean_prep_vivo,
        psd_mean_prep_sili,
        ci_raw_vivo,
        ci_raw_sili,
        ci_prep_vivo,
        ci_prep_sili,
        out_raw_vivo["freq"],
        out_raw_sili["freq"],
        out_prep_vivo["freq"],
        out_prep_sili["freq"],
    )


def get_psd_plot_mean_and_ci_for_layer_5(
    lyrs_v: list,
    lyrs_s: list,
    out_raw_v: dict,
    out_raw_s: dict,
    out_raw_b: dict,
    out_prep_v: dict,
    out_prep_s: dict,
    out_prep_b: dict,
):

    # get sites in layer
    power_v = out_raw_v["power"][lyrs_v == "L5", :]
    power_s = out_raw_s["power"][lyrs_s == "L5", :]
    power_b = out_raw_b["power"]
    pre_power_v = out_prep_v["power"][lyrs_v == "L5", :]
    pre_power_s = out_prep_s["power"][lyrs_s == "L5", :]
    pre_power_b = out_prep_b["power"]

    # (11s) average over sites
    psd_mean_raw_vivo = np.median(power_v, axis=0)
    psd_mean_raw_sili = np.median(power_s, axis=0)
    psd_mean_raw_b = np.median(power_b, axis=0)
    psd_mean_prep_vivo = np.median(pre_power_v, axis=0)
    psd_mean_prep_sili = np.median(pre_power_s, axis=0)
    psd_mean_prep_b = np.median(pre_power_b, axis=0)

    # confidence intervals
    # vivo
    n_samples = power_v.shape[0]
    ci_raw_vivo = 1.96 * np.std(power_v, axis=0) / np.sqrt(n_samples)
    ci_prep_vivo = 1.96 * np.std(pre_power_v, axis=0) / np.sqrt(n_samples)

    # sili
    n_samples = power_s.shape[0]
    ci_raw_sili = 1.96 * np.std(power_s, axis=0) / np.sqrt(n_samples)
    ci_prep_sili = 1.96 * np.std(pre_power_s, axis=0) / np.sqrt(n_samples)

    # buccino
    n_samples = power_b.shape[0]
    ci_raw_b = 1.96 * np.std(power_b, axis=0) / np.sqrt(n_samples)
    ci_prep_b = 1.96 * np.std(pre_power_b, axis=0) / np.sqrt(n_samples)

    return (
        psd_mean_raw_vivo,
        psd_mean_raw_sili,
        psd_mean_raw_b,
        psd_mean_prep_vivo,
        psd_mean_prep_sili,
        psd_mean_prep_b,
        ci_raw_vivo,
        ci_raw_sili,
        ci_raw_b,
        ci_prep_vivo,
        ci_prep_sili,
        ci_prep_b,
        out_raw_v["freq"],
        out_raw_s["freq"],
        out_raw_b["freq"],
        out_prep_v["freq"],
        out_prep_s["freq"],
        out_prep_b["freq"],
    )


def plot_for_layer(
    psd_mean_raw_vivo,
    psd_mean_raw_sili,
    psd_mean_prep_vivo,
    psd_mean_prep_sili,
    ci_raw_vivo,
    ci_prep_vivo,
    ci_raw_sili,
    ci_prep_sili,
    freq_raw_v,
    freq_raw_s,
    freq_prep_v,
    freq_prep_s,
    out_raw_vivo,
    out_raw_sili,
    out_prep_vivo,
    out_prep_sili,
    site_layers_vivo,
    site_layers_sili,
    layer,
    ylim_r,
    ylim_p,
):
    """plot power spectrum density for the
    specified layer

    Args:
        layer (str): layer "L1", "L2_3", "L4", "L5", "L6"
        site_layers_sili (list): layers of each in silico sites
        site_layers_vivo (list): layers of each in vivo sites
        out_raw_vivo (dict): _description_
        out_raw_sili (dict): _description_
    """

    N_MAJOR_TICKS = 4
    SIZE = 3
    ALPHA = 0.7
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)

    # Raw panel --------------------------------

    # VIVO
    axes[0].plot(
        freq_raw_v,
        psd_mean_raw_vivo,
        linestyle="none",
        marker="o",
        markersize=SIZE,
        color=COLOR_VIVO,
        label="vivo",
        rasterized=True,  # cheaper
    )
    axes[0].plot(
        freq_raw_s,
        psd_mean_raw_sili,
        linestyle="none",
        marker="o",
        markersize=SIZE,
        color=COLOR_SILI,
        label="sili",
        rasterized=True,  # cheaper
    )

    # confidence interval
    axes[0].fill_between(
        freq_raw_v,
        (psd_mean_raw_vivo - ci_raw_vivo),
        (psd_mean_raw_vivo + ci_raw_vivo),
        color=COLOR_VIVO,
        linewidth=0,
        alpha=ALPHA,
        rasterized=True,
    )
    axes[0].fill_between(
        freq_raw_s,
        (psd_mean_raw_sili - ci_raw_sili),
        (psd_mean_raw_sili + ci_raw_sili),
        color=COLOR_SILI,
        linewidth=0,
        alpha=ALPHA,
        rasterized=True,
    )

    # legend
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_xlabel("")
    axes[0].set_xscale("log")
    axes[0].spines[["right", "top"]].set_visible(False)
    axes[0].set_ylim(ylim_r)
    axes[0].set_xlim([-1, SFREQ_S / 2])
    axes[0].tick_params(axis="x", which="minor", colors="black")
    axes[0].tick_params(axis="x", which="major", colors="black")

    # show minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
        numticks=N_MINOR_TICKS,
    )
    axes[0].tick_params(which="both")
    axes[0].xaxis.set_major_locator(locmaj)
    axes[0].xaxis.set_minor_locator(locmin)
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # disconnect axes (R style)
    axes[0].spines["bottom"].set_position(("axes", -0.05))
    axes[0].yaxis.set_ticks_position("left")
    axes[0].spines["left"].set_position(("axes", -0.05))
    axes[0].set_ylim(ylim_r)
    axes[0].set_xlim([0, SFREQ_S / 2])

    # Fit panel ************************************************************

    # get sites in layer
    out_raw_vivo2 = copy.copy(out_raw_vivo)
    out_raw_sili2 = copy.copy(out_raw_sili)
    out_raw_vivo2["power"] = out_raw_vivo2["power"][site_layers_vivo == layer, :]
    out_raw_sili2["power"] = out_raw_sili2["power"][site_layers_sili == layer, :]

    # plot
    plot_fits_all(axes[1], out_raw_vivo2, out_raw_sili2, pm)

    # Preprocessing ************************************************************

    # VIVO
    axes[2].plot(
        freq_prep_v,
        psd_mean_prep_vivo,
        linestyle="none",
        marker="o",
        markersize=SIZE,
        color=COLOR_VIVO,
        label="vivo",
        rasterized=True,  # cheaper
    )
    axes[2].plot(
        freq_prep_s,
        psd_mean_prep_sili,
        color=COLOR_SILI,
        label="silico",
        marker="o",
        markersize=SIZE,
        rasterized=True,  # cheaper
    )

    # legend
    # axes[1].set_xlabel("Frequency (Hz)")
    axes[2].set_xlabel("")
    axes[2].set_xscale("log")
    axes[2].spines[["right", "top"]].set_visible(False)
    axes[2].set_ylim(ylim_p)
    axes[2].set_xlim([-1, SFREQ_S / 2])
    axes[2].tick_params(axis="x", which="minor", colors="black")
    axes[2].tick_params(axis="x", which="major", colors="black")

    # show minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
        numticks=N_MINOR_TICKS,
    )
    axes[2].tick_params(which="both")
    axes[2].xaxis.set_major_locator(locmaj)
    axes[2].xaxis.set_minor_locator(locmin)
    axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # disconnect axes (R style)
    axes[2].spines["bottom"].set_position(("axes", -0.05))
    axes[2].yaxis.set_ticks_position("left")
    axes[2].spines["left"].set_position(("axes", -0.05))
    axes[2].set_ylim(ylim_p)
    axes[2].set_xlim([0, SFREQ_S / 2])

    fig.tight_layout(**tight_layout_cfg)
    return axes


def plot_for_layer_5(
    psd_mean_raw_vivo,
    psd_mean_raw_sili,
    psd_mean_raw_b,
    psd_mean_prep_vivo,
    psd_mean_prep_sili,
    psd_mean_prep_b,
    ci_raw_vivo,
    ci_raw_sili,
    ci_raw_b,
    ci_prep_vivo,
    ci_prep_sili,
    ci_prep_b,
    freq_raw_v,
    freq_raw_s,
    freq_raw_b,
    freq_prep_v,
    freq_prep_s,
    freq_prep_b,
    out_raw_vivo,
    out_raw_sili,
    out_raw_b,
    site_layers_vivo,
    site_layers_sili,
    ylim_r,
    ylim_p,
):
    """plot power spectrum density for the
    specified layer

    Args:
        layer (str): layer "L1", "L2_3", "L4", "L5", "L6"
        site_layers_sili (list): layers of each in silico sites
        site_layers_vivo (list): layers of each in vivo sites
        out_raw_vivo (dict): _description_
        out_raw_sili (dict): _description_
    """

    N_MAJOR_TICKS = 4
    SIZE = 3
    ALPHA = 0.7
    pm = {
        "linestyle": "none",
        "marker": "o",
        "markersize": SIZE,
        "rasterized": True,
    }
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)

    # Raw
    axes[0].plot(freq_raw_v, psd_mean_raw_vivo, color=COLOR_VIVO, label="vivo", **pm)
    axes[0].plot(freq_raw_b, psd_mean_raw_b, color=COLOR_BUCCI, label="buccino", **pm)
    axes[0].plot(freq_raw_s, psd_mean_raw_sili, color=COLOR_SILI, label="sili", **pm)
    # confidence interval
    axes[0].fill_between(
        freq_raw_v,
        (psd_mean_raw_vivo - ci_raw_vivo),
        (psd_mean_raw_vivo + ci_raw_vivo),
        color=COLOR_VIVO,
        linewidth=0,
        alpha=ALPHA,
        rasterized=True,
    )
    axes[0].fill_between(
        freq_raw_b,
        (psd_mean_raw_b - ci_raw_b),
        (psd_mean_raw_b + ci_raw_b),
        color=COLOR_BUCCI,
        linewidth=0,
        alpha=ALPHA,
        rasterized=True,
    )
    axes[0].fill_between(
        freq_raw_s,
        (psd_mean_raw_sili - ci_raw_sili),
        (psd_mean_raw_sili + ci_raw_sili),
        color=COLOR_SILI,
        linewidth=0,
        alpha=ALPHA,
        rasterized=True,
    )

    # legend
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_xlabel("")
    axes[0].set_xscale("log")
    axes[0].spines[["right", "top"]].set_visible(False)
    axes[0].set_ylim(ylim_r)
    axes[0].set_xlim([-1, SFREQ_S / 2])
    axes[0].tick_params(axis="x", which="minor", colors="black")
    axes[0].tick_params(axis="x", which="major", colors="black")

    # show minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
        numticks=N_MINOR_TICKS,
    )
    axes[0].tick_params(which="both")
    axes[0].xaxis.set_major_locator(locmaj)
    axes[0].xaxis.set_minor_locator(locmin)
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # disconnect axes (R style)
    axes[0].spines["bottom"].set_position(("axes", -0.05))
    axes[0].yaxis.set_ticks_position("left")
    axes[0].spines["left"].set_position(("axes", -0.05))
    axes[0].set_ylim(ylim_r)
    axes[0].set_xlim([0, SFREQ_S / 2])

    # Fit panel ************************************************************

    # get sites in layer
    out_raw_vivo2 = copy.copy(out_raw_vivo)
    out_raw_sili2 = copy.copy(out_raw_sili)
    out_raw_vivo2["power"] = out_raw_vivo2["power"][site_layers_vivo == "L5", :]
    out_raw_sili2["power"] = out_raw_sili2["power"][site_layers_sili == "L5", :]

    # plot
    plot_fits_all_for_layer_5(axes[1], out_raw_vivo2, out_raw_sili2, out_raw_b, pm)

    # Preprocessing ************************************************************

    axes[2].plot(
        freq_prep_v,
        psd_mean_prep_vivo,
        linestyle="none",
        marker="o",
        markersize=SIZE,
        color=COLOR_VIVO,
        label="vivo",
        rasterized=True,  # cheaper
    )
    axes[2].plot(
        freq_prep_b,
        psd_mean_prep_b,
        color=COLOR_BUCCI,
        label="Buccino",
        marker="o",
        markersize=SIZE,
        rasterized=True,  # cheaper
    )
    axes[2].plot(
        freq_prep_s,
        psd_mean_prep_sili,
        color=COLOR_SILI,
        label="silico",
        marker="o",
        markersize=SIZE,
        rasterized=True,  # cheaper
    )

    # confidence interval
    axes[2].fill_between(
        freq_prep_v,
        (psd_mean_prep_vivo - ci_prep_vivo),
        (psd_mean_prep_vivo + ci_prep_vivo),
        color=COLOR_VIVO,
        linewidth=0,
        alpha=ALPHA,
        rasterized=True,
    )
    axes[2].fill_between(
        freq_prep_b,
        (psd_mean_prep_b - ci_prep_b),
        (psd_mean_prep_b + ci_prep_b),
        color=COLOR_BUCCI,
        linewidth=0,
        alpha=ALPHA,
        rasterized=True,
    )
    axes[2].fill_between(
        freq_prep_s,
        (psd_mean_prep_sili - ci_prep_sili),
        (psd_mean_prep_sili + ci_prep_sili),
        color=COLOR_SILI,
        linewidth=0,
        alpha=ALPHA,
        rasterized=True,
    )

    # legend
    # axes[1].set_xlabel("Frequency (Hz)")
    axes[2].set_xlabel("")
    axes[2].set_xscale("log")
    axes[2].spines[["right", "top"]].set_visible(False)
    axes[2].set_ylim(ylim_p)

    axes[2].set_xlim([-1, SFREQ_S / 2])
    axes[2].tick_params(axis="x", which="minor", colors="black")
    axes[2].tick_params(axis="x", which="major", colors="black")

    # show minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
        numticks=N_MINOR_TICKS,
    )
    axes[2].tick_params(which="both")
    axes[2].xaxis.set_major_locator(locmaj)
    axes[2].xaxis.set_minor_locator(locmin)
    axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # disconnect axes (R style)
    axes[2].spines["bottom"].set_position(("axes", -0.05))
    axes[2].yaxis.set_ticks_position("left")
    axes[2].spines["left"].set_position(("axes", -0.05))
    axes[2].set_ylim(ylim_p)
    axes[2].set_xlim([0, SFREQ_S / 2])

    fig.tight_layout(**tight_layout_cfg)

    return axes


def eval_freq_scaling(powers: dict, freq_range=tuple):

    # count sites
    n_sites = powers["power"].shape[0]

    log_powers = []
    alphas = []
    intercepts = []

    for s_i in range(n_sites):

        # get frequencies with non-inf log power
        # and for slow freq 0 to 20 Hz [Bedard, 2006] with
        # alpha = 1
        log_freq = np.array([np.log10(powers["freq"])]).T
        good_ix = np.where(
            (np.log10(powers["freq"]) != -np.inf)
            & (powers["freq"] > freq_range[0])
            & (powers["freq"] <= freq_range[1])
        )[0]
        log_freq = log_freq[good_ix]

        # get log power
        log_power = np.log10(powers["power"][s_i, :])[good_ix]

        # calculate scaling factor (coef) alpha
        reg = LinearRegression().fit(log_freq, log_power)

        # store
        alphas.append(reg.coef_[0])
        intercepts.append(reg.intercept_)
        log_powers.append(log_power)

    # report stats
    #print("mean alpha:", np.mean(alphas))
    #print("std alpha:", np.std(alphas))
    #print("n=", n_sites)
    return alphas, log_powers, log_freq, intercepts


def get_log_freq_and_powers(out_raw_vivo: dict):
    """get log frequencies and powers

    Args:
        out_raw_vivo (dict): _description_

    Returns:
        _type_: _description_
    """
    # count sites
    n_sites = out_raw_vivo["power"].shape[0]
    log_powers = []

    for s_i in range(n_sites):

        # get frequencies with non-inf log power
        log_freq = np.array([np.log10(out_raw_vivo["freq"])]).T
        good_ix = np.where(np.log10(out_raw_vivo["freq"]) != -np.inf)[0]
        log_freq = log_freq[good_ix]

        # get log power
        log_power = np.log10(out_raw_vivo["power"][s_i, :])[good_ix]

        # store
        log_powers.append(log_power)

    # report stats
    return log_powers, log_freq


def plot_fits_all(axis, psd:dict, sf:int, color_hv:tuple, pm: dict, pm_fit_lfp:dict, pm_fit_spiking:dict):
    """_summary_

    Two bands:
        lfp band: 0 - 90 Hz
        spiking band: 300 Hz - 6000 Hz (or Nyquist?)

    Args:
        axis (_type_): _description_
        psd (_type_): _description_
        pm (dict): _description_
    """
    
    # plot data
    log_powers_all, log_freq_all = get_log_freq_and_powers(psd)
    axis.plot(
        10**log_freq_all,
        10 ** np.array(log_powers_all).mean(axis=0),
        color=color_hv,
        **pm
    )

    # plot fit LFP band (0 - 90 Hz)
    alpha_lfp, _, log_freq, interc = eval_freq_scaling(psd, freq_range=(0, 90))
    axis.plot(
        10**log_freq,
        10 ** (log_freq * np.mean(alpha_lfp) + np.mean(interc)),
        **pm_fit_lfp,
    )

    # plot fit spiking band (300 Hz - 6000 Hz (Nyquist))
    alpha_spiking, _, log_freq, interc = eval_freq_scaling(
        psd, freq_range=(300, 6000)
    )
    axis.plot(
        10**log_freq,
        10 ** (log_freq * np.mean(alpha_spiking) + np.mean(interc)),
        **pm_fit_spiking
    )
    return axis, alpha_lfp, alpha_spiking


def plot_log_slope(raw, freq_band, offset, plot: bool):

    
    COLOR_HV = [0.75, 0.75, 0.75] # light
    COLOR_MV = [0.4, 0.4, 0.4]
    COLOR_HS = [0.9, 0.64, 0.65] # light
    COLOR_MS = [0.9, 0.14, 0.15]    
    
    # get psd power and frequencies
    log_powers, log_freq = get_log_freq_and_powers(raw)

    # get slope
    alphas, _, _, intrcpt = eval_freq_scaling(raw, freq_range=freq_band)
    # average power
    log_power_mean = np.array(log_powers).mean(axis=0)
    log_freq = log_freq.squeeze()

    if plot:
        _, axis = plt.subplots(1)
        axis.plot(
            10**log_freq,
            10 ** np.array(log_powers).mean(axis=0),
            color=COLOR_HV,
            # **pm
        )
        if offset is "first_power":
            offset = log_power_mean[0]
        elif offset is "intercept":
            offset = np.mean(intrcpt)
        
        pink_noise = 10 ** (log_freq * np.mean(alphas) + offset)
        plt.plot(10**log_freq, pink_noise, "r")
        axis.set_xscale("log")
        axis.set_yscale("log")
    return alphas, log_freq, log_power_mean, offset


def plot_whitened_psd(
    axis, log_freq, alphas, data, offset, norm: bool, color: tuple, pm: dict
):

    # whiten psd (subtract power law fit)
    # - remove pink noise
    pink_noise_spectrum = 10 ** (log_freq * np.mean(alphas) + offset)
    unpinked = 10**data - pink_noise_spectrum

    # case normalize
    if norm:
        unpinked /= sum(unpinked)

    # plot
    axis.plot(10**log_freq, unpinked, color=color, **pm)
    return axis


def plot_whitened(
    axis, psd, freq_band, offset: str, norm: bool, color, pm: dict, plot_fit: bool
):
    """_summary_

    Args:
        axis (_type_): _description_
        psd (_type_): _description_
        freq_band (_type_): _description_
        offset (str):
        - "intercept": power <- 10**(slope * log_freq + intercept)
        - "first_power": power <-  10**(slope * log_freq + power[0])
        norm (bool):
        - True: divide by total power
        pm (dict): _description_
        plot_fit (bool): _description_

    Returns:
        _type_: _description_
    """
    # get fit
    alphas, log_freq, log_power_mean, offset = plot_log_slope(
        psd, freq_band, offset, plot=plot_fit
    )
    
    # plot whitened
    if offset == "first_power":
        offset = log_power_mean[0]
    elif offset == "intercept":
        offset = offset
    
    plot_whitened_psd(axis, log_freq, alphas, log_power_mean, offset, norm, color, pm)
    
    # format data
    whitening_data = {
        "alpha": alphas,
        "log_freq": log_freq,
        "log_power_mean": log_power_mean,
    }
    return axis, whitening_data


def get_power_snr(power, fq, sp_cutoff, sp_cutoff_up, lfp_cutoff):

    # power stats for spiking band
    mean_power = power[:, np.where((fq >= sp_cutoff) & (fq <= sp_cutoff_up))[0]].mean()
    ci_power = utils.conf_interv95(power[:, np.where((fq >= sp_cutoff) & (fq <= sp_cutoff_up))[0]])

    # power stats for lfp band
    mean_power_hv_lfp = power[:, np.where(fq <= lfp_cutoff)[0]].mean()
    ci_power_hv_lfp = utils.conf_interv95(power[:, np.where(fq <= lfp_cutoff)[0]])

    # snr
    print(f"SNR: {mean_power / mean_power_hv_lfp}")
    return mean_power, ci_power, mean_power_hv_lfp, ci_power_hv_lfp


def get_snr_df(psd, sp_cutoff, sp_cutoff_up, lfp_cutoff, exp):
    """build dataframe with powers per site and experiment
    """
    df = pd.DataFrame()
    df["power"] = psd["power"][:, np.where((psd["freq"] >= sp_cutoff) & (psd["freq"] <= sp_cutoff_up))[0]].mean(axis=1) / psd[
        "power"
    ][:, np.where(psd["freq"] <= lfp_cutoff)[0]].mean(axis=1)
    df["Experiment"] = exp
    return df


def get_spiking_power(
    psd: dict,
    sp_cutoff: float=300,
    sp_cutoff_up: float=6000,
    exp: str="MS",
    layer: str="L1"
    ):
    """build dataframe of median power
    of spiking activity between 300 and 6000 Hz
    for each site, experiment and layer
    """
    df = pd.DataFrame()
    df["power"] = np.median(psd["power"][:, np.where((psd["freq"] >= sp_cutoff) & (psd["freq"] <= sp_cutoff_up))[0]], axis=1)
    df["Layer"] = layer
    df["Experiment"] = exp
    return df


def get_psd_data_prepro(layer, hv, hs, nv, ns, ne, sites_hv, sites_hs, sites_nv, sites_ns, sites_ne, norm=True):
    
    # return data structure
    d = dict()
    
    # horvath vivo (probe 1)
    d["psd_pre_hv_"] = copy.copy(hv)
    d["psd_pre_hv_"]["power"] = hv["power"][sites_hv == layer, :]

    # biophy
    d["psd_pre_hs_"] = copy.copy(hs)
    d["psd_pre_hs_"]["power"] = hs["power"][sites_hs == layer, :]

    # neuropixels
    # vivo
    d["psd_pre_nv_"] = copy.copy(nv)
    d["psd_pre_nv_"]["power"] = nv["power"][sites_nv == layer, :]

    # biophy spont
    d["psd_pre_ns_"] = copy.copy(ns)
    d["psd_pre_ns_"]["power"] = ns["power"][sites_ns == layer, :]

    # biophy evoked
    d["psd_pre_ne_"] = copy.copy(ne)
    d["psd_pre_ne_"]["power"] = ne["power"][sites_ne == layer, :]

    # (11s) Divide by total power ***********************
    if norm:
        d["psd_pre_hv_"]["power"] /= d["psd_pre_hv_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_hs_"]["power"] /= d["psd_pre_hs_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_nv_"]["power"] /= d["psd_pre_nv_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_ns_"]["power"] /= d["psd_pre_ns_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_ne_"]["power"] /= d["psd_pre_ne_"]["power"].sum(axis=1)[:, None]

    # (11s) Median over sites ***********************

    # horvath
    # in vivo
    d["mean_hv"] = np.median(d["psd_pre_hv_"]["power"], axis=0)
    # biophy
    d["mean_hs"] = np.median(d["psd_pre_hs_"]["power"], axis=0)

    # neuropixels
    # pre
    d["mean_nv"] = np.median(d["psd_pre_nv_"]["power"], axis=0)
    d["mean_ns"] = np.median(d["psd_pre_ns_"]["power"], axis=0)
    d["mean_ne"] = np.median(d["psd_pre_ne_"]["power"], axis=0)

    # Calculate 95% confidence intervals  ******************

    # horvath
    # vivo
    n_samples = d["psd_pre_nv_"]["power"].shape[0]
    d["ci_hv"] = 1.96 * np.std(d["psd_pre_nv_"]["power"], axis=0) / np.sqrt(n_samples)

    # biophy.
    n_samples = d["psd_pre_hs_"]["power"].shape[0]
    d["ci_hs"] = 1.96 * np.std(d["psd_pre_hs_"]["power"], axis=0) / np.sqrt(n_samples)

    # neuropixels
    # vivo
    n_samples = d["psd_pre_nv_"]["power"].shape[0]
    d["ci_nv"] = 1.96 * np.std(d["psd_pre_nv_"]["power"], axis=0) / np.sqrt(n_samples)
    # biophy. spont.
    n_samples = d["psd_pre_ns_"]["power"].shape[0]
    d["ci_ns"] = 1.96 * np.std(d["psd_pre_ns_"]["power"], axis=0) / np.sqrt(n_samples)
    # biophy. evoked
    n_samples = d["psd_pre_ne_"]["power"].shape[0]
    d["ci_ne"] = 1.96 * np.std(d["psd_pre_ne_"]["power"], axis=0) / np.sqrt(n_samples)
    return d


def get_psd_data_prepro_dense(layer, hv, hs, sites_hv, sites_hs, norm=True):
    
    # return data structure
    d = dict()
    
    # horvath vivo (probe 1)
    d["psd_pre_hv_"] = copy.copy(hv)
    d["psd_pre_hv_"]["power"] = hv["power"][sites_hv == layer, :]

    # biophy
    d["psd_pre_hs_"] = copy.copy(hs)
    d["psd_pre_hs_"]["power"] = hs["power"][sites_hs == layer, :]

    # (11s) Divide by total power ***********************
    if norm:
        d["psd_pre_hv_"]["power"] /= d["psd_pre_hv_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_hs_"]["power"] /= d["psd_pre_hs_"]["power"].sum(axis=1)[:, None]

    # (11s) Median over sites ***********************

    # horvath
    # in vivo
    d["mean_hv"] = np.median(d["psd_pre_hv_"]["power"], axis=0)
    # biophy
    d["mean_hs"] = np.median(d["psd_pre_hs_"]["power"], axis=0)

    # Calculate 95% confidence intervals  ******************

    # horvath
    # vivo
    n_samples = d["psd_pre_hv_"]["power"].shape[0]
    d["ci_hv"] = 1.96 * np.std(d["psd_pre_hv_"]["power"], axis=0) / np.sqrt(n_samples)

    # biophy.
    n_samples = d["psd_pre_hs_"]["power"].shape[0]
    d["ci_hs"] = 1.96 * np.std(d["psd_pre_hs_"]["power"], axis=0) / np.sqrt(n_samples)
    return d


def get_psd_data_prepro_demo(layer, ns, ne, sites_ns, sites_ne, norm=True):
    
    # return data structure
    d = dict()
    
    # biophy spont
    d["psd_pre_ns_"] = copy.copy(ns)
    d["psd_pre_ns_"]["power"] = ns["power"][sites_ns == layer, :]

    # biophy evoked
    d["psd_pre_ne_"] = copy.copy(ne)
    d["psd_pre_ne_"]["power"] = ne["power"][sites_ne == layer, :]

    # (11s) Divide by total power ***********************
    if norm:
        d["psd_pre_ns_"]["power"] /= d["psd_pre_ns_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_ne_"]["power"] /= d["psd_pre_ne_"]["power"].sum(axis=1)[:, None]

    # (11s) Median over sites ***********************

    d["mean_ns"] = np.median(d["psd_pre_ns_"]["power"], axis=0)
    d["mean_ne"] = np.median(d["psd_pre_ne_"]["power"], axis=0)

    # Calculate 95% confidence intervals  ******************

    # biophy. spont.
    n_samples = d["psd_pre_ns_"]["power"].shape[0]
    d["ci_ns"] = 1.96 * np.std(d["psd_pre_ns_"]["power"], axis=0) / np.sqrt(n_samples)
    # biophy. evoked
    n_samples = d["psd_pre_ne_"]["power"].shape[0]
    d["ci_ne"] = 1.96 * np.std(d["psd_pre_ne_"]["power"], axis=0) / np.sqrt(n_samples)
    return d


def get_psd_data_prepro_layer_5(layer, hv, hs, nv, ns, ne, nb, sites_hv, sites_hs, sites_nv, sites_ns, sites_ne, norm=True):

    # return data structure
    d = dict()
    
    # horvath vivo (probe 1)
    d["psd_pre_hv_"] = copy.copy(hv)
    d["psd_pre_hv_"]["power"] = hv["power"][sites_hv == layer, :]

    # biophy
    d["psd_pre_hs_"] = copy.copy(hs)
    d["psd_pre_hs_"]["power"] = hs["power"][sites_hs == layer, :]

    # neuropixels
    # vivo
    d["psd_pre_nv_"] = copy.copy(nv)
    d["psd_pre_nv_"]["power"] = nv["power"][sites_nv == layer, :]

    # biophy spont
    d["psd_pre_ns_"] = copy.copy(ns)
    d["psd_pre_ns_"]["power"] = ns["power"][sites_ns == layer, :]

    # biophy evoked
    d["psd_pre_ne_"] = copy.copy(ne)
    d["psd_pre_ne_"]["power"] = ne["power"][sites_ne == layer, :]

    # synthetic buccino
    d["psd_pre_nb_"] = copy.copy(ne)
    d["psd_pre_nb_"]["power"] = nb["power"]
    
    # (11s) Divide by total power ***********************
    if norm:
        d["psd_pre_hv_"]["power"] /= d["psd_pre_hv_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_hs_"]["power"] /= d["psd_pre_hs_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_nv_"]["power"] /= d["psd_pre_nv_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_ns_"]["power"] /= d["psd_pre_ns_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_ne_"]["power"] /= d["psd_pre_ne_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_nb_"]["power"] /= d["psd_pre_nb_"]["power"].sum(axis=1)[:, None]

    # (11s) Average over sites ***********************

    # horvath
    # in vivo
    d["mean_hv"] = np.mean(d["psd_pre_hv_"]["power"], axis=0)
    # biophy
    d["mean_hs"] = np.mean(d["psd_pre_hs_"]["power"], axis=0)

    # neuropixels
    # pre
    d["mean_nv"] = np.mean(d["psd_pre_nv_"]["power"], axis=0)
    d["mean_ns"] = np.mean(d["psd_pre_ns_"]["power"], axis=0)
    d["mean_ne"] = np.mean(d["psd_pre_ne_"]["power"], axis=0)
    d["mean_nb"] = np.mean(d["psd_pre_nb_"]["power"], axis=0)

    # Calculate 95% confidence intervals  ******************

    # horvath
    # vivo
    n_samples = d["psd_pre_nv_"]["power"].shape[0]
    d["ci_hv"] = 1.96 * np.std(d["psd_pre_nv_"]["power"], axis=0) / np.sqrt(n_samples)

    # biophy.
    n_samples = d["psd_pre_hs_"]["power"].shape[0]
    d["ci_hs"] = 1.96 * np.std(d["psd_pre_hs_"]["power"], axis=0) / np.sqrt(n_samples)

    # neuropixels
    # vivo
    n_samples = d["psd_pre_nv_"]["power"].shape[0]
    d["ci_nv"] = 1.96 * np.std(d["psd_pre_nv_"]["power"], axis=0) / np.sqrt(n_samples)
    # biophy. spont.
    n_samples = d["psd_pre_ns_"]["power"].shape[0]
    d["ci_ns"] = 1.96 * np.std(d["psd_pre_ns_"]["power"], axis=0) / np.sqrt(n_samples)
    # biophy. evoked
    n_samples = d["psd_pre_ne_"]["power"].shape[0]
    d["ci_ne"] = 1.96 * np.std(d["psd_pre_ne_"]["power"], axis=0) / np.sqrt(n_samples)
    # synthetic Buccino
    n_samples = d["psd_pre_nb_"]["power"].shape[0]
    d["ci_nb"] = 1.96 * np.std(d["psd_pre_nb_"]["power"], axis=0) / np.sqrt(n_samples)
    return d


def get_psd_data_prepro_dense_layer_5(layer, hv, hs, sites_hv, sites_hs, norm=True):

    # return data structure
    d = dict()
    
    # horvath vivo (probe 1)
    d["psd_pre_hv_"] = copy.copy(hv)
    d["psd_pre_hv_"]["power"] = hv["power"][sites_hv == layer, :]

    # biophy
    d["psd_pre_hs_"] = copy.copy(hs)
    d["psd_pre_hs_"]["power"] = hs["power"][sites_hs == layer, :]

    # (11s) Divide by total power ***********************
    if norm:
        d["psd_pre_hv_"]["power"] /= d["psd_pre_hv_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_hs_"]["power"] /= d["psd_pre_hs_"]["power"].sum(axis=1)[:, None]

    # (11s) Average over sites ***********************

    # horvath
    # in vivo
    d["mean_hv"] = np.mean(d["psd_pre_hv_"]["power"], axis=0)
    # biophy
    d["mean_hs"] = np.mean(d["psd_pre_hs_"]["power"], axis=0)

    # Calculate 95% confidence intervals  ******************

    # horvath
    # vivo
    n_samples = d["psd_pre_hv_"]["power"].shape[0]
    d["ci_hv"] = 1.96 * np.std(d["psd_pre_hv_"]["power"], axis=0) / np.sqrt(n_samples)

    # biophy.
    n_samples = d["psd_pre_hs_"]["power"].shape[0]
    d["ci_hs"] = 1.96 * np.std(d["psd_pre_hs_"]["power"], axis=0) / np.sqrt(n_samples)
    return d


def get_psd_data_prepro_layer_5_demo(layer, ns, ne, sites_ns, sites_ne, norm=True):

    # return data structure
    d = dict()
    
    # biophy spont
    d["psd_pre_ns_"] = copy.copy(ns)
    d["psd_pre_ns_"]["power"] = ns["power"][sites_ns == layer, :]

    # biophy evoked
    d["psd_pre_ne_"] = copy.copy(ne)
    d["psd_pre_ne_"]["power"] = ne["power"][sites_ne == layer, :]

    # (11s) Divide by total power ***********************
    if norm:
        d["psd_pre_ns_"]["power"] /= d["psd_pre_ns_"]["power"].sum(axis=1)[:, None]
        d["psd_pre_ne_"]["power"] /= d["psd_pre_ne_"]["power"].sum(axis=1)[:, None]

    # (11s) Average over sites ***********************

    # pre
    d["mean_ns"] = np.mean(d["psd_pre_ns_"]["power"], axis=0)
    d["mean_ne"] = np.mean(d["psd_pre_ne_"]["power"], axis=0)

    # Calculate 95% confidence intervals  ******************

    # biophy. spont.
    n_samples = d["psd_pre_ns_"]["power"].shape[0]
    d["ci_ns"] = 1.96 * np.std(d["psd_pre_ns_"]["power"], axis=0) / np.sqrt(n_samples)
    # biophy. evoked
    n_samples = d["psd_pre_ne_"]["power"].shape[0]
    d["ci_ne"] = 1.96 * np.std(d["psd_pre_ne_"]["power"], axis=0) / np.sqrt(n_samples)
    return d


def plot_power_law_fits(ax, d, prms, cl, pm, pm_fit1, pm_fit2):
    """_summary_

    Args:
        ax (axis): _description_
        d (dict): power spectral density data dictionary
        prms (dict): experiment frequencies
        cl (dict): experiment colors
        pm (dict): _description_
        pm_fit1 (dict): _description_
        pm_fit2 (dict): _description_

    Returns:
        _type_: _description_
    """
    # return data structure
    dd = dict()
    
    # Fitting ************************************************************

    ax, dd["alphas_lfp_hv"], dd["alphas_spiking_hv"] = plot_fits_all(
        ax, d["psd_pre_hv_"], prms["SFREQ_HV"], cl["COLOR_HV"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_nv"], dd["alphas_spiking_nv"] = plot_fits_all(
        ax, d["psd_pre_nv_"], prms["SFREQ_NV"], cl["COLOR_NV"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_ns"], dd["alphas_spiking_ns"] = plot_fits_all(
        ax, d["psd_pre_ns_"], prms["SFREQ_NS"], cl["COLOR_NS"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_ne"], dd["alphas_spiking_ne"] = plot_fits_all(
        ax, d["psd_pre_ne_"], prms["SFREQ_NE"], cl["COLOR_NE"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_hs"], dd["alphas_spiking_hs"] = plot_fits_all(
        ax, d["psd_pre_hs_"], prms["SFREQ_HS"], cl["COLOR_HS"], pm, pm_fit1, pm_fit2
    )
    
    # axes legend
    # esthetics
    ax.set_xscale("log")
    ax.set_yscale("log")
    # minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=5)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.2, 0.4, 0.6, 0.8),
        numticks=5,
    )
    ax.tick_params(which="both")
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))

    # report slopes
    # lfp band
    print("\nLFP band")
    print(
        f"""hv: \u03B1={np.round(np.mean(dd["alphas_lfp_hv"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_hv"]),1)}"""
    )
    print(
        f"""nv: \u03B1={np.round(np.mean(dd["alphas_lfp_nv"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_nv"]),1)}"""
    )
    print(
        f"""ns: \u03B1={np.round(np.mean(dd["alphas_lfp_ns"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_ns"]),1)}"""
    )
    print(
        f"""ne: \u03B1={np.round(np.mean(dd["alphas_lfp_ne"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_ne"]),1)}"""
    )
    print(
        f"""hs: \u03B1={np.round(np.mean(dd["alphas_lfp_hs"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_hs"]),1)}"""
    )

    # spiking band
    print("\nSpiking band")
    print(
        f"""hv: \u03B1={np.round(np.mean(dd["alphas_spiking_hv"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_hv"]),1)}"""
    )
    print(
        f"""nv: \u03B1={np.round(np.mean(dd["alphas_spiking_nv"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_nv"]),1)}"""
    )
    print(
        f"""ns: \u03B1={np.round(np.mean(dd["alphas_spiking_ns"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_ns"]),1)}"""
    )
    print(
        f"""ne: \u03B1={np.round(np.mean(dd["alphas_spiking_ne"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_ne"]),1)}"""
    )
    print(
        f"""hs: \u03B1={np.round(np.mean(dd["alphas_spiking_hs"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_hs"]),1)}"""
    )

    # Power SNR **************************

    # print("\nPower SNR")

    # o_hv_l1 = get_power_snr(
    #     d["psd_pre_hv_"]["power"], d["psd_pre_hv_"]["freq"], 300, 6000, 90
    # )
    # o_nv_l1 = get_power_snr(
    #     d["psd_pre_nv_"]["power"], d["psd_pre_nv_"]["freq"], 300, 6000, 90
    # )
    # o_ns_l1 = get_power_snr(
    #     d["psd_pre_ns_"]["power"], d["psd_pre_ns_"]["freq"], 300, 6000, 90
    # )
    # o_ne_l1 = get_power_snr(
    #     d["psd_pre_ne_"]["power"], d["psd_pre_ne_"]["freq"], 300, 6000, 90
    # )
    # o_hs_l1 = get_power_snr(
    #     d["psd_pre_hs_"]["power"], d["psd_pre_hs_"]["freq"], 300, 6000, 90
    # )
    return ax, dd


def plot_power_law_fits_demo(ax, d, prms, cl, pm, pm_fit1, pm_fit2):
    """_summary_

    Args:
        ax (axis): _description_
        d (dict): power spectral density data dictionary
        prms (dict): experiment frequencies
        cl (dict): experiment colors
        pm (dict): _description_
        pm_fit1 (dict): _description_
        pm_fit2 (dict): _description_

    Returns:
        _type_: _description_
    """
    # return data structure
    dd = dict()
    
    # Fitting ************************************************************

    ax, dd["alphas_lfp_ns"], dd["alphas_spiking_ns"] = plot_fits_all(
        ax, d["psd_pre_ns_"], prms["SFREQ_NS"], cl["COLOR_NS"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_ne"], dd["alphas_spiking_ne"] = plot_fits_all(
        ax, d["psd_pre_ne_"], prms["SFREQ_NE"], cl["COLOR_NE"], pm, pm_fit1, pm_fit2
    )
    
    # axes legend
    # esthetics
    ax.set_xscale("log")
    ax.set_yscale("log")
    # minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=5)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.2, 0.4, 0.6, 0.8),
        numticks=5,
    )
    ax.tick_params(which="both")
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))

    # report slopes
    # lfp band
    print("\nLFP band")
    print(
        f"""ns: \u03B1={np.round(np.mean(dd["alphas_lfp_ns"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_ns"]),1)}"""
    )
    print(
        f"""ne: \u03B1={np.round(np.mean(dd["alphas_lfp_ne"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_ne"]),1)}"""
    )

    # spiking band
    print("\nSpiking band")
    print(
        f"""ns: \u03B1={np.round(np.mean(dd["alphas_spiking_ns"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_ns"]),1)}"""
    )
    print(
        f"""ne: \u03B1={np.round(np.mean(dd["alphas_spiking_ne"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_ne"]),1)}"""
    )    
    return ax, dd


def plot_power_law_fits_layer_5(ax, d, prms, cl, pm, pm_fit1, pm_fit2):
    """_summary_

    Args:
        ax (axis): _description_
        d (dict): power spectral density data dictionary
        prms (dict): experiment frequencies
        cl (dict): experiment colors
        pm (dict): _description_
        pm_fit1 (dict): _description_
        pm_fit2 (dict): _description_

    Returns:
        _type_: _description_
    """
    # return data structure
    dd = dict()
    
    # Fitting ************************************************************

    ax, dd["alphas_lfp_hv"], dd["alphas_spiking_hv"] = plot_fits_all(
        ax, d["psd_pre_hv_"], prms["SFREQ_HV"], cl["COLOR_HV"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_nv"], dd["alphas_spiking_nv"] = plot_fits_all(
        ax, d["psd_pre_nv_"], prms["SFREQ_NV"], cl["COLOR_NV"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_ns"], dd["alphas_spiking_ns"] = plot_fits_all(
        ax, d["psd_pre_ns_"], prms["SFREQ_NS"], cl["COLOR_NS"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_ne"], dd["alphas_spiking_ne"] = plot_fits_all(
        ax, d["psd_pre_ne_"], prms["SFREQ_NE"], cl["COLOR_NE"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_hs"], dd["alphas_spiking_hs"] = plot_fits_all(
        ax, d["psd_pre_hs_"], prms["SFREQ_HS"], cl["COLOR_HS"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_nb"], dd["alphas_spiking_nb"] = plot_fits_all(
        ax, d["psd_pre_nb_"], prms["SFREQ_NB"], cl["COLOR_NB"], pm, pm_fit1, pm_fit2
    )
    
    # axes legend
    # esthetics
    ax.set_xscale("log")
    ax.set_yscale("log")
    # minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=5)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.2, 0.4, 0.6, 0.8),
        numticks=5,
    )
    ax.tick_params(which="both")
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))

    # report slopes
    # lfp band
    print("\nLFP band")
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_lfp_hv"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_hv"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_lfp_nv"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_nv"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_lfp_ns"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_ns"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_lfp_ne"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_ne"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_lfp_hs"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_hs"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_lfp_nb"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_nb"]),1)}"""
    )    

    # spiking band
    print("\nSpiking band")
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_spiking_hv"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_hv"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_spiking_nv"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_nv"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_spiking_ns"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_ns"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_spiking_ne"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_ne"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_spiking_hs"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_hs"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_spiking_nb"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_nb"]),1)}"""
    )

    # Power SNR **************************

    print("\nPower SNR")

    o_hv_l1 = get_power_snr(
        d["psd_pre_hv_"]["power"], d["psd_pre_hv_"]["freq"], 300, 6000, 90
    )
    o_nv_l1 = get_power_snr(
        d["psd_pre_nv_"]["power"], d["psd_pre_nv_"]["freq"], 300, 6000, 90
    )
    o_ns_l1 = get_power_snr(
        d["psd_pre_ns_"]["power"], d["psd_pre_ns_"]["freq"], 300, 6000, 90
    )
    o_ne_l1 = get_power_snr(
        d["psd_pre_ne_"]["power"], d["psd_pre_ne_"]["freq"], 300, 6000, 90
    )
    o_hs_l1 = get_power_snr(
        d["psd_pre_hs_"]["power"], d["psd_pre_hs_"]["freq"], 300, 6000, 90
    )
    o_nb_l1 = get_power_snr(
        d["psd_pre_nb_"]["power"], d["psd_pre_nb_"]["freq"], 300, 6000, 90
    )    
    return ax, dd



def plot_power_law_fits_layer_5_demo(ax, d, prms, cl, pm, pm_fit1, pm_fit2):
    """_summary_

    Args:
        ax (axis): _description_
        d (dict): power spectral density data dictionary
        prms (dict): experiment frequencies
        cl (dict): experiment colors
        pm (dict): _description_
        pm_fit1 (dict): _description_
        pm_fit2 (dict): _description_

    Returns:
        _type_: _description_
    """
    # return data structure
    dd = dict()
    
    # Fitting ************************************************************

    ax, dd["alphas_lfp_ns"], dd["alphas_spiking_ns"] = plot_fits_all(
        ax, d["psd_pre_ns_"], prms["SFREQ_NS"], cl["COLOR_NS"], pm, pm_fit1, pm_fit2
    )
    ax, dd["alphas_lfp_ne"], dd["alphas_spiking_ne"] = plot_fits_all(
        ax, d["psd_pre_ne_"], prms["SFREQ_NE"], cl["COLOR_NE"], pm, pm_fit1, pm_fit2
    )
    
    # axes legend
    # esthetics
    ax.set_xscale("log")
    ax.set_yscale("log")
    # minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=5)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.2, 0.4, 0.6, 0.8),
        numticks=5,
    )
    ax.tick_params(which="both")
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))

    # report slopes
    # lfp band
    print("\nLFP band")
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_lfp_ns"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_ns"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_lfp_ne"]),1)}\u00B1{np.round(np.std(dd["alphas_lfp_ne"]),1)}"""
    )

    # spiking band
    print("\nSpiking band")
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_spiking_ns"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_ns"]),1)}"""
    )
    print(
        f"""\u03B1={np.round(np.mean(dd["alphas_spiking_ne"]),1)}\u00B1{np.round(np.std(dd["alphas_spiking_ne"]),1)}"""
    )

    # Power SNR **************************

    print("\nPower SNR")

    o_ns_l1 = get_power_snr(
        d["psd_pre_ns_"]["power"], d["psd_pre_ns_"]["freq"], 300, 6000, 90
    )
    o_ne_l1 = get_power_snr(
        d["psd_pre_ne_"]["power"], d["psd_pre_ne_"]["freq"], 300, 6000, 90
    )
    return ax, dd


def plot_lfp_freq_scaling_stats(ax, dd, cl):
    
    # create lfp scaling plot data ***************
    
    # neuropixels
    df1 = pd.DataFrame()
    df1["Alpha"] = np.absolute(dd["alphas_lfp_nv"])
    df1["Experiment"] = "MS"  # NPX marques-smith
    df2 = pd.DataFrame()
    df2["Alpha"] = np.absolute(dd["alphas_lfp_ns"])
    df2["Experiment"] = "NS"  # NPX biophy. spont.
    df3 = pd.DataFrame()
    df3["Alpha"] = np.absolute(dd["alphas_lfp_ne"])
    df3["Experiment"] = "NE"  # NPX biophy. evoked
    # denser probe
    df4 = pd.DataFrame()
    df4["Alpha"] = np.absolute(dd["alphas_lfp_hv"])
    df4["Experiment"] = "DH"  # dense horvath
    df5 = pd.DataFrame()
    df5["Alpha"] = np.absolute(dd["alphas_lfp_hs"])
    df5["Experiment"] = "DS"  # dense biophy spont.
    plot_data_lfp = pd.concat([df1, df2, df3, df4, df5])

    # plot lfp band scaling
    ax = sns.boxplot(
        ax=ax,
        data=plot_data_lfp,
        x="Experiment",
        y="Alpha",
        hue="Experiment",
        notch=True,
        hue_order=[
            "MS",
            "NS",
            "NE",
            "DH",
            "DS",
        ],
        palette=[
            cl["COLOR_NV"],
            cl["COLOR_NS"],
            cl["COLOR_NE"],
            cl["COLOR_HV"],
            cl["COLOR_HS"],
        ],
        width=0.8,
        fliersize=3,
        flierprops={"marker": ".", "markerfacecolor": "k"},
        linewidth=0.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))


def plot_spiking_freq_scaling_stats(ax, dd, cl):
    
    # neuropixels
    df1 = pd.DataFrame()
    df1["Alpha"] = np.absolute(dd["alphas_spiking_nv"])
    df1["Experiment"] = "MS"  # NPX marques-smith
    df2 = pd.DataFrame()
    df2["Alpha"] = np.absolute(dd["alphas_spiking_ns"])
    df2["Experiment"] = "NS"  # NPX biophy. spont.
    df3 = pd.DataFrame()
    df3["Alpha"] = np.absolute(dd["alphas_spiking_ne"])
    df3["Experiment"] = "NE"  # NPX biophy. evoked
    # denser probe
    df4 = pd.DataFrame()
    df4["Alpha"] = np.absolute(dd["alphas_spiking_hv"])
    df4["Experiment"] = "DH"  # dense horvath
    df5 = pd.DataFrame()
    df5["Alpha"] = np.absolute(dd["alphas_spiking_hs"])
    df5["Experiment"] = "DS"  # dense biophy spont.
    plot_data_spik = pd.concat([df1, df2, df3, df4, df5])

    # plot spiking band scaling
    ax = sns.boxplot(
        ax=ax,
        data=plot_data_spik,
        x="Experiment",
        y="Alpha",
        hue="Experiment",
        notch=True,
        hue_order=[
            "MS",
            "NS",
            "NE",
            "DH",
            "DS",
        ],
        palette=[cl["COLOR_NV"], cl["COLOR_NS"], cl["COLOR_NE"], cl["COLOR_HV"], cl["COLOR_HS"]],
        width=0.8,
        fliersize=3,
        flierprops={"marker": ".", "markerfacecolor": "k"},
        linewidth=0.5,
        medianprops={"color": cl["COLOR_MEDIAN"], "linewidth": 2},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.set_ylabel("Slope of power law fit (\u03B1)") 
    

def plot_lfp_freq_scaling_stats_layer_5(ax, dd, cl):
    
    # create lfp scaling plot data ***************
    
    # neuropixels
    df1 = pd.DataFrame()
    df1["Alpha"] = np.absolute(dd["alphas_lfp_nv"])
    df1["Experiment"] = "MS"  # NPX marques-smith
    df2 = pd.DataFrame()
    df2["Alpha"] = np.absolute(dd["alphas_lfp_ns"])
    df2["Experiment"] = "NS"  # NPX biophy. spont.
    df3 = pd.DataFrame()
    df3["Alpha"] = np.absolute(dd["alphas_lfp_ne"])
    df3["Experiment"] = "NE"  # NPX biophy. evoked
    # denser probe
    df4 = pd.DataFrame()
    df4["Alpha"] = np.absolute(dd["alphas_lfp_hv"])
    df4["Experiment"] = "DH"  # dense horvath
    df5 = pd.DataFrame()
    df5["Alpha"] = np.absolute(dd["alphas_lfp_hs"])
    df5["Experiment"] = "DS"  # dense biophy spont.
    df6 = pd.DataFrame()
    df6["Alpha"] = np.absolute(dd["alphas_lfp_nb"])
    df6["Experiment"] = "NB"  # dense biophy spont.    
    plot_data_lfp = pd.concat([df1, df2, df3, df4, df5, df6])

    # plot lfp band scaling
    ax = sns.boxplot(
        ax=ax,
        data=plot_data_lfp,
        x="Experiment",
        y="Alpha",
        hue="Experiment",
        notch=True,
        hue_order=[
            "MS",
            "NS",
            "NE",
            "DH",
            "DS",
            "NB",
        ],
        palette=[
            cl["COLOR_NV"],
            cl["COLOR_NS"],
            cl["COLOR_NE"],
            cl["COLOR_HV"],
            cl["COLOR_HS"],
            cl["COLOR_NB"],
        ],
        width=0.8,
        fliersize=3,
        flierprops={"marker": ".", "markerfacecolor": "k"},
        linewidth=0.5,
        medianprops={"color": cl["COLOR_MEDIAN"], "linewidth": 2},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))


def plot_spiking_freq_scaling_stats_layer_5(ax, dd, cl):
    
    # neuropixels
    df1 = pd.DataFrame()
    df1["Alpha"] = np.absolute(dd["alphas_spiking_nv"])
    df1["Experiment"] = "MS"  # NPX marques-smith
    df2 = pd.DataFrame()
    df2["Alpha"] = np.absolute(dd["alphas_spiking_ns"])
    df2["Experiment"] = "NS"  # NPX biophy. spont.
    df3 = pd.DataFrame()
    df3["Alpha"] = np.absolute(dd["alphas_spiking_ne"])
    df3["Experiment"] = "NE"  # NPX biophy. evoked
    # denser probe
    df4 = pd.DataFrame()
    df4["Alpha"] = np.absolute(dd["alphas_spiking_hv"])
    df4["Experiment"] = "DH"  # dense horvath
    df5 = pd.DataFrame()
    df5["Alpha"] = np.absolute(dd["alphas_spiking_hs"])
    df5["Experiment"] = "DS"  # dense biophy spont.
    df6 = pd.DataFrame()
    df6["Alpha"] = np.absolute(dd["alphas_spiking_nb"])
    df6["Experiment"] = "NB"  # dense biophy spont.    
    plot_data_spik = pd.concat([df1, df2, df3, df4, df5, df6])

    # plot spiking band scaling
    ax = sns.boxplot(
        ax=ax,
        data=plot_data_spik,
        x="Experiment",
        y="Alpha",
        hue="Experiment",
        notch=True,
        hue_order=[
            "MS",
            "NS",
            "NE",
            "DH",
            "DS",
            "NB",
        ],
        palette=[cl["COLOR_NV"], cl["COLOR_NS"], cl["COLOR_NE"], cl["COLOR_HV"], cl["COLOR_HS"], cl["COLOR_NB"]],
        width=0.8,
        fliersize=3,
        flierprops={"marker": ".", "markerfacecolor": "k"},
        linewidth=0.5,
        medianprops={"color": cl["COLOR_MEDIAN"], "linewidth": 2},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.set_ylabel("Slope of power law fit (\u03B1)")


def plot_power_snr_stats(ax, d: dict, cl: dict):
    """calculate and plot spiking activity 
    power-to-lfp-noise ratio

    Args:
        ax (_type_): _description_
        d (dict): _description_
        cl (dict): _description_

    Returns:
        _type_: _description_
    """

    # calculate spiking activity power-to-lfp-noise ratio
    # spiking band is defined between 300 and 6000 Hz
    df_nv = get_snr_df(d["psd_pre_nv_"], 300, 6000, 90, "MS")
    df_ns = get_snr_df(d["psd_pre_ns_"], 300, 6000, 90, "NS")
    df_ne = get_snr_df(d["psd_pre_ne_"], 300, 6000, 90, "NE")
    df_hv = get_snr_df(d["psd_pre_hv_"], 300, 6000, 90, "DH")
    df_hs = get_snr_df(d["psd_pre_hs_"], 300, 6000, 90, "DS")
    
    # stack
    plot_data = pd.concat([df_nv, df_ns, df_ne, df_hv, df_hs])

    # plot
    ax = sns.boxplot(
        ax=ax,
        data=plot_data,
        x="Experiment",
        y="power",
        hue="Experiment",
        notch=True,
        hue_order=[
            "MS",
            "NS",
            "NE",
            "DH",
            "DS",
        ],
        palette=[cl["COLOR_NV"], cl["COLOR_NS"], cl["COLOR_NE"], cl["COLOR_HV"], cl["COLOR_HS"]],
        width=0.8,
        fliersize=1.1,
        flierprops={"marker": ".", "markerfacecolor": "k"},
        linewidth=0.5,
        medianprops={"color": cl["COLOR_MEDIAN"], "linewidth": 2},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)
    ax.set_ylabel("Power SNR (ratio)")
    ax.set_yscale("log")

    # minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=5)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.2, 0.4, 0.6, 0.8),
        numticks=5,
    )
    ax.tick_params(which="both")
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.legend("", frameon=False)
    return ax


def plot_power_snr_stats_layer_5(ax, d:dict, cl:dict):

    # spiking band is defined between 300 and 6000 Hz
    df_nv = to_df(d["psd_pre_nv_"], 300, 6000, 90, "MS")
    df_ns = to_df(d["psd_pre_ns_"], 300, 6000, 90, "NS")
    df_ne = to_df(d["psd_pre_ne_"], 300, 6000, 90, "NE")
    df_hv = to_df(d["psd_pre_hv_"], 300, 6000, 90, "DH")
    df_hs = to_df(d["psd_pre_hs_"], 300, 6000, 90, "DS")
    df_nb = to_df(d["psd_pre_nb_"], 300, 6000, 90, "NB")
    
    # stack
    plot_data = pd.concat([df_nv, df_ns, df_ne, df_hv, df_hs, df_nb])

    # plot
    ax = sns.boxplot(
        ax=ax,
        data=plot_data,
        x="Experiment",
        y="power",
        hue="Experiment",
        notch=True,
        hue_order=[
            "MS",
            "NS",
            "NE",
            "DH",
            "DS",
            "NB",
        ],
        palette=[cl["COLOR_NV"], cl["COLOR_NS"], cl["COLOR_NE"], cl["COLOR_HV"], cl["COLOR_HS"], cl["COLOR_NB"]],
        width=0.8,
        fliersize=1.1,
        flierprops={"marker": ".", "markerfacecolor": "k"},
        linewidth=0.5,
        medianprops={"color": cl["COLOR_MEDIAN"], "linewidth": 2},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)
    ax.set_ylabel("Power SNR (ratio)")
    ax.set_yscale("log")

    # minor ticks
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=5)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.2, 0.4, 0.6, 0.8),
        numticks=5,
    )
    ax.tick_params(which="both")
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.legend("", frameon=False)
    return ax
