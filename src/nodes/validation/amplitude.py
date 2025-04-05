"""Metrics of trace amplitude node

Returns:
    _type_: _description_
"""

# import libs
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface as si
import spikeinterface.preprocessing as spre
import shutil 
import matplotlib
from scipy.stats import mannwhitneyu # stats
from scipy.stats import kruskal # stats
import scikit_posthocs as sp

N_MAJOR_TICKS = 4
N_MINOR_TICKS = 12


def preprocess_reyes(raw_path, prepro_path, freq_min=300, freq_max=6000):
    """preprocess reyes

    Args:
        raw_path (_type_): _description_
        freq_min (int, optional): _description_. Defaults to 300.
        freq_max (int, optional): _description_. Defaults to 6000.

    Returns:
        _type_: _description_
    """
    trace = se.read_mcsraw(raw_path)
    prepro = spre.bandpass_filter(trace, freq_min=freq_min, freq_max=freq_max)
    prepro = spre.common_reference(prepro, reference='global', operator='median')
    print("is filtered:", prepro.is_filtered())
    
    # rewrite
    shutil.rmtree(prepro_path, ignore_errors=True)
    prepro.save(folder=prepro_path, format="binary")
    return prepro


def preprocess_horvath(raw_path, prepro_path, freq_min=300, freq_max=9999):
    """_summary_

    Args:
        raw_path (_type_): _description_
        prepro_path (_type_): _description_
        freq_min (int, optional): _description_. Defaults to 300.
        freq_max (int, optional): _description_. Defaults to 9999.

    Returns:
        _type_: _description_
    """
    trace = se.NwbRecordingExtractor(raw_path)
    prepro = spre.bandpass_filter(trace, freq_min=freq_min, freq_max=freq_max)
    prepro = spre.common_reference(prepro, reference='global', operator='median')
    print("is filtered:", prepro.is_filtered())
    
    # rewrite
    shutil.rmtree(prepro_path, ignore_errors=True)
    prepro.save(folder=prepro_path, format="binary")
    return prepro


def preprocess_silico(raw_path, prepro_path, freq_min=300, freq_max=4999):
    """_summary_

    Args:
        raw_path (_type_): _description_
        prepro_path (_type_): _description_
        freq_min (int, optional): _description_. Defaults to 300.
        freq_max (int, optional): _description_. Defaults to 4999.

    Returns:
        _type_: _description_
    """
    trace = si.load_extractor(raw_path)
    prepro = spre.bandpass_filter(trace, freq_min=freq_min, freq_max=freq_max)
    prepro = spre.common_reference(prepro, reference='global', operator='median')
    print("is filtered:", prepro.is_filtered())

    # rewrite
    shutil.rmtree(prepro_path, ignore_errors=True)
    prepro.save(folder=prepro_path, format="binary")
    return prepro


def plot_dist_stats(
        axis, trace, n_sites: int, color: list, ci_color: list, label: str
        ):
    """plot amplitude mean distributions with confidence intervals

    Args:
        trace (_type_): preprocessed trace with spikeinterface
        n_sites (_type_): _description_
        color (_type_): _description_
        ci_color (_type_): _description_
        label (_type_): _description_
    """
    # set parameters
    contact_ids = np.arange(0, n_sites,1)
    step = 0.1
    bins = np.arange(-1,1.1,step)

    # plot    
    counts_all = []
    for c_i in contact_ids:
        
        # trace
        reyes_trace = trace.get_traces()[:, c_i]

        # normalize
        norm_reyes_trace = reyes_trace / np.max(np.abs(reyes_trace))

        # histogram
        counts, _ = np.histogram(norm_reyes_trace, bins=bins)
        counts_all.append(counts)

    # mean
    mean_count = np.array(counts_all).mean(axis=0)
    axis.plot(bins[:-1]+step/2, mean_count, color=color, label=label);

    # confidence interval
    ci = 1.96 * np.std(counts_all, axis=0) / np.sqrt(len(counts_all[0]))
    axis.fill_between(bins[:-1]+step/2, (mean_count-ci), (mean_count+ci), color=ci_color, linewidth=1, alpha=0.2)


def compute_proba_dist_stats(trace, contact_ids: np.array, abs: bool = True):
    """plot amplitude mean distributions with confidence intervals

    Args:
        trace (_type_): preprocessed trace with spikeinterface
        n_contacts (_type_): _description_
        color (_type_): _description_
        ci_color (_type_): _description_
        label (_type_): _description_
    """
    # set parameters
    step = 0.1
    bins = np.arange(-1, 1.1, step)
    if abs:
        bins = np.arange(0, 1.1, step)

    # calculate average + ci of probabilities density
    # over contacts
    proba_all = []
    for c_i in contact_ids:
        trace_i = trace.get_traces()[:, c_i]
        # get absolute amplitude
        if abs:
            trace_i = np.absolute(trace_i)
        norm_trace = trace_i / np.max(np.abs(trace_i))
        counts, _ = np.histogram(norm_trace, bins=bins)
        proba = counts / np.sum(counts)
        proba_all.append(proba)

    # return stats
    dist_mean = np.array(proba_all).mean(axis=0)
    dist_ci = 1.96 * np.std(proba_all, axis=0) / np.sqrt(len(proba_all[0]))
    return dist_mean, dist_ci, bins


def compute_anr_old(trace, contact_ids: np.array):
    """compute signal-to-noise ratios for each recording site

    Args:
        trace (_type_): preprocessed trace with spikeinterface
        n_contacts (_type_): number of contacts
        color (_type_): color for plot
        ci_color (_type_): color for confidence interval area
        label (_type_): legend

    Returns:
        list
    """
    norm_trace_all = []
    for c_i in contact_ids:
        trace_i = trace.get_traces()[:, c_i]
        mad = pd.DataFrame(trace_i).mad().values
        norm_trace = trace_i / mad
        norm_trace_all.append(norm_trace)
    return norm_trace_all


def compute_snr(traces: np.array, cols: list):
    """compute signal-to-noise ratios for each recording site

    Args:
        traces (np.array): timepoints x sites preprocessed traces
        cols (list): trace columns for which to compute snr

    Returns:
        np.ndarray
    """
    snrs = []
    for col in cols:
        mad = pd.DataFrame(traces[:, col]).mad().values
        norm_trace = traces[:, col] / mad
        snrs.append(norm_trace)
    return np.array(snrs)


def get_snr_pdfs(norm_traces: np.ndarray, bins):
    """calculate amplitude-to-noise ratio distributions,
    their median and 95% confidence interval
    
    Args:
        norm_traces (np.ndarray): _description_
        bins (_type_): _description_
        data (list(array)): pdf by site

    Returns:
        dist_mean (np.array): 1-D array of mean snr pdf over sites
        dist_ci:  95% confidence interval of snr pdf over sites
        data (dict[list]):
        - key: "pdf_by_site"
        - value: list of 1-D arrays. One list entry per site.
    """

    # calculate average + ci of probabilities density
    # over contacts
    # calculate mean absolute deviation mad and divide
    # amplitude by mad
    proba_all = []
    for c_i in range(len(norm_traces)):
        counts, bins = np.histogram(norm_traces[c_i], bins=bins)
        proba = counts / np.sum(counts)
        proba_all.append(proba)

    # return stats
    dist_mean = np.median(np.array(proba_all), axis=0)
    #dist_mean = np.array(proba_all).mean(axis=0)
    dist_ci = 1.96 * np.std(proba_all, axis=0) / np.sqrt(len(proba_all[0]))
    
    # store sites' data
    data = {"pdf_by_site": proba_all}
    return dist_mean, dist_ci, data


def plot_snr_pdf_all(
    axis, mean_v, mean_s, mean_e, ci_v, ci_s, ci_e, bins, color_v:tuple, color_s:tuple, color_e:tuple, pm: dict
):
    """Plot distribution of voltage trace signal-to-noise ratio
    with mean and 95% confidence intervals for the in vivo data and the biophysical 
    model
    
    Args:
    
    Returns
    """

    # vivo
    plot_proba_dist_stats(
        axis,
        mean_v[mean_v > 0],
        ci_v[mean_v > 0],
        bins[:-1][mean_v > 0], # plot non-zero because of logscale
        color=color_v,
        ci_color=color_v,
        label="vivo",
        pm=pm
    )
    # silico
    plot_proba_dist_stats(
        axis,
        mean_s[mean_s > 0],
        ci_s[mean_s > 0],
        bins[:-1][mean_s > 0],
        color=color_s,
        ci_color=color_s,
        label="silico",
        pm=pm
    )
    # evoked
    if len(mean_e) > 1:
        plot_proba_dist_stats(
            axis,
            mean_e[mean_e > 0],
            ci_e[mean_e > 0],
            bins[:-1][mean_e > 0],
            color=color_e,
            ci_color=color_e,
            label="evoked",
            pm=pm
        )
    
    # legend
    axis.set_yscale("log")
    axis.spines[["right", "top"]].set_visible(False)
    axis.tick_params(which="both")

    # show minor ticks
    axis.tick_params(which="major")
    # y
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.5, 1),
        numticks=2,
    )
    axis.yaxis.set_major_locator(locmaj)
    axis.yaxis.set_minor_locator(locmin)
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # disconnect axes (R style)
    axis.spines["bottom"].set_position(("axes", -0.05))
    axis.yaxis.set_ticks_position("left")
    axis.spines["left"].set_position(("axes", -0.05))
    # square axis
    #axis.set_box_aspect(1)
    return axis


def plot_anr_pdf_l5(
    axis, mean_v, mean_s, mean_e, mean_b, ci_v, ci_s, ci_e, ci_b, bins, color_v:tuple, color_s:tuple, color_e:tuple, color_b:tuple , pm: dict
):
    """Plot distribution of voltage trace signal-to-noise ratio
    with mean and 95% confidence intervals for the in vivo data and the biophysical 
    model
    
    Args:
    
    Returns
    """

    # vivo
    plot_proba_dist_stats(
        axis,
        mean_v[mean_v > 0],
        ci_v[mean_v > 0],
        bins[:-1][mean_v > 0],
        color=color_v,
        ci_color=color_v,
        label="vivo",
        pm=pm
    )
    # silico
    plot_proba_dist_stats(
        axis,
        mean_s[mean_s > 0],
        ci_s[mean_s > 0],
        bins[:-1][mean_s > 0],
        color=color_s,
        ci_color=color_s,
        label="silico",
        pm=pm
    )
    # evoked
    if len(mean_e) > 1:
        plot_proba_dist_stats(
            axis,
            mean_e[mean_e > 0],
            ci_e[mean_e > 0],
            bins[:-1][mean_e > 0],
            color=color_e,
            ci_color=color_e,
            label="evoked",
            pm=pm
        )
    # synthetic
    if len(mean_b) > 1:
        plot_proba_dist_stats(
            axis,
            mean_b[mean_b > 0],
            ci_b[mean_b > 0],
            bins[:-1][mean_b > 0],
            color=color_b,
            ci_color=color_b,
            label="synth.",
            pm=pm
        )   
         
    # legend
    axis.set_yscale("log")
    axis.spines[["right", "top"]].set_visible(False)
    axis.tick_params(which="both")

    # show minor ticks
    axis.tick_params(which="major")
    # y
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MAJOR_TICKS)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,
        subs=(0.5, 1),
        numticks=2,
    )
    axis.yaxis.set_major_locator(locmaj)
    axis.yaxis.set_minor_locator(locmin)
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # disconnect axes (R style)
    axis.spines["bottom"].set_position(("axes", -0.05))
    axis.yaxis.set_ticks_position("left")
    axis.spines["left"].set_position(("axes", -0.05))
    # square axis
    #axis.set_box_aspect(1)
    return axis

def plot_proba_dist_stats(
        axis, dist_mean, dist_ci, bins, color: list, ci_color: list, label: str, pm:dict
        ):
    """plot amplitude mean distributions with confidence intervals

    Args:
        color (_type_): _description_
        ci_color (_type_): _description_
        label (_type_): _description_
    """
    # set parameters
    step = 0.1

    # plot mean and ci
    axis.plot(
        bins+step/2, dist_mean, color=color, label=label, **pm
        )

    # correct for meaningless negative bottom ci lines
    # because this is the probability space
    # don't set to 0 but to a small number. 0 produces meaningless 
    # bottom CI toward -inf on a log scale
    # we set bottom ci to 0 because on the probability space (y-axis)
    # bottom ci <= 0 produce meaningless values toward -inf on a log scale
    # we set bottom ci to the mean
    
    # add confidence interval
    axis.fill_between(
        bins+step/2,
        (dist_mean),
        (dist_mean + dist_ci),
        color=ci_color,
        linewidth=0.1,        
        alpha=0.4,
        rasterized=True
        )
    
    
def count_sites(df, exp, layer):
    return len(df[(df["experiment"] == exp) & (df["layer"] == layer)])


def get_amplitude(df, exp, layer):
    return df[(df["experiment"] == exp) & (df["layer"] == layer)]["amplitude"].values


def get_mwu(df, exp1, exp2, layer):
    """Perform the Mann-Whitney U rank test on two independent samples
    Args:
        layer: "L1", "L2/3", "L4", "L5", "L6")
    """
    z, p = mannwhitneyu(
        get_amplitude(df, exp1, layer),
        get_amplitude(df, exp2, layer),
        method="exact",
    )
    print(
        f"""1 vs. 2, z={z}, p={p}, N_1={count_sites(df, exp1, layer)}, N_2={count_sites(df, exp2, layer)}"""
    )


def get_kk(df, exp):
    """kruskall wallis test
    """
    h, p = kruskal(
        get_amplitude(df, exp, "L1"),
        get_amplitude(df, exp, "L2/3"),
        get_amplitude(df, exp, "L4"),
        get_amplitude(df, exp, "L5"),
        get_amplitude(df, exp, "L6"),
    )
    print(f"H={h}, p={p}")
    print(f"""N_L1 = {count_sites(df, exp, "L1")} sites""")
    print(f"""N_L23 = {count_sites(df, exp, "L2/3")} sites""")
    print(f"""N_L4 = {count_sites(df, exp, "L4")} sites""")
    print(f"""N_L5 = {count_sites(df, exp, "L5")} sites""")
    print(f"""N_L6 = {count_sites(df, exp, "L6")} sites""")


def get_kk_demo(df, exp):
    """kruskall wallis test
    """
    h, p = kruskal(
        get_amplitude(df, exp, "L5"),
        get_amplitude(df, exp, "L6"),
    )
    print(f"H={h}, p={p}")
    print(f"""N_L5 = {count_sites(df, exp, "L5")} sites""")
    print(f"""N_L6 = {count_sites(df, exp, "L6")} sites""")

    
def get_posthoc_dunn_holm_sidak(plot_data, exp):
    """posthoc test after kruskall wallis with Dunn and holm_sidak
    multiple comparison correction of p-values

    Args:
        plot_data (_type_): _description_
        exp (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = [
        get_amplitude(plot_data, exp, "L1"),
        get_amplitude(plot_data, exp, "L2/3"),
        get_amplitude(plot_data, exp, "L4"),
        get_amplitude(plot_data, exp, "L5"),
        get_amplitude(plot_data, exp, "L6"),
    ]
    # holm sidak method has more power than Bonferroni which is more conservative
    # Non-significance can indicate subtle differences, power issues, samll sample size,
    # or the balancing be due to how the Holm-Sidak correction controls Type I errors
    # while retaining power.
    # we can still look at the p-values to identify trends.
    df = sp.posthoc_dunn(data, p_adjust="holm-sidak")
    df.columns = ["L1", "L2/3", "L4", "L5", "L6"]
    df.index = ["L1", "L2/3", "L4", "L5", "L6"]
    return df


def get_posthoc_dunn_holm_sidak_demo(plot_data, exp):
    """posthoc test after kruskall wallis with Dunn and holm_sidak
    multiple comparison correction of p-values

    Args:
        plot_data (_type_): _description_
        exp (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = [
        get_amplitude(plot_data, exp, "L5"),
        get_amplitude(plot_data, exp, "L6"),
    ]
    # holm sidak method has more power than Bonferroni which is more conservative
    # Non-significance can indicate subtle differences, power issues, samll sample size,
    # or the balancing be due to how the Holm-Sidak correction controls Type I errors
    # while retaining power.
    # we can still look at the p-values to identify trends.
    df = sp.posthoc_dunn(data, p_adjust="holm-sidak")
    df.columns = [ "L5", "L6"]
    df.index = ["L5", "L6"]
    return df