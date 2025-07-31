"""_summary_

Returns:
    _type_: _description_

Requirements
    - Sorting Extractors property attribute contains postprocessing metadata ("firing rates")
"""

import numpy as np
from numpy.linalg import norm as lalgnorm
import spikeinterface.qualitymetrics as qm
from src.nodes.postpro import waveform
import spikeinterface as si
#import spikeinterface.postprocessing as spost
from src.nodes.utils import from_dict_to_df
from scipy.stats import norm, poisson
import spikeinterface.core.template_tools as ttools


def _euclidean_distance(coord_1, coord_2):
    return np.sqrt(np.sum((coord_1 - coord_2) ** 2))


def get_true_units_SNRs(
    rec_folder: str, study_folder: str, MS_BEFORE: float, MS_AFTER: float
):
    """measure ground truth units' signal-to-noise ratios

    Args:
        rec_folder (str): _description_
        study_folder (str): _description_
        MS_BEFORE (float): _description_
        MS_AFTER (float): _description_
    
    Prerequisites:
     - have extracted ground truth waveforms
    """
    Recording = si.load_extractor(rec_folder)
    we = waveform.load(Recording, study_folder, ms_before=MS_BEFORE, ms_after=MS_AFTER)
    return qm.compute_snrs(we)


def get_spatial_spread(
    We, unit_id: int, max_chids: dict, channel_ids, channel_coord
):
    """measure unit's spatial spread

    Args:
        unit_id (int): _description_
        max_chids (dict): _description_
        Recording (_type_): _description_
        channel_ids (_type_): _description_
        channel_coord (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get waveforms
    wv, _ = We.get_waveforms(unit_id=unit_id, with_index=True)    

    # get the site ids (sparse)
    c_ids = We.sparsity.unit_id_to_channel_ids[unit_id]

    # get the nearest site
    max_chid = max_chids[unit_id]
    max_chid_ix = np.where(c_ids == max_chid)[0][0]
    
    # get the average spike on each site
    mean_spikes = wv.mean(axis=0)
    max_spike = mean_spikes[:, max_chid_ix]
    
    # measure the average spike's similarity 
    # of each site to the average spike
    # of the nearest site to the unit
    # (ratio between 0 and 1)
    n_sites = mean_spikes.shape[1]
    cosim_weights = []
    for s_i in range(n_sites):
        cosim_weights.append(
            np.dot(max_spike, mean_spikes[:, s_i])
            / (lalgnorm(max_spike) * lalgnorm(mean_spikes[:, s_i]))
        )
    cosim_weights = np.array(cosim_weights)

    # threshold the similarity metric to be positive
    # we only look at similarity (not inverse similarity (<0))
    cosim_weights[cosim_weights < 0] = 0

    # measure the distance of the site to the nearest site
    # to the unit
    channel_coord = channel_coord[np.isin(channel_ids, c_ids), :]
    try:
        max_chids_coord = channel_coord[max_chid_ix, :]
    except:
        from ipdb import set_trace; set_trace()
        
    dist = []
    for ix, _ in enumerate(c_ids):
        dist.append(_euclidean_distance(max_chids_coord, channel_coord[ix]))
    dist = np.array(dist)

    # return spatial spread
    return {
        "spatial_spread": np.dot(cosim_weights, dist),
        "channel_distance": dist,
        "weights": cosim_weights,
    }


def get_spatial_spread_all_units(
    recording_path: str, study_path: str, ms_before=3, ms_after=3, peak_sign="neg"
):
    """get all units' spatial extent metrics

    Args:
        recording_path (str): Path of the Recording Extractor
        study_path (str): _description_
        ms_before (float): _description_
        ms_after (float): _description_

    Returns:
        (dict): spatial spread of each unit
        - key: true unit id
        - value: spatial spread
    
    Prerequisites:
        - have extracted ground truth waveforms
    """
    # takes 1:30 min

    # get Waveform extractor
    Recording = si.load_extractor(recording_path)
    
    # WvfExtractor = waveform.load(
    #     Recording, study_path, ms_before=ms_before, ms_after=ms_after
    # )
    We = si.WaveformExtractor.load_from_folder(study_path)
    
    # get sites' distance to the max site
    # get 3D coordinates
    #Recording = si.load_extractor(recording_path)
    #channel_ids = Recording.get_channel_ids()
    Rec = si.load_extractor(recording_path)
    channel_coord = Rec.get_probe().contact_positions
    channel_ids = Rec.get_channel_ids()

    # get channels where spike amplitude is maximal
    max_chids = ttools.get_template_extremum_channel(
        We, peak_sign=peak_sign)

    # takes 1:30 min (1310 units)
    spatial_spread = dict()
    for _, unit in enumerate(We.unit_ids):
        spatial_spread[unit] = get_spatial_spread(
            We, unit, max_chids, channel_ids, channel_coord
        )["spatial_spread"]
    return spatial_spread


def get_true_unit_true_distance_to_nearest_site(SortingTrue, recording_path):
    """_summary_

    Args:
        SortingTrue (_type_): _description_

    Returns:
        _type_: _description_
    """
    # calculate each unit's euclidean distance to nearest site
    # - get site coordinates
    Recording = si.load_extractor(recording_path)
    Probe = Recording.get_probe()
    sites_coord = Probe.contact_positions

    # - calculate distance
    unit_ids = SortingTrue.unit_ids
    distance_to_site = []
    for unit in unit_ids:
        x = SortingTrue.get_property("x", ids=[unit])
        y = SortingTrue.get_property("y", ids=[unit])
        z = SortingTrue.get_property("z", ids=[unit])
        # store shortest distance between this unit and all sites
        site_dist = []
        for site in sites_coord:
            site_dist.append(_euclidean_distance(np.array([x, y, z]).T, site))
        distance_to_site.append(np.min(np.array(site_dist)))
    return distance_to_site


def get_distance_for(selected_units: list, SortingTrue, RECORDING_PATH: str):
    """get selected units' true distances to their nearest recording site

    Args:
        selected_units (list): _description_
        SortingTrue (_type_): _description_
        RECORDING_PATH (str): _description_

    Returns:
        _type_: _description_
    """

    true_unit_ids = SortingTrue.get_unit_ids()

    # feature: get unit distance to nearest site
    distances = np.array(
        get_true_unit_true_distance_to_nearest_site(SortingTrue, RECORDING_PATH)
    )

    missed_ix = np.searchsorted(true_unit_ids, selected_units, "left")
    return distances[missed_ix]


def get_firing_rates_for(selected_units: list, SortingTrue):
    """get selected units' firing rates

    Args:
        selected_units (list): _description_
        SortingTrue (_type_): _description_

    Returns:
        _type_: _description_
    """
    true_unit_ids = SortingTrue.get_unit_ids()
    firing_rates = SortingTrue.get_property("firing_rates")
    selected_units = np.searchsorted(true_unit_ids, selected_units, "left")
    return firing_rates[selected_units]


def get_layer_for(selected_units: list, SortingTrue):
    """get selected units' layer

    Args:
        selected_units (list): selected true unit ids
        SortingTrue (_type_): _description_

    Returns:
        _type_: _description_
    """
    true_unit_ids = SortingTrue.get_unit_ids()
    layer = SortingTrue.get_property("layer")
    selected_units = np.searchsorted(true_unit_ids, selected_units, "left")
    return layer[selected_units]


def get_synapse_class_for(selected_units: list, SortingTrue):
    """get selected units' synapse class

    Args:
        selected_units (list): selected true unit ids
        SortingTrue (_type_): _description_

    Returns:
        _type_: _description_
    """
    true_unit_ids = SortingTrue.get_unit_ids()
    synapse_class = SortingTrue.get_property("synapse_class")
    selected_units = np.searchsorted(true_unit_ids, selected_units, "left")
    return synapse_class[selected_units]


def get_etype_for(selected_units: list, SortingTrue):
    """get etypes of selected units

    Args:
        selected_units (list): selected true unit ids
        SortingTrue (_type_): _description_

    Returns:
        _type_: _description_
    """
    true_unit_ids = SortingTrue.get_unit_ids()
    etype = SortingTrue.get_property("etype")
    selected_units = np.searchsorted(true_unit_ids, selected_units, "left")
    return etype[selected_units]


def get_mtype_for(selected_units: list, SortingTrue):
    """get mtypes of selected units

    Args:
        selected_units (list): selected true unit ids
        SortingTrue (_type_): _description_

    Returns:
        _type_: _description_
    """
    true_unit_ids = SortingTrue.get_unit_ids()
    mtype = SortingTrue.get_property("mtype")
    selected_units = np.searchsorted(true_unit_ids, selected_units, "left")
    return mtype[selected_units]


def get_spike_count(SortingTrue):
    """Get spike count by cell

    Args:
        SortingTrue (_type_): SpikeInterface Sorting extractor
        Recording (_type_): SpikeInterface Recording extractor

    Returns:
        _type_: _description_
    """
    spike_count = SortingTrue.get_total_num_spikes()
    return from_dict_to_df(spike_count, columns=["cells", "total spike"])


def get_firing_rates(SortingTrue, Recording):
    """Calculate firing rate by cell

    Args:
        SortingTrue (_type_): SpikeInterface Sorting extractor
        Recording (_type_): SpikeInterface Recording extractor

    Returns:
        pd.DataFrame: _description_
    """
    total_spike_by_unit = SortingTrue.get_total_num_spikes()
    firing_rate_by_unit_df = from_dict_to_df(
        total_spike_by_unit, columns=["cells", "total spike"]
    )
    firing_rate_by_unit_df["firing rate"] = (
        firing_rate_by_unit_df["total spike"] / Recording.get_total_duration()
    )
    return firing_rate_by_unit_df.drop(columns=["total spike"])


def get_firing_rate(unit_id: int, Sorting, rec_duration: float):
    """get a unit's firing rate
    
    Returns:
        (float): firing rate in spikes/secs
    """
    n_spikes = Sorting.get_total_num_spikes()[unit_id]
    return n_spikes / rec_duration


def plot_firing_rate_hist_vs_lognorm(
        data_all: np.array, 
        log_x_min, 
        log_x_max, 
        nbins, 
        t_dec, 
        ax, 
        label, 
        color=(0.13, 0.23, 0.98), 
        markerfacecolor=(0.13, 0.23, 0.98), 
        markeredgecolor="w", 
        markeredgewidth=0.5, 
        linestyle="-",
        markersize=3, 
        dashes=(5,0), 
        legend=True, 
        lognormal=True
        ):

    """_summary_

    Returns:
        dict: _description_
        
    """
    p_pickup = lambda _freq: 1.0 - poisson(_freq * t_dec).cdf(0)

    x_bins = np.logspace(log_x_min, log_x_max, nbins)
    p_hist = p_pickup(x_bins[1:])
    p_all = p_pickup(data_all)

    H_all = np.histogram(data_all, bins=x_bins)[0] * p_hist

    # mean and standard deviation of log(x), which we expect to be distributed normally
    mn_all = np.sum(np.log10(data_all) * p_all) / np.sum(p_all)
    sd_all = np.sum(np.abs(np.log10(data_all) * p_all - mn_all)) / np.sum(p_all)

    # data points
    ax.plot(
        x_bins[1:], 
        H_all/H_all.sum(), 
        marker="o", 
        ls="none", 
        markersize=markersize, 
        label=label, 
        markerfacecolor=markerfacecolor, 
        markeredgecolor=markeredgecolor,
        markeredgewidth=markeredgewidth
        )

    # lognormal fit
    y_fit = norm(mn_all, sd_all).pdf(np.log10(x_bins[1:]))
    y_fit_all = H_all.sum() * y_fit / y_fit.sum()
    y_fit = []
    if lognormal:
        y_fit = y_fit_all/sum(y_fit_all)
        ax.plot(x_bins[1:], y_fit, color=color, linestyle=linestyle, dashes=dashes)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")

    # show minor ticks
    #ax.tick_params(which='both')
    #locmaj = matplotlib.ticker.LogLocator(base=10, numticks=3) 
    #ax.xaxis.set_major_locator(locmaj)    
    #locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=12)
    #ax.xaxis.set_minor_locator(locmin)
    #ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    #ax.set_yticks([0, 0.4, 0.8, 1])
    #ax.set_yticklabels([0, 0.4, 0.8, 1])

    # if legend:
    #     plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    #     plt.xlabel("spontaneous firing rate (Hz)")
    #     plt.ylabel("probability (ratio)")
    #ax.set_xticklabels([])
    
    return {
        "x_data": x_bins[1:],
        "y_data": H_all/H_all.sum(),
        "y_fit": y_fit_all/sum(y_fit_all),
        "mean": mn_all,
        "std": sd_all
            }


def mad(data):
    mean_data = np.mean(data)
    return np.mean(np.absolute(data - mean_data))

def get_mad_ratio(spike_amp, noise_amp):
    """calculate an sd_ratio robust to outliers

    Args:
        spike_amp (_type_): _description_
        noise_amp (_type_): _description_

    Returns:
        _type_: _description_
    """
    mad_unit = mad(spike_amp)  # twice smaller than std
    mad_noise = mad(noise_amp)
    return mad_unit / mad_noise