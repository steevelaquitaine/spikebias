
import os 
from matplotlib import pyplot as plt
import numpy as np
# import spikeforest as sf
import spikeinterface as si
from scipy.stats import norm, poisson
from src.nodes.load import load_campaign_params 
from src.nodes.truth.silico import ground_truth 
import matplotlib
import pandas as pd

# def plot_each_rec(uri, rec_i, plot=False):

#     # get recording
#     all_recordings = sf.load_spikeforest_recordings(uri)
#     study_name = all_recordings[rec_i].study_name
#     recording_name = all_recordings[rec_i].recording_name
#     # print(study_name)
#     # print(recording_name)
#     x = [
#         R for R in all_recordings
#         if R.study_name == study_name and R.recording_name == recording_name
#     ]
#     if len(x) == 0: raise Exception(f'Recording not found: {study_name}/{recording_name}')
#     R = x[0]

#     # load recording extractor
#     recording = R.get_recording_extractor()

#     # load ground truth sorting extractor
#     sorted_true = R.get_sorting_true_extractor()

#     # calculate firing rate
#     firing_rate = []
#     for unit_id in sorted_true.get_unit_ids():

#         # get spike train
#         st = sorted_true.get_unit_spike_train(unit_id=unit_id)

#         # calculate firing rate
#         firing_rate.append(len(st) / recording.get_total_duration())

#     print("unit count:", len(sorted_true.get_unit_ids()))
    
#     # plot distribution 
#     if plot:
#         _, axis = plt.subplots(1,1,figsize=(5,2))
#         axis.hist(firing_rate, bins=np.arange(0, 1.1*max(firing_rate), 0.1), width=0.2);
#         axis.set_xticks(np.arange(0, 1.1*max(firing_rate), 1));
#         axis.set_xlabel("firing rate (Hz)");
#         axis.set_ylabel("neuron (count)");
#         axis.spines[["right", "top"]].set_visible(False)
#     return sorted_true, firing_rate


# def plot_all_rec_monotrode(all_recordings, plot=False):
#     """plot histogram of firing rates over all recordings

#     Args:
#         all_recordings (_type_): _description_

#     Raises:
#         Exception: _description_

#     Returns:
#         _type_: _description_
#     """

#     # initialize firing rates
#     firing_rates = []

#     # get firing rate
#     for rec_i in range(len(all_recordings)):

#         # get recording info
#         study_name = all_recordings[rec_i].study_name
#         recording_name = all_recordings[rec_i].recording_name
#         x = [
#             R for R in all_recordings
#             if R.study_name == study_name and R.recording_name == recording_name
#         ]
#         if len(x) == 0: raise Exception(f'Recording not found: {study_name}/{recording_name}')
#         R = x[0]

#         # load recording extractor
#         recording = R.get_recording_extractor()

#         # load ground truth sorting extractor
#         sorted_true = R.get_sorting_true_extractor()

#         # calculate firing rate
#         for unit_id in sorted_true.get_unit_ids():

#             # get spike train
#             st = sorted_true.get_unit_spike_train(unit_id=unit_id)

#             # calculate firing rate
#             firing_rates += [len(st) / recording.get_total_duration()]

#     # plot distribution 
#     if plot:
#         _, axis = plt.subplots(1,1,figsize=(5,2))
#         axis.hist(firing_rates, bins=np.arange(0, 1.1*max(firing_rates), 0.1), width=0.2);
#         axis.set_xticks(np.arange(0, 1.1*max(firing_rates), 1));
#         axis.set_xlabel("firing rate (Hz)");
#         axis.set_ylabel("neuron (count)");
#         axis.spines[["right", "top"]].set_visible(False)
#     return sorted_true, firing_rates


# def compute_spike_rate_janelia(JANELIA_FR_FILE_PATH, save=False):

#     # compare firing rate histograms
#     uri = 'sha1://43298d72b2d0860ae45fc9b0864137a976cb76e8?hybrid-janelia-spikeforest-recordings.json'

#     # firing rate hist. population 1
#     _, fr_1 = plot_each_rec(uri, 0)

#     # firing rate hist. population 2
#     _, fr_2 = plot_each_rec(uri, 4)

#     # concatenate population into one
#     firing_rates_janelia = fr_1 + fr_2

#     # cast as array
#     firing_rates_janelia = np.array(firing_rates_janelia)

#     # save firing rates
#     if save:
#         parent_path = os.path.dirname(JANELIA_FR_FILE_PATH)
#         if not os.path.isdir(parent_path):
#             os.makedirs(parent_path)
#         np.save(JANELIA_FR_FILE_PATH, firing_rates_janelia)
#     return firing_rates_janelia


# def compute_spike_rate_monotrode(SYNTH_MONOTRODE_FR_FILE_PATH, save=False):

#     print("\nMonotrode")

#     # compare firing rate histograms
#     uri = 'sha1://3b265eced5640c146d24a3d39719409cceccc45b?synth-monotrode-spikeforest-recordings.json'
#     all_recordings = sf.load_spikeforest_recordings(uri)
#     _, firing_rates_synth_monotrode = plot_all_rec_monotrode(all_recordings)

#     # cast as array
#     firing_rates_synth_monotrode = np.array(firing_rates_synth_monotrode)

#     # save firing rates
#     if save:
#         parent_path = os.path.dirname(SYNTH_MONOTRODE_FR_FILE_PATH)
#         if not os.path.isdir(parent_path):
#             os.makedirs(parent_path)
#         np.save(SYNTH_MONOTRODE_FR_FILE_PATH, firing_rates_synth_monotrode)
#     return firing_rates_synth_monotrode


# def compute_spike_rate_buccino2020(BUCCI_RECORDING_PATH, BUCCI_GT_SORTING_PATH, BUCCI_FR_FILE_PATH, save=False):

#     print("\nBuccino_2020")

#     # load recording just to get its duration
#     recording = si.load_extractor(BUCCI_RECORDING_PATH)
#     sorted_true = si.load_extractor(BUCCI_GT_SORTING_PATH)
#     buccino_firing_rates = []

#     # calculate firing rate
#     for unit_id in sorted_true.get_unit_ids():

#         # get spike train
#         st = sorted_true.get_unit_spike_train(unit_id=unit_id)

#         # calculate firing rate
#         buccino_firing_rates += [len(st) / recording.get_total_duration()]

#     # cast as array
#     buccino_firing_rates = np.array(buccino_firing_rates)

#     # save firing rates
#     if save:
#         parent_path = os.path.dirname(BUCCI_FR_FILE_PATH)
#         if not os.path.isdir(parent_path):
#             os.makedirs(parent_path)
#         np.save(BUCCI_FR_FILE_PATH, buccino_firing_rates)
#     return buccino_firing_rates


# def compute_spike_rate_npx32(data_conf, NMC_RECORDING_PATH, NMC_GT_SORTING_PATH, NMC_FR_FILE_PATH, save=False):

#     print("\nnpx32")

#     recording = si.load_extractor(NMC_RECORDING_PATH)
#     sorted_true = si.load_extractor(NMC_GT_SORTING_PATH)
#     nmc_firing_rates = []
#     nmc_spike_count = []

#     # calculate firing rate
#     for unit_id in sorted_true.get_unit_ids():

#         # get spike train
#         st = sorted_true.get_unit_spike_train(unit_id=unit_id)

#         # calculate firing rate
#         nmc_firing_rates += [len(st) / recording.get_total_duration()]

#         # count spikes firing rate
#         nmc_spike_count += [len(st)]

#     # cast as array
#     nmc_firing_rates = np.array(nmc_firing_rates)

#     # save firing rates
#     if save:
#         parent_path = os.path.dirname(NMC_FR_FILE_PATH)
#         if not os.path.isdir(parent_path):
#             os.makedirs(parent_path)
#         np.save(NMC_FR_FILE_PATH, nmc_firing_rates)

#     # filter all near-contact pyramidal cells
#     simulation = load_campaign_params(data_conf)
#     SortingTrue = ground_truth.load(data_conf)

#     # get cell types
#     cell_morph = simulation["circuit"].cells.get(SortingTrue.unit_ids, properties=['morph_class'])

#     # sort pyramidal and interneurons
#     PYR_IDS = cell_morph[cell_morph["morph_class"] == "PYR"].index.values
#     INT_IDS = cell_morph[cell_morph["morph_class"] == "INT"].index.values

#     # pyr spiking rate
#     loc_pyr = []
#     for pyr_i in PYR_IDS:
#         loc_pyr.append(np.where(sorted_true.get_unit_ids() == pyr_i)[0][0])
#     pyramidal_firing_rate = np.array(nmc_firing_rates)[loc_pyr]

#     # interneurons spiking rate
#     loc_int = []
#     for int_i in INT_IDS:
#         loc_int.append(np.where(sorted_true.get_unit_ids() == int_i)[0][0])
#     interneuron_firing_rate = np.array(nmc_firing_rates)[loc_int]
#     return nmc_firing_rates, pyramidal_firing_rate, interneuron_firing_rate


def compute_spike_rate_npx384(data_conf, NMC_RECORDING_PATH, NMC_GT_SORTING_PATH, NMC_FR_FILE_PATH, save=False):

    print("\nnpx384")

    recording = si.load_extractor(NMC_RECORDING_PATH)
    sorted_true = si.load_extractor(NMC_GT_SORTING_PATH)
    nmc_firing_rates = []
    nmc_spike_count = []

    # calculate firing rate
    for unit_id in sorted_true.get_unit_ids():

        # get spike train
        st = sorted_true.get_unit_spike_train(unit_id=unit_id)

        # calculate firing rate
        nmc_firing_rates += [len(st) / recording.get_total_duration()]

        # count spikes firing rate
        nmc_spike_count += [len(st)]

    # cast as array
    nmc_firing_rates = np.array(nmc_firing_rates)

    # save firing rates
    if save:
        parent_path = os.path.dirname(NMC_FR_FILE_PATH)
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path)
        np.save(NMC_FR_FILE_PATH, nmc_firing_rates)

    # filter all near-contact pyramidal cells
    # simulation = load_campaign_params(data_conf)
    SortingTrue = ground_truth.load(data_conf)

    # get cell types
    # cell_morph = simulation["circuit"].cells.get(SortingTrue.unit_ids, properties=['morph_class'])

    # # sort pyramidal and interneurons
    # PYR_IDS = cell_morph[cell_morph["morph_class"] == "PYR"].index.values
    # INT_IDS = cell_morph[cell_morph["morph_class"] == "INT"].index.values

    # # pyr spiking rate
    # loc_pyr = []
    # for pyr_i in PYR_IDS:
    #     loc_pyr.append(np.where(sorted_true.get_unit_ids() == pyr_i)[0][0])
    # pyramidal_firing_rate = np.array(nmc_firing_rates)[loc_pyr]

    # interneurons spiking rate
    # loc_int = []
    # for int_i in INT_IDS:
    #     loc_int.append(np.where(sorted_true.get_unit_ids() == int_i)[0][0])
    # interneuron_firing_rate = np.array(nmc_firing_rates)[loc_int]
    return nmc_firing_rates, SortingTrue.unit_ids


def plot_firing_rate_hist_all_vs_interneurons_removed(data_all, data_pyr, log_x_min, log_x_max, nbins, t_dec, ax, color_all=(0.13, 0.23, 0.98), color_removed=(1, 0.07, 0.08), label_all="All units", label_removed="Interneurons removed"):

    p_pickup = lambda _freq: 1.0 - poisson(_freq * t_dec).cdf(0)

    x_bins = np.logspace(log_x_min, log_x_max, nbins)
    p_hist = p_pickup(x_bins[1:])
    p_pyr = p_pickup(data_pyr)
    p_all = p_pickup(data_all)

    H_pyr = np.histogram(data_pyr, bins=x_bins)[0] * p_hist
    H_all = np.histogram(data_all, bins=x_bins)[0] * p_hist

    mn_pyr = np.sum(np.log10(data_pyr) * p_pyr) / np.sum(p_pyr)
    sd_pyr = np.sum(np.abs(np.log10(data_pyr) * p_pyr - mn_pyr)) / np.sum(p_pyr)
    mn_all = np.sum(np.log10(data_all) * p_all) / np.sum(p_all)
    sd_all = np.sum(np.abs(np.log10(data_all) * p_all - mn_all)) / np.sum(p_all)
    
    # interneurons removed
    ax.plot(x_bins[1:], H_pyr/H_pyr.sum(), marker="o", ls="none", markerfacecolor=color_removed, markeredgecolor=color_removed, markersize=6, label=label_all)
    y_fit = norm(mn_pyr, sd_pyr).pdf(np.log10(x_bins[1:]))
    y_fit_pyr = H_pyr.sum() * y_fit / y_fit.sum()
    ax.plot(x_bins[1:], y_fit_pyr/sum(y_fit_pyr), color=color_removed, linewidth=2)

    # all neurons
    ax.plot(x_bins[1:], H_all/H_all.sum(), marker="o", ls="none", markerfacecolor=color_all, markeredgecolor=color_all, markersize=6, label=label_removed)
    y_fit = norm(mn_all, sd_all).pdf(np.log10(x_bins[1:]))
    y_fit_all = H_all.sum() * y_fit / y_fit.sum()
    ax.plot(x_bins[1:], y_fit_all/sum(y_fit_all), color=color_all, linewidth=2)
    
    ax.spines[["right","top"]].set_visible(False)
    ax.set_xscale("log")
    plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("spontaneous firing rate (Hz)")
    plt.ylabel("probability (ratio)")


def plot_firing_rate_hist_ground_truths(
        data_all:np.array, log_x_min, log_x_max, nbins, t_dec, ax, label, color=(0.13, 0.23, 0.98), markerfacecolor=(0.13, 0.23, 0.98), markeredgecolor=(0.13, 0.23, 0.98), linestyle="-",
        markersize=3, dashes=(5,0), legend=True, lognormal=True
        ):
    """TO DEPRECATE: see postpro.spikestats

    Args:
        data_all (np.array): _description_
        log_x_min (_type_): _description_
        log_x_max (_type_): _description_
        nbins (_type_): _description_
        t_dec (_type_): _description_
        ax (_type_): _description_
        label (_type_): _description_
        color (tuple, optional): _description_. Defaults to (0.13, 0.23, 0.98).
        markerfacecolor (tuple, optional): _description_. Defaults to (0.13, 0.23, 0.98).
        markeredgecolor (tuple, optional): _description_. Defaults to (0.13, 0.23, 0.98).
        linestyle (str, optional): _description_. Defaults to "-".
        markersize (int, optional): _description_. Defaults to 3.
        dashes (tuple, optional): _description_. Defaults to (5,0).
        legend (bool, optional): _description_. Defaults to True.
        lognormal (bool, optional): _description_. Defaults to True.
    """

    p_pickup = lambda _freq: 1.0 - poisson(_freq * t_dec).cdf(0)

    x_bins = np.logspace(log_x_min, log_x_max, nbins)
    p_hist = p_pickup(x_bins[1:])
    p_all = p_pickup(data_all)

    H_all = np.histogram(data_all, bins=x_bins)[0] * p_hist
    mn_all = np.sum(np.log10(data_all) * p_all) / np.sum(p_all)
    sd_all = np.sum(np.abs(np.log10(data_all) * p_all - mn_all)) / np.sum(p_all)

    # data points
    ax.plot(x_bins[1:], H_all/H_all.sum(), marker="o", ls="none", markersize=markersize, label=label, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
    y_fit = norm(mn_all, sd_all).pdf(np.log10(x_bins[1:]))
    y_fit_all = H_all.sum() * y_fit / y_fit.sum()

    # lognormal fit
    if lognormal:
        ax.plot(x_bins[1:], y_fit_all/sum(y_fit_all), color=color, linewidth=1.5, linestyle=linestyle, dashes=dashes)
        ax.spines[["right","top"]].set_visible(False)
        ax.set_xscale("log")
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([1e-3, 150])

    # show minor ticks
    ax.tick_params(which='both', width=1)
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    ax.xaxis.set_major_locator(locmaj)    
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    if legend:
        plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("spontaneous firing rate (Hz)")
        plt.ylabel("probability (ratio)")


def compute_sorted_spike_rate_buccino2020(BUCCI_RECORDING_PATH, BUCCI_SORTED_PATH, BUCCI_SORTED_FR_FILE_PATH, save=False):

    print("\nBuccino_2020")

    recording = si.load_extractor(BUCCI_RECORDING_PATH)
    Sorted = si.load_extractor(BUCCI_SORTED_PATH)
    buccino_firing_rates = []

    # calculate firing rate
    for unit_id in Sorted.get_unit_ids():
        st = Sorted.get_unit_spike_train(unit_id=unit_id)
        buccino_firing_rates += [len(st) / recording.get_total_duration()]

    # cast as array
    buccino_firing_rates = np.array(buccino_firing_rates)

    # save firing rates
    if save:
        parent_path = os.path.dirname(BUCCI_SORTED_FR_FILE_PATH)
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path)
        np.save(BUCCI_SORTED_FR_FILE_PATH, buccino_firing_rates)
    return buccino_firing_rates


def compute_sorted_spike_rate_npx(data_conf, NMC_RECORDING_PATH, SORTED_PATH, NMC_FR_FILE_PATH, save=False):
    """TO ODEPRECATE

    Args:
        data_conf (_type_): _description_
        NMC_RECORDING_PATH (_type_): _description_
        SORTED_PATH (_type_): _description_
        NMC_FR_FILE_PATH (_type_): _description_
        save (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    recording = si.load_extractor(NMC_RECORDING_PATH)
    Sorted = si.load_extractor(SORTED_PATH)
    nmc_firing_rates = []
    nmc_spike_count = []

    # calculate firing rate
    for unit_id in Sorted.get_unit_ids():

        # get spike train
        st = Sorted.get_unit_spike_train(unit_id=unit_id)

        # calculate firing rate
        nmc_firing_rates += [len(st) / recording.get_total_duration()]

        # count spikes firing rate
        nmc_spike_count += [len(st)]
        
    # cast as array
    nmc_firing_rates = np.array(nmc_firing_rates)

    # save firing rates
    if save:
        parent_path = os.path.dirname(NMC_FR_FILE_PATH)
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path)
        np.save(NMC_FR_FILE_PATH, nmc_firing_rates)
    return {
        "firing_rate": nmc_firing_rates,
        "unit_id": Sorted.get_unit_ids()
        }


def compute_sorted_spike_rate_for_unit_ids_npx(unit_ids, rec_path, sorting_path, fr_path, save=False):
    """TO DEPRECATE

    Args:
        unit_ids (_type_): _description_
        rec_path (_type_): _description_
        sorting_path (_type_): _description_
        fr_path (_type_): _description_
        save (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    recording = si.load_extractor(rec_path)
    sorted_true = si.load_extractor(sorting_path)
    firg_rate = []

    # calculate firing rate
    for unit_id in unit_ids:

        # get spike train
        st = sorted_true.get_unit_spike_train(unit_id=unit_id)

        # calculate firing rate
        firg_rate += [len(st) / recording.get_total_duration()]

    # cast as array
    firg_rate = np.array(firg_rate)

    # save firing rates
    if save:
        parent_path = os.path.dirname(fr_path)
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path)
        np.save(fr_path, firg_rate)
    return firg_rate


def get_spike_count_for_unit_ids_npx(unit_ids, data_conf, sorting_path, fr_path, save=False):

    sorted_true = si.load_extractor(sorting_path)
    count = []

    # calculate firing rate
    for unit_id in unit_ids:

        # get spike train
        st = sorted_true.get_unit_spike_train(unit_id=unit_id)

        # get spike count
        count += [len(st)]

    # cast as array
    count = np.array(count)

    # save spike count
    if save:
        parent_path = os.path.dirname(fr_path)
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path)
        np.save(fr_path, count)
    return count


def get_layerwise_fr(firing_rate:dict, unit_metadata:pd.DataFrame):
    """get layer-wise firing rates for this recording depth
    
    Args:
        firing_rate (dict):
        - key "unit_id": ids of the sorted units
        - key "firing_rate": firing rate associated with each unit id
        unit_metadata (pd.DataFrame):
        - layers (str): "L1", "L2"
    
    Returns:
        (np.array): arrays of unit firing rates for each layer
    """
    df_1 = pd.DataFrame(data=[firing_rate["unit_id"], firing_rate["firing_rate"]], index=["neuron", "firing_rate"]).T

    # standardize data
    # - convert unit ids to integers
    # - join L2 and L3 units
    # - rename Outside of the cortex -> Outside
    df_1["neuron"] = df_1["neuron"].astype(int)
    unit_metadata["layer"] = unit_metadata["layer"].apply(lambda x: x.replace(" ",""))
    unit_metadata["layer"] = unit_metadata["layer"].replace("L2", "L2/3")
    unit_metadata["layer"] = unit_metadata["layer"].replace("L3", "L2/3")
    unit_metadata["layer"] = unit_metadata["layer"].replace("Outsideofthecortex","Outside")

    # get units' layers
    df1 = df_1.merge(unit_metadata, on="neuron")
    layer_1_fr = df1[df1["layer"] == "L1"]["firing_rate"].values
    layer_2_3_fr = df1[df1["layer"] == "L2/3"]["firing_rate"].values
    layer_4_fr = df1[df1["layer"] == "L4"]["firing_rate"].values
    layer_5_fr = df1[df1["layer"] == "L5"]["firing_rate"].values
    layer_6_fr = df1[df1["layer"] == "L6"]["firing_rate"].values
    outside_fr = df1[df1["layer"] == "Outside"]["firing_rate"].values
    return layer_1_fr, layer_2_3_fr, layer_4_fr, layer_5_fr, layer_6_fr, outside_fr


def get_layerwise_fr_all_depths(fr_1:dict, fr_2:dict, fr_3:dict, meta_1:pd.DataFrame, meta_2:pd.DataFrame, meta_3:pd.DataFrame):
    """get layer-wise firing rates across all three recording depths in Horvath

    Args:
        fr_1 (dict): firing rates at depth 1
        - key "unit_id": unit ids 
        - key "firing_rate": firing rates
        fr_2 (dict): firing rates at depth 2
        fr_3 (dict): firing rates at depth 3
    
    TODO:
    - cast all args as dataframes
    """
    # depth 1
    (
        layer_1_fr_horvath_1, 
        layer_2_3_fr_horvath_1, 
        layer_4_fr_horvath_1, 
        layer_5_fr_horvath_1, 
        layer_6_fr_horvath_1, 
        outside_fr_horvath_1) = get_layerwise_fr(fr_1, meta_1)
    
    # depth 2
    (
        layer_1_fr_horvath_2, 
        layer_2_3_fr_horvath_2, 
        layer_4_fr_horvath_2, 
        layer_5_fr_horvath_2, 
        layer_6_fr_horvath_2, 
        outside_fr_horvath_2) = get_layerwise_fr(fr_2, meta_2)

    # depth 3
    (
        layer_1_fr_horvath_3, 
        layer_2_3_fr_horvath_3, 
        layer_4_fr_horvath_3, 
        layer_5_fr_horvath_3, 
        layer_6_fr_horvath_3, 
        outside_fr_horvath_3) = get_layerwise_fr(fr_3, meta_3)

    # STACK ACROSS DEPTH FILES
    fr_layer_1 = np.hstack([layer_1_fr_horvath_1, layer_1_fr_horvath_2, layer_1_fr_horvath_3])
    fr_layer_2_3 = np.hstack([layer_2_3_fr_horvath_1, layer_2_3_fr_horvath_2, layer_2_3_fr_horvath_3])
    fr_layer_4 = np.hstack([layer_4_fr_horvath_1, layer_4_fr_horvath_2, layer_4_fr_horvath_3])
    fr_layer_5 = np.hstack([layer_5_fr_horvath_1, layer_5_fr_horvath_2, layer_5_fr_horvath_3])
    fr_layer_6 = np.hstack([layer_6_fr_horvath_1, layer_6_fr_horvath_2, layer_6_fr_horvath_3])
    outside_fr = np.hstack([outside_fr_horvath_1, outside_fr_horvath_2, outside_fr_horvath_3])
    return fr_layer_1, fr_layer_2_3, fr_layer_4, fr_layer_5, fr_layer_6, outside_fr