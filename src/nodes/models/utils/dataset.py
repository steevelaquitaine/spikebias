"""Useful functions to load and format datasets
for modeling

author: steeve.laquitaine@epfl.ch
"""

import os
import spikeinterface as si
import spikeinterface.core.template_tools as ttools
from spikeinterface import comparison
from spikeinterface.qualitymetrics import compute_quality_metrics as qm
from spikeinterface import qualitymetrics
import pandas as pd
from cebra import CEBRA
import cebra
import torch
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns
from sklearn import metrics
import cebra.models
import shutil
from spikeinterface.postprocessing import compute_principal_components
import multiprocessing
from src.nodes.utils import euclidean_distance


def add_spike_amplitude_extension(we, n_sites, load_if_exists: bool):
    """Add spike amplitudes to WaveformExtractor

    Args:
        we (WaveformExtractor): _description_
        n_sites (int): typically 384 for neuropixels
        load_if_exists (bool): load if exists

    Returns:
        WaveformExtractor: _description_
    """

    # these two properties are required to compute amplitudes
    we.recording.set_property("gain_to_uV", np.ones((n_sites,)))
    we.recording.set_property("offset_to_uV", np.zeros((n_sites,)))

    # compute spike amplitudes
    # or it as an extension
    if not load_if_exists:
        _ = si.postprocessing.compute_spike_amplitudes(we, outputs="by_unit")
    else:
        we.load_extension("spike_amplitudes")

    # unit-test
    assert we.has_extension("spike_amplitudes"), "load spike_amplitudes extension"
    return we


def get_waveformExtractor_for_single_units(
    sort_path: str,
    study_path: str, 
    save_path: str,
    n_sites: int = 384,
    load_if_exists: bool = False,
    add_pca: bool = True,
    job_kwargs: dict = {"n_jobs":-1},
    
):
    """Setup WaveformExtractors to calculate quality metrics for single units

    Args:
        sort_path (str): path of SpikeInterface SortingExtractor
        study_path (str): path of SpikeInterface Study
        save_path (str): path to save SpikeInterface WaveformExtractor
        n_sites (int, optional): _description_. Defaults to 384. number of probe sites
        load_if_exists: bool=False (bool): load all pre-computed data if they exist
        add_pca (bool): only if load_if_exists=False, pre-compute PCA data,
        - required for all cluster isolation quality metrics
        job_kwargs (dict): config for parallel computing 
        - n_jobs (int): -1 means all cpu cores
        - progress_bar (bool)

    Returns:
        WaveformExtractor: WaveformExtractor setup to compute quality metrics 
        for a subset of selected units
    """
    # compute
    if not load_if_exists:
        # get single units
        Sorting = si.load_extractor(sort_path)
        su_ix = np.where(Sorting.get_property("KSLabel") == "good")[0]
        su_unit_ids = Sorting.unit_ids[su_ix]

        # load WaveformExtractor
        We = si.WaveformExtractor.load_from_folder(study_path)

        # create waveformExtractor for single units
        # which we will keep for all downstream analyses
        # this should speed up computations
        shutil.rmtree(save_path, ignore_errors=True)
        WeSu = We.select_units(unit_ids=su_unit_ids, new_folder=save_path)

        # setup two properties required to calculate some quality metrics
        WeSu.recording.set_property("gain_to_uV", np.ones((n_sites,)))
        WeSu.recording.set_property("offset_to_uV", np.zeros((n_sites,)))

        # augment extractors with pca results
        if add_pca:
            _ = compute_principal_components(
                waveform_extractor=WeSu,
                n_components=5,
                mode="by_channel_local",
                **job_kwargs,
            )
    else:
        # or load existing
        WeSu = si.WaveformExtractor.load_from_folder(save_path)
    return WeSu


def get_good_sorted_unit_ids(
    quality, quality_path: str, sorter: str, exp: str, layer: str, fltd_unit: list
):
    """_summary_

    Args:
        quality (_type_): _description_
        quality_path (str): _description_
        sorter (str): _description_
        exp (str): _description_
        layer (str): _description_
        fltd_unit (list): _description_

    Returns:
        np.array(int): filtered sorted unit ids
    """
    # standardize to quality table
    layer = layer.upper()
    
    # load quality results
    unit_quality = pd.read_csv(quality_path)

    # select a sorted unit and conditions
    df = unit_quality[
        (unit_quality["quality"].str.contains(quality))
        & (unit_quality["experiment"] == exp)
        & (unit_quality["sorter"] == sorter)
        & (unit_quality["layer"] == layer)
    ]
    # filter units based on previous conditions
    df = df[df["sorted"].isin(fltd_unit)]
    return df["sorted"].values.astype(int)


def get_poor_sorted_unit_ids(
    quality, quality_path: str, sorter: str, exp: str, layer: str, fltd_unit: list
):
    """_summary_

    Args:
        quality (_type_): 'mixed: overmerger + oversplitter'
        quality_path (str): _description_
        sorter (str): _description_
        exp (str): _description_
        layer (str): _description_
        fltd_unit (list[int]): _description_

    Returns:
        np.array[int]: filtered sorted unit ids
    """
    
    # standardize to quality table
    layer = layer.upper()
    np.array(fltd_unit).astype(int)
    
    # load quality results
    unit_quality = pd.read_csv(quality_path)

    # select a sorted unit and conditions
    df = unit_quality[
        (unit_quality["quality"] == quality)
        & (unit_quality["experiment"] == exp)
        & (unit_quality["sorter"] == sorter)
        & (unit_quality["layer"] == layer)
    ]
    df = df[df["sorted"].isin(fltd_unit)]
    return df["sorted"].values.astype(int)


def get_spike_dataset_for(
    unit_ids: np.array,
    we,
    max_spikes: int,
    interval_ms: float,
    sfreq: int,
    downsample: int,
):
    """_summary_

    Args:
        unit_ids (np.array[int]): _description_
        we (_type_): _description_
        max_spikes (int): _description_
        interval_ms (float): _description_
        sfreq (int): _description_
        downsample (int): downsample waveforms to produce
        - a lower sampling frequency (e.g., 2 to reduce a 40 KHz frequency
        to 20 KHz)

    Returns:
        _type_: _description_
    """
    # convert interval in ms to samples
    ttp_sp = we.nbefore
    bef_aft_sp = interval_ms * sfreq / 1000
    interval = np.arange(ttp_sp - bef_aft_sp, ttp_sp + bef_aft_sp, 1).astype(int)

    # get all units nearest channels (with extremum amplitude)
    max_chids = ttools.get_template_extremum_channel(we, peak_sign="both")

    # loop over good units
    # 240 samples (-3 to 3 ms at 40KHz)
    wvs = np.zeros((int(max_spikes * we.nbefore * 2 / downsample), 1))
    unit_label = []
    for unit in unit_ids:
        
        # get its waveforms (num_spikes, num_samples, num_channels)
        wv = we.get_waveforms(unit_id=unit)

        # get its nearest channel
        c_ids = we.sparsity.unit_id_to_channel_ids[unit]
        max_chid = max_chids[unit]
        max_chid_ix = np.where(c_ids == max_chid)[0][0]

        # get waveform for that channels (2D)
        # and (num_samples, num_spikes)
        # TODO: sample instead of taking the first ones
        wv_i = np.array(wv[:max_spikes, interval[::downsample], max_chid_ix]).flatten()[
            :, None
        ]

        # record waveforms
        wvs = np.hstack([wvs, wv_i])

    wvs = wvs[:, 1:]
    # unit_label = np.array(unit_label)
    unit_label = np.array(unit_ids)
    return wvs, unit_label


def drop_sites(wv, c_ids, max_chid_ix, n_sites):

    # drop channels to get the number of
    # sites for all waveforms
    n_to_drop = wv.shape[2] - n_sites

    # if the channel is located closest to the
    # first site
    if len(c_ids) - max_chid_ix > max_chid_ix:
        # remove the n last sites
        wv = wv[:, :, :-n_to_drop]
    else:
        # remove the n first sites
        n_to_drop = wv.shape[2] - n_sites
        wv = wv[:, :, n_to_drop:]
    return wv


def get_n_nearest_sites(site_coord, max_chid, c_ids, n_sites):
    """get the id and indices of the n_sites nearest
    to the max_chids (site of unit). The sites are
    ordered not by distance but by id to conserve
    their probe grid layout.

    Args:
        site_coord (_type_): _description_
        max_chid (_type_): _description_
        c_ids (_type_): _description_
        n_sites (_type_): _description_

    Returns:
        _type_: _description_
    """
    eudist = []
    for c_i in c_ids:
        eudist.append(euclidean_distance(site_coord[max_chid, :], site_coord[c_i, :]))
    ix = np.argsort(eudist)
    new_c_ix = np.sort(ix[:n_sites])
    new_c_ids = c_ids[new_c_ix]
    return new_c_ids, new_c_ix


def get_2d_spike_dataset_for(
    unit_ids: np.array,
    we,
    max_spikes: int,
    interval_ms: float,
    sfreq: int,
    downsample: int,
    n_sites: int = 25,
    site_coord: np.array = None,
):
    """_summary_

    Args:
        unit_ids (np.array[int]): _description_
        we (_type_): _description_
        max_spikes (int): _description_
        interval_ms (float): _description_
        sfreq (int): _description_
        downsample (int): downsample waveforms to produce
        - a lower sampling frequency (e.g., 2 to reduce a 40 KHz frequency
        to 20 KHz)

    Returns:
        _type_: _description_
    """

    # convert interval in ms to samples
    ttp_sp = we.nbefore
    bef_aft_sp = interval_ms * sfreq / 1000
    interval = np.arange(ttp_sp - bef_aft_sp, ttp_sp + bef_aft_sp, 1).astype(int)

    # loop over units
    # get number of sites subsampled (sparsity)

    # get all units nearest channels (with extremum amplitude)
    max_chids = ttools.get_template_extremum_channel(we, peak_sign="both")

    # preallocate flattened 2D waveform
    wvs = np.zeros((int(max_spikes * we.nbefore * 2 / downsample * n_sites), 1))
    unit_label = []

    # lopp over units and collect waveforms
    for unit in unit_ids:

        # get its waveforms (num_spikes, num_samples, num_channels)
        wv = we.get_waveforms(unit_id=unit)

        # get its nearest channel
        c_ids = we.sparsity.unit_id_to_channel_ids[unit]
        max_chid = max_chids[unit]
        # max_chid_ix = np.where(c_ids == max_chid)[0][0]

        # drop sites away from nearest site to get the
        # same number of sites for all waveforms
        # wv = drop_sites(wv, c_ids, max_chid_ix, n_sites)

        # get nearest sites to the unit (zoom in)
        _, ix_new = get_n_nearest_sites(site_coord, max_chid, c_ids, n_sites)

        # flatten the 2D waveform
        wv_i = np.array(
            wv[:max_spikes, :, ix_new][:, interval[::downsample], :]
        ).flatten()[:, None]

        # record waveforms
        wvs = np.hstack([wvs, wv_i])

    wvs = wvs[:, 1:]
    unit_label = np.array(unit_ids)
    return wvs, unit_label


def get_sorted_unit_best_score(
    KS4_ns_10m: str,
    GT_ns_10m: str,
):
    """Get sorted unit best agreement scores

    Args:
        KS4_ns_10m (str): path of SortingExtractor
        GT_ns_10m (str): path of GroundTruth SortingExtractor

    Returns:
        pd.Series:
        - index_ sorted units
        - values: agreement scores
    """
    SortingTrue = si.load_extractor(GT_ns_10m)
    SortingTrue = SortingTrue.remove_empty_units()
    Sorting = si.load_extractor(KS4_ns_10m)
    comp = comparison.compare_sorter_to_ground_truth(
        SortingTrue,
        Sorting,
        match_mode="hungarian",
        exhaustive_gt=True,
        delta_time=1.3,
        compute_labels=True,
        compute_misclassifications=False,
        well_detected_score=0.8,
        match_score=0.8,  # modified
        redundant_score=0.2,  # default - we don't use that info in this analysis
        overmerged_score=0.2,  # default - we don't use that info in this analysis
        chance_score=0.1,  # default - we don't use that info in this analysis
    )
    return comp.agreement_scores.max()


def get_dataset_for(
    we,
    quality_path: str,
    sorter: str,
    exp: str,
    layer: str,
    max_spikes: int,
    flt_unit: list[int],
    interval_ms: float,
    sfreq: int,
    downsample: int,
    wave_dim: int = 1,
    n_sites: int = 25,
    site_coord: np.array = None,
):
    """get dataset

    Args:
        we (WaveformExtractor): WaveformExtractor
        quality_path (str): path of the pandas dataframe
        - containing sorted unit quality classification
        sorter (str): one of "KS4", "KS3", "KS2.5", "KS2"...
        - contained in the quality dataframe in the "sorter" column
        exp (str): _description_
        layer (str): _description_
        max_spikes (int): _description_
        flt_unit (list[int]): _description_
        interval_ms (float): _description_
        sfreq (int): _description_
        downsample
        wave_dim: waveform dimension
        - 1: for temporal. Waveform are
        are collected from the nearest channel only)
        - 2: spatiotemporal. Waveform are collected from
        the "n_sites" nearest channels
        n_sites: common number of sites to keep if 2D-waveforms

    Returns:
        _type_: _description_
    """
    # get good units (filtered based on some conditions)
    g_units = get_good_sorted_unit_ids(
        "good", quality_path, sorter, exp, layer, flt_unit
    )

    # poor units (filtered based on some conditions)
    p_units = get_poor_sorted_unit_ids(
        "mixed: overmerger + oversplitter", quality_path, sorter, exp, layer, flt_unit
    )
    
    # if waveform are 1-dim (temporal on the nearest channel)
    if wave_dim == 1:
        wvs_good, good_unit_label = get_spike_dataset_for(
            g_units, we, max_spikes, interval_ms, sfreq, downsample
        )
        wvs_poor, poor_unit_label = get_spike_dataset_for(
            p_units, we, max_spikes, interval_ms, sfreq, downsample
        )
        
    # if waveform are 2-dim (spatio-temporal on 30 nearest channels)
    elif wave_dim == 2:
        
        wvs_good, good_unit_label = get_2d_spike_dataset_for(
            g_units, we, max_spikes, interval_ms, sfreq, downsample, n_sites, site_coord
        )
        wvs_poor, poor_unit_label = get_2d_spike_dataset_for(
            p_units, we, max_spikes, interval_ms, sfreq, downsample, n_sites, site_coord
        )

    # spike dataset
    spike_data = np.hstack([wvs_good, wvs_poor]).T

    # quality label (1D discrete, CEBRA can handle only one)
    quality_label = np.hstack(
        [np.array([1] * len(good_unit_label)), np.array([0] * len(poor_unit_label))]
    )

    # unit ids
    unit_ids = np.hstack([g_units, p_units])

    return spike_data, quality_label, unit_ids


def get_amplitudes_for(unit_ids, amplitudes):
    return {unit: amplitudes[unit] for unit in unit_ids}


def get_abs_CV(amplitude):
    """calculate the absolute coefficient
    of variation of the waveform amplitudes.

    Args:
        amplitudes (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.absolute(np.std(amplitude) / np.median(amplitude))


def get_abs_CV_all_units(amplitudes, we, unit_ids):

    # get waveform amplitudes
    amplitudes = get_amplitudes_for(unit_ids, amplitudes)

    # calculate coefficient of variation
    cv = [get_abs_CV(amplitudes[unit]) for unit in unit_ids]

    # replace infinite values (when amplitude is 0) by a large number
    cv = [1e6 if cv_i == np.inf else cv_i for cv_i in cv]
    return cv


def get_quality_metrics_table(
    qmetrics,
    amplitudes,
    we,
    unit_ids: list,
):
    """_summary_

    Args:
        qmetrics
        we (_type_): _description_
        unit_ids (_type_): _description_
        load_if_exists (bool, optional): _description_. Defaults to False.
        - False takes 20 secs/unit (1h for 180 units; 2h30 for 400 units)

    Returns:
        pd.DataFrame: table of quality metrics (column) for each unit id (index)

    Requirements:
        * pca has been pre-computed and added to WaveformExtractors with
        get_waveformExtractor_for_single_units()
    """

    # keep metrics for specified units
    qmetrics = qmetrics.loc[unit_ids]
    
    # add custom coefficient of variation of amplitude
    cvs = get_abs_CV_all_units(amplitudes, we, unit_ids)
    qmetrics["amplitude_cv"] = cvs
    return qmetrics


def get_dataset_by_layer(
    sort_path: str,
    gt_path: str,
    we,
    qpath: str,
    sorter: str,
    exp: str,
    num_spike: int,
    interval_ms: float,
    downsample: int,
    continuous_qmetrics: bool = False,
    qmetrics_in_dataset: bool = False,
    load_amp_if_exists: bool = True,
    load_qm_if_exists: bool = True,
    wave_dim: int = 1,
    n_sites: int = 384,
    site_coord: np.array = None,
    layers: list=["l23", "l4", "l5", "l6"]
):
    """get a dataset by layer

    Args:
        sort_path (str): _description_
        gt_path (str): ground truth SortingExtractor
        we (WaveformExtractor): WaveformExtractor
        qpath (str): file path of .csv dataframe with sorted unit
            - quality evaluation
        sorter (str): spike sorter
        exp (str): experiment
        num_spike (int): number of spikes kept from units
        interval_ms (float): interval before and after timestamp in ms
        downsample (int): waveform downsampling factor (1, 2, 3 ...)
        continuous_qmetrics (bool): compute quality metrics
            - needed to use them as auxiliary variables
            - or needed to add them the waveform dataset
        load_if_exists (bool): load all pre-computed datasets if they
        exist.
        n_sites (int): if wave_dim=2, number of nearest sites to keep
        layer (list):

    Returns:
        dict: _description_
    """

    # unit filtering ************************************
    
    # calculate the common maximum number
    # of spikes that it is possible to use
    # across all units (the number of spikes of the
    # least active unit)
    Sorting = si.load_extractor(sort_path)
    unit_spike = Sorting.get_total_num_spikes()
    sfreq = Sorting.get_sampling_frequency()
    print("Recording sampling frequency:", sfreq)

    # filter units with at least num_spikes
    n_spike = [unit_spike[key] for key in unit_spike]
    
    if num_spike == None:
        num_spike = min(n_spike)
        flt_unit = [unit for unit in unit_spike]
    else:
        flt_unit = [unit for unit in unit_spike if unit_spike[unit] > num_spike]

    # filter units with non-nan quality metrics    
    if continuous_qmetrics:
        
        print("calculating continuous quality metrics labels")
        job_kwargs = dict(n_jobs=-1, verbose=True, progress_bar=True)

        # add spike amplitudes
        we = add_spike_amplitude_extension(
            we, n_sites=n_sites, load_if_exists=load_amp_if_exists
        )
                
        # pre-compute Spiketinterface quality metrics
        # 20 secs/unit, other pca-based metrics "rp_violation",
        # "nearest_neighbor", "nn_isolation", "nn_noise_overlap"
        # were much too slow to include in the iteration
        # all "nn_noise_overlap" were nan:
        # frpm all metrics that don't require PCA
        # select only those that are not nan nor constant
        # and that count relative (generalizable to new recordings)
        # and not absolute values such as number of spike counts
        # we remove the minimum ratio of amplitudes to bins
        # for amplitude_cutoff to always prevent nan values which the
        # model cannot handle
        
        # test remove amplitude_cutoff
        qmetrics = qm(
            we,
            qm_params={
                'amplitude_cutoff': 
                    {
                        'peak_sign': 'neg', 
                        'num_histogram_bins': 100, 
                        'histogram_smoothing_value': 3, 
                        'amplitudes_bins_min_ratio': 2 # default 5; tested: 1
                     }
                    },
            load_if_exists=load_qm_if_exists,
            skip_pc_metrics=True,
            **job_kwargs
            )
        
        # add silhouette metric (pca-based but fast enough)
        silhouette = qm(
            we,
            metric_names=["silhouette"],
            **job_kwargs,
        )
        qmetrics["silhouette"] = silhouette.values
        
        # handle missing metrics
        print("****************** Analysing data completion ***************")
        
        print("Data completion:", qmetrics.notna().sum())
        
        # complete_metrics = ["amplitude_median", "firing_range", "firing_rate", 
        #                     "isi_violations_ratio", "isi_violations_count",
        #                     "num_spikes", "presence_ratio", "rp_contamination", 
        #                     "rp_violations", "sd_ratio", "snr", "sync_spike_2",
        #                     "sync_spike_4", "sync_spike_8"]
        #complete_metrics = ["snr", "isi_violations_ratio", "isi_violations_count",
        #                    "sd_ratio", "firing_rate", "presence_ratio"]
        complete_metrics = ["snr", "isi_violations_ratio", "isi_violations_count",
                            "sd_ratio", "firing_rate", "presence_ratio", "silhouette"]

        # amplitude cutoff can result in dropping half the unit sample
        # because CEBRA fails with nan. If it has too many nan. It should
        # be removed. Preliminary results indicates that including it 
        # increases recall. 
        #incomplete_metrics = ["amplitude_cutoff"]
        incomplete_metrics = []
        
        # select metrics to use
        metric_names = complete_metrics + incomplete_metrics
        qmetrics = qmetrics[metric_names]
        
        # keep non-nan units only
        print("unit sample size before dropping NaN:", qmetrics.shape[0])
        qmetrics = qmetrics.dropna()
        qm_unit_ids = qmetrics.index.values
        print("unit sample size after dropping NaN:", qmetrics.shape[0])

        # update unit filter
        flt_unit = np.intersect1d(qm_unit_ids, flt_unit)
        
        # pre-compute amplitudes for amplitude coefficient of variation metric
        amplitudes = si.postprocessing.compute_spike_amplitudes(
            we, peak_sign="neg", outputs="by_unit", load_if_exists=load_amp_if_exists
        )[0]
    else:
        print("Continuous quality metrics labels are NOT calculated.")
        
    # get dataset for CEBRA by layer ***************************
    
    # initialize
    spike_data_l23 = None
    spike_data_l4 = None
    spike_data_l5 = None
    spike_data_l6 = None
    quality_label_l23 = None
    quality_label_l4 = None
    quality_label_l5 = None
    quality_label_l6 = None
    unit_ids_l23 = None
    unit_ids_l4 = None
    unit_ids_l5 = None
    unit_ids_l6 = None
    
    if "l23" in layers:
        spike_data_l23, quality_label_l23, unit_ids_l23 = get_dataset_for(
            we,
            qpath,
            sorter,
            exp,
            "l2/3",
            num_spike,
            flt_unit,
            interval_ms,
            sfreq,
            downsample,
            wave_dim,
            n_sites,
            site_coord,
        )     
    
    if "l4" in layers:
        spike_data_l4, quality_label_l4, unit_ids_l4 = get_dataset_for(
            we,
            qpath,
            sorter,
            exp,
            "l4",
            num_spike,
            flt_unit,
            interval_ms,
            sfreq,
            downsample,
            wave_dim,
            n_sites,
            site_coord,
        )
    if "l5" in layers:
        spike_data_l5, quality_label_l5, unit_ids_l5 = get_dataset_for(
            we,
            qpath,
            sorter,
            exp,
            "l5",
            num_spike,
            flt_unit,
            interval_ms,
            sfreq,
            downsample,
            wave_dim,
            n_sites,
            site_coord,
        )
    if "l6" in layers:
        spike_data_l6, quality_label_l6, unit_ids_l6 = get_dataset_for(
            we,
            qpath,
            sorter,
            exp,
            "l6",
            num_spike,
            flt_unit,
            interval_ms,
            sfreq,
            downsample,
            wave_dim,
            n_sites,
            site_coord,
        )

    # if we use continuous quality metrics
    cont_label_l23 = None
    cont_label_l4 = None
    cont_label_l5 = None
    cont_label_l6 = None
    
    if continuous_qmetrics:
        
        # add quality metrics per layer
        if "l23" in layers:
            cont_label_l23 = get_quality_metrics_table(
                qmetrics,
                amplitudes,
                we,
                unit_ids_l23,
            )
            print("Computed quality metrics for layer 23")
        if "l4" in layers:
            cont_label_l4 = get_quality_metrics_table(
                qmetrics,
                amplitudes,
                we,
                unit_ids_l4,
            )
            print("Computed quality metrics for layer 4")
        if "l5" in layers:
            cont_label_l5 = get_quality_metrics_table(
                qmetrics,
                amplitudes,
                we,
                unit_ids_l5,
            )
            print("Computed quality metrics for layer 5")
        if "l6" in layers:
            cont_label_l6 = get_quality_metrics_table(
                qmetrics,
                amplitudes,
                we,
                unit_ids_l6,
            )
            print("Computed quality metrics for layer 6")

        print("********************************************")
        print("The quality metrics added to the dataset are:", cont_label_l6.columns)
        print("********************************************")
        
    # we add quality metrics to the dataset
    if qmetrics_in_dataset:
        if not continuous_qmetrics:
            raise ValueError("continuous_qmetrics must be set to True")

        # add quality metrics to waveform dataset
        if "l23" in layers:
            spike_data_l23 = np.hstack([spike_data_l23, cont_label_l23])
        if "l4" in layers:
            spike_data_l4 = np.hstack([spike_data_l4, cont_label_l4])
        if "l5" in layers:
            spike_data_l5 = np.hstack([spike_data_l5, cont_label_l5])
        if "l6" in layers:
            spike_data_l6 = np.hstack([spike_data_l6, cont_label_l6])
    else:
        print("Continuous quality metrics labels are NOT added to the dataset.")
        
    # get best scores of sorted unit
    best_score = get_sorted_unit_best_score(
        sort_path,
        gt_path,
    )
    print("ex. data shape (L4):", spike_data_l4.shape)
    print("ex. label shape (L4):", quality_label_l4.shape)

    # bundle dataset for model 1 (by layer)
    dataset1 = {
        "data_l23": spike_data_l23,
        "data_l4": spike_data_l4,
        "data_l5": spike_data_l5,
        "data_l6": spike_data_l6,
        "label_l23": quality_label_l23,
        "label_l4": quality_label_l4,
        "label_l5": quality_label_l5,
        "label_l6": quality_label_l6,
        "cont_label_l23": cont_label_l23,
        "cont_label_l4": cont_label_l4,
        "cont_label_l5": cont_label_l5,
        "cont_label_l6": cont_label_l6,
        "unit_ids_l23": unit_ids_l23,
        "unit_ids_l4": unit_ids_l4,
        "unit_ids_l5": unit_ids_l5,
        "unit_ids_l6": unit_ids_l6,
        "best_score": best_score,
        "nb_spikes": num_spike,
    }
    return dataset1


def get_dataset_pooled(
    dat1: dict, layers: list=["l23", "l4", "l5", "l6"], continuous_qmetrics: bool = False, qmetrics_in_dataset: bool = False
):
    """_summary_

    Args:
        dat1 (dict): _description_
        layers (list, optional): _description_. Defaults to ["l23", "l4", "l5", "l6"].
        continuous_qmetrics (bool, optional): _description_. Defaults to False.
        qmetrics_in_dataset (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # concatenate waveform data
    spike_data = []
    for l_i in layers:
        spike_data.append(dat1[f"data_{l_i}"])
    spike_data = np.vstack(spike_data)
            
    # concatenate waveform data
    quality_label = []
    for l_i in layers:
       quality_label.append(dat1[f"label_{l_i}"])
    quality_label = np.hstack(quality_label)

    # continuous quality metrics label
    cont_label = None
    
    # concatenate continuous labels
    if continuous_qmetrics:
        cont_label = []
        for l_i in layers:
            cont_label.append(dat1[f"cont_label_{l_i}"])
        cont_label = pd.concat(cont_label)

    # augment waveform dataset with quality metrics
    if qmetrics_in_dataset:
        spike_data = np.hstack([spike_data, cont_label.values])

    # unit ids
    # concatenate waveform data
    unit_ids = []
    for l_i in layers:
       unit_ids.append(dat1[f"unit_ids_{l_i}"])
    unit_ids = np.hstack(unit_ids)
        
    return {
        "data": spike_data,
        "label": quality_label,
        "cont_label": cont_label,
        "unit_ids": unit_ids,
    }

