import pickle
import shutil
from time import time

import numpy as np
import pandas as pd
import probeinterface as pi
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
from matplotlib import pyplot as plt
from spikeinterface.comparison import GroundTruthStudy

from src.nodes.load import load_campaign_params
from src.nodes.prepro import preprocess
from src.nodes.utils import get_config


def run_from_files(
    experiment: str,
    simulation_date: str,
    lfp_trace_file: str,
    spike_file: str,
    study_folder: str,
    ms_before: float,
    ms_after: float,
    load_preprocessed: bool,
):
    """Run spike waveform extraction

    TODO:
    - add a cell_id argument to filter by cell_ids

    Args:
        experiment (str):
        simulation_date (str): _description_
        lfp_trace_file (str): _description_
        spike_file (str): _description_
        cell_id (int): _description_
        study_folder (str): _description_
        ms_before (float): _description_
        ms_after (float): _description_
        load_preprocessed (bool): load preprocessed recording and spike timestamps

    Returns:
        _type_: _description_
    """

    # load parameters
    t0 = time()
    data_conf, param_conf = get_config(experiment, simulation_date).values()
    simulation = load_campaign_params(data_conf)
    print("Set parameters in ", time() - t0, "sec")

    # load data
    t0 = time()
    lfp_trace = pd.read_pickle(lfp_trace_file)
    spike_ms = pd.read_pickle(spike_file)
    print("Loaded raw recording and spikes in ", time() - t0, "sec")

    # preprocessed spike timepoints
    t0 = time()
    spike_loc = []

    # load preprocessed location
    if load_preprocessed:
        with open(
            data_conf["preprocessing"]["output"]["spike_file_path"], "rb"
        ) as file:
            spike_loc = pickle.load(file)
    else:
        # find location indices of the spike time indices nearest to
        # the trace time indices
        for spike_ms_i in spike_ms.index:
            spike_loc.append(np.abs(lfp_trace.index - spike_ms_i).argmin())

        # write preprocessed spike timestamps
        with open(
            data_conf["preprocessing"]["output"]["spike_file_path"], "wb"
        ) as f:
            pickle.dump(spike_loc, f)

    # map spikes with units in SI's SortingObject
    times = np.array(spike_loc)
    labels = spike_ms.values
    SortingObject = se.NumpySorting.from_times_labels(
        [times], [labels], simulation["lfp_sampling_freq"]
    )
    print(
        "Found spike loc for trace sampfreq in ",
        time() - t0,
        "sec",
    )

    # load preprocess recording
    t0 = time()
    if load_preprocessed:
        lfp_recording = preprocess.load(data_conf)
        print("Loaded preprocessed recording in ", time() - t0, "sec")
    else:
        # preprocess recording
        lfp_recording = preprocess.run(data_conf, param_conf)
        print("Preprocessed recording in ", time() - t0, "sec")

    # create study
    t0 = time()
    gt_dict = {
        "rec0": (lfp_recording, SortingObject),
    }
    shutil.rmtree(study_folder, ignore_errors=True)
    study = GroundTruthStudy.create(study_folder, gt_dict)
    print("Created study in ", time() - t0, "sec")

    # compute waveforms
    # - this creates and write waveforms on disk
    t0 = time()
    study.compute_waveforms(lfp_recording)
    print("Computed waveforms in ", time() - t0, "sec")

    # setup waveform extractor
    t0 = time()
    WaveformExtractor = study.get_waveform_extractor(lfp_recording)
    WaveformExtractor.set_params(ms_before=ms_before, ms_after=ms_after)
    WaveformExtractor.run_extract_waveforms()
    print("Extracted waveforms in ", time() - t0, "sec")
    return WaveformExtractor


def load(lfp_recording, study_folder: str, ms_before: float, ms_after: float):
    """load WaveformExtractor from existing study
    takes 10 min for 500s units

    Args:
        lfp_recording (_type_): _description_
        study_folder (str): _description_

    Returns:
        _type_: _description_
    """
    study = GroundTruthStudy(study_folder)
    Extractor = study.get_waveform_extractor(lfp_recording)
    #Extractor.set_params(ms_before=ms_before, ms_after=ms_after)
    #Extractor.run_extract_waveforms()
    return Extractor


def plot(WaveformExtractor, cell_id: int, colors, linewidth_instance, linewidth_mean, nspike):
    """plot waveform for cell

    Args:
        WaveformExtractor (_type_): _description_
        cell_id (int): _description_

    Returns:
        _type_: _description_
    """
    # Turn interactive plotting off (lots of plots !)
    plt.ioff()

    # get channels where spike amplitude is maximal
    max_chids = spost.get_template_extremum_channel(
        WaveformExtractor, peak_sign="both"
    )

    # get cell waveforms
    waveform = WaveformExtractor.get_waveforms(cell_id)

    # derived peri-spike time period (ms)
    ms_before = (
        -WaveformExtractor.nbefore
        / WaveformExtractor.sampling_frequency
        * 1000
    )
    ms_after = (
        WaveformExtractor.nafter / WaveformExtractor.sampling_frequency * 1000
    )
    timepoint_ms = 1 / WaveformExtractor.sampling_frequency * 1000
    n_timepoints = int(
        (abs(ms_before) / timepoint_ms) + (ms_after / timepoint_ms)
    )
    time_axis = np.linspace(ms_before, ms_after, num=n_timepoints)

    # set plot
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)

    # plot waveform instances
    waveform_instances = waveform[:nspike, :, max_chids[cell_id]].T
    ax.plot(
        time_axis,
        waveform_instances,
        color=colors[0],
        lw=linewidth_instance,
        label="spike instances",
    )

    # plot average waveform
    tmp = WaveformExtractor.get_template(cell_id)
    ax.plot(
        time_axis,
        tmp[:, max_chids[cell_id]],
        color=colors[1],
        label="average spike",
        lw=linewidth_mean
    )

    # setup legends
    ax.set_xlabel(
        "time to spike timestamp (msecs)",
        fontsize=10,
    )
    ax.set_ylabel("extracellular potential (V)", fontsize=10)
    ax.set_title(
        f"Cell id: {cell_id} on Channel {max_chids[cell_id]}", fontsize=10
    )
    handles, labels = ax.get_legend_handles_labels()
    display = (0, waveform_instances.shape[1])
    return fig, ax

def plot2(WaveformExtractor, cell_id: int, ms_before, ms_after, colors, linewidth_instance, linewidth_mean, nspike):
    """plot waveform for cell

    Args:
        WaveformExtractor (_type_): _description_
        cell_id (int): _description_
        ms_before (): in ms
        ms_after (): in ms

    Returns:
        _type_: _description_
    """
    # Turn interactive plotting off (lots of plots !)
    plt.ioff()

    # get channels where spike amplitude is maximal
    max_chids = spost.get_template_extremum_channel(
        WaveformExtractor, peak_sign="both"
    )

    # get cell waveforms
    waveform = WaveformExtractor.get_waveforms(cell_id)

    # get timepoint samples
    timepoint_ms = 1 / WaveformExtractor.sampling_frequency * 1000
    n_timepoints = int(
        (abs(ms_before) / timepoint_ms) + (ms_after / timepoint_ms)
    )
    time_axis = np.linspace(ms_before, ms_after, num=n_timepoints)

    # set plot
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)

    # plot waveform instances
    waveform_instances = waveform[:nspike, :, max_chids[cell_id]].T
    ax.plot(
        time_axis,
        waveform_instances,
        color=colors[0],
        lw=linewidth_instance,
        label="spike instances",
    )

    # plot average waveform
    tmp = WaveformExtractor.get_template(cell_id)
    ax.plot(
        time_axis,
        tmp[:, max_chids[cell_id]],
        color=colors[1],
        label="average spike",
        lw=linewidth_mean
    )

    # setup legends
    ax.set_xlabel(
        "time to spike timestamp (msecs)",
        fontsize=10,
    )
    ax.set_ylabel("extracellular potential (V)", fontsize=10)
    ax.set_title(
        f"Cell id: {cell_id} on Channel {max_chids[cell_id]}", fontsize=10
    )
    handles, labels = ax.get_legend_handles_labels()
    display = (0, waveform_instances.shape[1])
    return fig, ax

def plot_by_channel(
    WaveformExtractor, channel_ids: list, cell_id: int, figsize: tuple
):
    """plot waveform for cell

    Args:
        WaveformExtractor (_type_): _description_
        cell_id (int): _description_

    Returns:
        _type_: _description_
    """

    # setup figure
    colors = ["Olive", "Teal", "Fuchsia"]
    fig, axes = plt.subplots(len(channel_ids), 1, figsize=figsize)

    for c_i, channel_id in enumerate(channel_ids):
        # get cell waveforms
        waveform = WaveformExtractor.get_waveforms(cell_id)

        # plot waveform instances
        axes[c_i].plot(waveform[:, :, channel_id].T, color=colors[1], lw=0.3)

        # plot average waveform
        tmp = WaveformExtractor.get_template(cell_id)
        axes[c_i].plot(tmp[:, channel_id], color=colors[2])
        axes[c_i].set_title(
            f"Cell id: {cell_id} on Channel {channel_id}", fontsize=6
        )
    plt.tight_layout()
    return fig
