"""Node to load the ground truth spikes of in-silico simulations

TODO:
- unit-test

Usage:
    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    python3.9 app.py simulation --pipeline sort --type ground_truth --conf 2023_01_13
    
Returns:
    _type_: _description_

Refs:
    https://spikeinterface.readthedocs.io/en/latest/modules/core/plot_2_sorting_extractor.html
"""

import logging
import logging.config
import shutil
from time import time

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import yaml

from src.nodes.load import load_campaign_params

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def create_sorting_object(sampling_freq, times, unit_ids):
    """Cast spikes as a SpikeInterface Sorting Extractor object

    Args:
        sorting_object_write_path (str): _description_
        sampling_freq (_type_): _description_
        times (_type_): spike sample location
        labels (_type_): _description_

    Returns:
        Sorting Extractor: Ground truth Sorting Extractor
    """
    logger.info(
        "Creating SpikeInterface's SortingTrue extractor"
    )
    return se.NumpySorting.from_times_labels(
        [times], [unit_ids], sampling_freq
    )


def load_spike(dataset_conf: dict, param_conf: dict):
    """Load raw ground truth spikes and write into a 
    SpikeInterface SortingExtractor object

    note: this function is only relevant to the raw Blue Brain simulations not 
    to the NWB DANDI archive dataset
    
    Args:
        simulation (dict): _description_
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # get read paths
    SPIKE_FILE_PATH = dataset_conf["dataeng"]["campaign"]["output"][
        "spike_file_path"
    ]
    TRACE_FILE_PATH = dataset_conf["dataeng"]["campaign"]["output"][
        "trace_file_path"
    ]

    # get spikes and lfp
    spike = pd.read_pickle(SPIKE_FILE_PATH)
    trace = pd.read_pickle(TRACE_FILE_PATH)

    # carefully deal with numerical precision issues
    SPIKE_SAMP_FREQ = get_spike_sampling_freq(param_conf)
    TRACE_SAMP_FREQ = get_lfp_sampling_freq(param_conf)
    
    # convert to ms
    SPIKE_SFREQ_MS = SPIKE_SAMP_FREQ / 1000
    TRACE_SFREQ_MS = TRACE_SAMP_FREQ / 1000
    
    # get number of timepoints on spike index reference
    spike_npoint_for_40KHz = spike.index.values * SPIKE_SFREQ_MS

    # get the corresponding sample location on the trace
    conv_factor = TRACE_SFREQ_MS / SPIKE_SFREQ_MS
    spike_tpoints_for_20KHz = (spike_npoint_for_40KHz * conv_factor).astype(int)

    # keep only spikes within the recording duration
    max_ntpoints_trace = trace.index.shape[0]
    spike_tpoints_for_20KHz = spike_tpoints_for_20KHz[spike_tpoints_for_20KHz <= max_ntpoints_trace]
    spike = spike.iloc[:len(spike_tpoints_for_20KHz)]
    
    # narrow the search space for each spike
    spike_loc = []
    
    for s_i, spike_ms_i in enumerate(spike.index):
        
        # define narrower search window
        start_wind = spike_tpoints_for_20KHz[s_i] - 30
        end_wind = spike_tpoints_for_20KHz[s_i] + 30

        # for the last spike, end window at lfp's last timepoint
        search_window = np.arange(start_wind, end_wind, 1)
        if search_window[-1] >= trace.index.shape[0]:
            search_window = np.arange(start_wind, trace.index.shape[0], 1)

        # improve spike timepoint location and record
        loc_in_window = np.abs(trace.index[search_window] - spike_ms_i).argmin()
        spike_loc.append(start_wind + loc_in_window)

    # find spike location on lfp (different sampling frequency)
    unit_id = spike.values
    return create_sorting_object(
        TRACE_SAMP_FREQ,
        times=spike_loc,
        unit_ids=unit_id,
    )


def load_spike_2X(simulation: dict, dataset_conf: dict, param_conf: dict):
    """Load raw ground truth spikes for 2X-accelerated traces and write 
    into a SpikeInterface Sorting extractor object

    Args:
        simulation (dict): _description_
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # get read paths
    SPIKE_FILE_PATH = dataset_conf["dataeng"]["campaign"]["output"][
        "spike_file_path"
    ]
    TRACE_FILE_PATH = dataset_conf["dataeng"]["campaign"]["output"][
        "trace_file_path"
    ]

    # get spikes and lfp
    spike = pd.read_pickle(SPIKE_FILE_PATH)
    trace = pd.read_pickle(TRACE_FILE_PATH)

    # carefully deal with numerical precision issues
    simulation = load_campaign_params(dataset_conf)
    SPIKE_SAMP_FREQ = get_spike_sampling_freq(simulation)
    TRACE_SAMP_FREQ = get_lfp_sampling_freq(simulation)
    SPIKE_SFREQ_MS = SPIKE_SAMP_FREQ / 1000
    TRACE_SFREQ_MS = TRACE_SAMP_FREQ / 1000
        
    # get number of timepoints on spike index reference
    spike_npoint_for_40KHz = spike.index.values * SPIKE_SFREQ_MS

    # get number of timepoints on trace index reference
    conv_factor = TRACE_SFREQ_MS / SPIKE_SFREQ_MS
    spike_tpoints_for_20KHz = (spike_npoint_for_40KHz * conv_factor).astype(int)

    # narrow the search space for each spike
    spike_loc = []
    for s_i, spike_ms_i in enumerate(spike.index):
        
        # define narrower search window
        start_wind = spike_tpoints_for_20KHz[s_i] - 30
        end_wind = spike_tpoints_for_20KHz[s_i] + 30

        # for the last spike, end window at lfp's last timepoint
        search_window = np.arange(start_wind, end_wind, 1)
        if search_window[-1] >= trace.index.shape[0]:
            search_window = np.arange(start_wind, trace.index.shape[0], 1)

        # improve spike timepoint location and record
        loc_in_window = np.abs(trace.index[search_window] - spike_ms_i).argmin()
        spike_loc.append(start_wind + loc_in_window)

    # reduce spike timepoint location by 2X
    # to match 2X accelerated trace samples
    spike_loc = np.floor(spike_loc / 2)
    
    # find spike location on recording (different sampling frequency)
    unit_id = spike.values
    return create_sorting_object(
        simulation["lfp_sampling_freq"],
        times=spike_loc,
        unit_ids=unit_id,
    )


def get_spike_sampling_freq(param_conf: dict):
    return (
        1 / param_conf["spike"]["Dt"]
    ) * 1000


def get_lfp_sampling_freq(param_conf: dict):
    return param_conf["sampling_freq"]


def load_spike_same_sampling_freq_as_lfp(dataset_conf: dict, param_conf: dict):
    """Load raw ground truth spikes and write into a SpikeInterface Sorting extractor object
    when spike and lfp were acquired with the same sampling frequency
    
    note: this function is only relevant to the raw Blue Brain simulations not to the NWB DANDI archive
    dataset

    Args:
        simulation (dict): _description_
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # get read paths
    SPIKE_FILE_PATH = dataset_conf["dataeng"]["campaign"]["output"][
        "spike_file_path"
    ]

    # get spikes and lfp
    spike = pd.read_pickle(SPIKE_FILE_PATH)

    # get spike sampling frequency (fastest way)
    # and accurate when lfp and spike were acquired with the same 
    # sampling frequency
    #SPIKE_SAMPLING_FREQ = get_spike_sampling_freq(simulation)
    SPIKE_SAMPLING_FREQ = get_spike_sampling_freq(param_conf)
    
    # get spike locations (as timepoints)
    spike_loc = np.int_(
        np.array(spike.index) * SPIKE_SAMPLING_FREQ / 1000
    )
    
    # cast as a SpikeInterface's sorting object
    unit_id = spike.values
    return create_sorting_object(
        SPIKE_SAMPLING_FREQ,
        times=spike_loc,
        unit_ids=unit_id,
    )


def load_spike_same_sampling_freq_as_lfp_2X(simulation: dict, dataset_conf: dict, param_conf: dict):
    """Load raw ground truth spikes for 2X-accelerated traces and write into a
    SpikeInterface Sorting extractor object when spike and lfp were acquired with 
    the same sampling frequency

    Args:
        simulation (dict): _description_
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # get read paths
    SPIKE_FILE_PATH = dataset_conf["dataeng"]["campaign"]["output"][
        "spike_file_path"
    ]
    
    # get spikes and lfp
    # pd.Series (index: spike time, unit)
    spike = pd.read_pickle(SPIKE_FILE_PATH)

    # get spike sampling frequency (fastest way)
    # and accurate when lfp and spike were acquired with the same 
    # sampling frequency
    SPIKE_SAMPLING_FREQ = get_spike_sampling_freq(simulation)
    
    # get spike locations (as timepoints)
    spike_loc = np.int_(
        np.array(spike.index) * SPIKE_SAMPLING_FREQ / 1000
    )
    
    # reduce spike timepoint location by 2X
    # to match 2X accelerated trace samples
    spike_loc = np.floor(spike_loc / 2)
    
    # cast as a SpikeInterface's sorting object
    unit_id = spike.values
    return create_sorting_object(
        SPIKE_SAMPLING_FREQ,
        times=spike_loc,
        unit_ids=unit_id,
    )
    
    
def load(data_conf: dict):
    """Load an already processed SpikeInterface's ground truth SortingExtractor

    Args:
        data_conf (dict): _description_

    Returns:
        (SortingExtractor): SpikeInterface ground truth SortingExtractor
    """

    # get write paths
    SORTING_GT_WRITE_PATH = data_conf["sorting"]["simulation"]["ground_truth"][
        "output"
    ]

    t0 = time()
    logger.info("loading already processed ground truth SortingExtractor ...")

    # load extractor
    GtSortingExtractor = si.load_extractor(SORTING_GT_WRITE_PATH)

    # log
    logger.info(
        "loading already processed true sorting - done in %s",
        round(time() - t0, 1),
    )
    return GtSortingExtractor


def run(dataset_conf: dict, param_conf: dict):
    """record ground truth spike timestamps into a spikeinterface 
    SortingExtractor object
    
    Args:
        simulation (dict): parameters of the campaign derived 
        from one simulation

    Returns:
        dict: _description_
    """
    t0 = time()
    if get_spike_sampling_freq(param_conf) == get_lfp_sampling_freq(param_conf):
        sorting_object = load_spike_same_sampling_freq_as_lfp(dataset_conf, param_conf)        
    else:
        sorting_object = load_spike(dataset_conf, param_conf)
        logger.info("converted spike sampling freq in %s", round(time()-t0, 1))
    return {"ground_truth_sorting_object": sorting_object}


def run2X(simulation: dict, dataset_conf: dict, param_conf: dict):
    """get ground truth for 2X accelerated traces as spikeinterface
    sorting object
    Args:
        simulation (dict): parameters of the campaign derived 
        from one simulation

    Returns:
        _type_: _description_
    """
    # track time
    t0 = time()
    
    # create ground truth
    # if spike sampling freq. is the same as recording sampling frequency
    if get_spike_sampling_freq(simulation) == get_lfp_sampling_freq(simulation):
        SortingTrue = load_spike_same_sampling_freq_as_lfp_2X(simulation, dataset_conf, param_conf)
    
    # else search best spike location
    else:
        SortingTrue = load_spike_2X(simulation, dataset_conf, param_conf)
    logger.info("Created ground truth spikes for 2X-accelerated traces in %s", round(time() - t0,1))
    return {"ground_truth_sorting_object": SortingTrue}


def write(SortingTrue, data_conf: dict):
    """write Spikeinterface's SortingTrue extractor object

    Args:
        SortingTrue (_type_): _description_
        data_conf (dict): _description_
    """
    
    # get write paths
    WRITE_PATH = data_conf["ground_truth"]["output"]

    # clear path
    shutil.rmtree(WRITE_PATH, ignore_errors=True)

    # write
    t0 = time()
    SortingTrue.save(folder=WRITE_PATH, n_jobs=-1, total_memory="2G")
    logger.info("written Ground truth SortingExtractor in %s", round(time()-t0, 1))