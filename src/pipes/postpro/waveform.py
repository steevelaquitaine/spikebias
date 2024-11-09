"""pipeline to extract spike waveforms for near-contact cells

author: steeve.laquitaine@epfl.ch
date: 25.08.2023
modified: 25.08.2023

Usage:
    
    # submit to cluster
    sbatch cluster/extract_waveform.sbatch
"""
# SETUP PACKAGES
import os
import logging
import logging.config
import shutil
import yaml
from time import time
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from spikeinterface.comparison import GroundTruthStudy
from src.nodes.prepro import preprocess

# SET PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj68/home/laquitai/bernstein_2023"
os.chdir(PROJ_PATH)
from src.nodes.load import load_campaign_params
from src.nodes.utils import get_config

# SET PARAMETERS
EXPERIMENT = "silico_neuropixels"  # specifies the experiment
SIMULATION_DATE = "2023_08_17"  # specifies the run (date)

# waveform period in ms
MS_BEFORE = 3  
MS_AFTER = 3

# SETUP CONFIG
data_conf, param_conf = get_config(EXPERIMENT, SIMULATION_DATE).values()

# SETUP PATH
SPIKE_FILE_PATH = data_conf["dataeng"]["campaign"]["output"]["spike_file_path"]
RAW_LFP_TRACE_FILE_PATH = data_conf["dataeng"]["campaign"]["output"][
    "trace_file_path"
]

# SET WAVEFORM FOLDER
STUDY_FOLDER = data_conf["postprocessing"]["waveform"]["study"]
WAVEFORM_FOLDER = data_conf["postprocessing"]["waveform"]["WaveformExtractor"]

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
        times (_type_): _description_
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


def get_spike_sampling_freq(simulation: dict):
    return (
        1 / float(simulation["blue_config"]["Run_Default"]["Dt"])
    ) * 1000


def load_spike(simulation: dict, data_conf: dict):
    """Load raw ground truth spikes and write into a SpikeInterface Sorting extractor object

    Args:
        simulation (dict): _description_
        data_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # get read paths
    # spikes of near-contact units
    SPIKE_FILE_PATH = data_conf["dataeng"]["campaign"]["output"][
        "spike_file_path"
    ]
    LFP_FILE_PATH = data_conf["dataeng"]["campaign"]["output"][
        "trace_file_path"
    ]

    # get spikes and lfp
    spike = pd.read_pickle(SPIKE_FILE_PATH)
    lfp = pd.read_pickle(LFP_FILE_PATH)

    # TODO: This currenty provides the best estimation
    # of spike timestamp locations from 40000 to 10000 Hz
    # - add to preprocessing pipeline -> preprocessed spikes
    spike_loc = []
    for spike_ms_i in spike.index:
        spike_loc.append(np.abs(lfp.index - spike_ms_i).argmin())
    
    # find spike location on lfp (different sampling frequency)
    unit_id = spike.values
    return create_sorting_object(
        simulation["lfp_sampling_freq"],
        times=spike_loc,
        unit_ids=unit_id,
    )


def load_spike_same_sampling_freq_as_lfp(sim: dict, data_conf: dict):
    """Load raw ground truth spikes and write into a SpikeInterface Sorting extractor object
    when spike and lfp were acquired with the same sampling frequency

    Args:
        sim (dict): simulation parameters
        data_conf (dict): _description_

    Returns:
        _type_: _description_
    """
    # get spikes and lfp
    spike = pd.read_pickle(SPIKE_FILE_PATH)

    # get spike sampling frequency (fastest way)
    # and accurate when lfp and spike were acquired with the same 
    # sampling frequency
    sp_freq = get_spike_sampling_freq(sim)
    
    # get spike locations (as timepoints)
    spike_loc = np.int_(
        np.array(spike.index) * sp_freq / 1000
    )
    
    # cast as a SpikeInterface's sorting object
    unit_id = spike.values
    return create_sorting_object(
        sp_freq,
        times=spike_loc,
        unit_ids=unit_id,
    )


def get_lfp_sampling_freq(sim:dict):
    return sim["lfp_sampling_freq"]


def run():
    """extract spike waveforms from near-contact cells
    add a cell_id argument to filter by cell_ids

    Args:
        experiment (str):
        simulation_date (str): _description_
        lfp_trace_file (str): _description_
        spike_file (str): _description_
        cell_id (int): _description_
        study_folder (str): _description_
        ms_before (float): _description_
        ms_after (float): _description_

    Returns:
        _type_: waveform extractor
    """
    # load parameters
    sim = load_campaign_params(data_conf)

    # load spike timestamps
    t0 = time()
    if get_spike_sampling_freq(sim) == get_lfp_sampling_freq(sim):
        SortingTrue = load_spike_same_sampling_freq_as_lfp(sim, data_conf)
    else:
        SortingTrue = load_spike(sim, data_conf)
    logger.info(f"set spikes as timepoints in {time()-t0} secs")        

    # load preprocessed (fast, 312 spikes)
    trace = preprocess.load(data_conf)

    # create study (12s)
    t0 = time()
    gt_dict = {"rec0": (trace, SortingTrue)}
    shutil.rmtree(STUDY_FOLDER, ignore_errors=True)
    study = GroundTruthStudy.create(STUDY_FOLDER, gt_dict)
    logger.info(f"created study in {time()-t0} secs")

    # compute waveforms (0.21s, 312 spikes)
    # - this creates and write waveforms on disk in study/waveforms/
    # - note: for a single cell, no gain from multiprocessing
    # - 100 jobs is slow
    t0 = time()
    study.compute_waveforms(trace, ms_before=MS_BEFORE, ms_after=MS_AFTER, n_jobs=-1, total_memory="1G")
    logger.info(f"computed waveforms in {time()-t0} secs")

    # create waveform extractor from saved folder
    we = study.get_waveform_extractor(trace)    
    return we


if __name__ == "__main__":
    we = run()
    logger.info("done")