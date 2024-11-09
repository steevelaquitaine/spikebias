"""
author: steeve.laquitaine@epfl.ch
Usage:
    
    sbatch cluster/figures/main/buccino/noise_buttw.sbatch 
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd
import numpy as np
import spikeinterface as si
from concurrent.futures import ProcessPoolExecutor

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config

# SETUP PARAMETERS
RND_SEED = 0                # random seed
SFREQ_VIVO = 30000          # sampling frequency
SFREQ_B = 32000             # sampling frequency
WIND_END = 3700             # last segment to calculate mad

# vivo
data_conf_h_vivo, _ = get_config("vivo_marques", "c26").values() 
RAW_PATH_h_vivo = data_conf_h_vivo["raw"]
PREP_PATH_h_vivo = data_conf_h_vivo["preprocessing"]["output"]["trace_file_path"]
NOISE_VIVO_DATAPATH = data_conf_h_vivo["analyses"]["noise_stats"]["buttw_noise_0uV"]
CONTACTS_h = np.arange(0,128,1)

# silico
data_cfg, _ = get_config("buccino_2020", "2020").values()
PREPRO_B = data_cfg["preprocessing"]["output"]["trace_file_path"]
NOISE_B_DATAPATH = data_cfg["analyses"]["noise_stats"]["buttw"]


def measure_trace_noise(traces, sfreq, wind_end):
    """measure noise (mean absolute deviation)
    at consecutive segments of 1 second

    Args:
        traces: 2D array
    """
    winds = np.arange(0, wind_end, 1)
    mads = []
    for wind_i in winds:
        segment = traces[wind_i * sfreq : (wind_i + 1) * sfreq]
        mads.append(pd.DataFrame(segment).mad().values[0])
    return mads


def measure_vivo_trace_noise_parallel(traces_vivo, site):
    """Measure the minimum absolute deviation of a single trace
    over contiguous segments of one second
    "Site" is only used implicitly to pass each row of the
    traces_vivo matrix (the original timepoints x site matrix was transposed
    such that sites are in rows) to this function.

    Args:
        traces_vivo (np.array): a 1D trace array of site x timepoints
        site (int): the row used to implicitely extract that row from traces_vivo

    Returns:
        _type_: _description_
    """
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces_vivo, SFREQ_VIVO, WIND_END))


def measure_silico_trace_noise_parallel(traces_silico, site):
    """Measure the minimum absolute deviation of a single trace
    over contiguous segments of one second
    "Site" is only used implicitly to pass each row of the
    traces_vivo matrix (the original timepoints x site matrix was transposed
    such that sites are in rows) to this function.

    Args:
        traces_silico (np.array): a 1D trace array of site x timepoints
        site (int): the row used to implicitely extract that row from traces_vivo

    Returns:
        _type_: _description_
    """
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces_silico, SFREQ_B, WIND_END))


# 1. load recordings --------------
# - get vivo traces
PreRecording_h_vivo = si.load_extractor(PREP_PATH_h_vivo)
traces_vivo = PreRecording_h_vivo.get_traces()

# - get silico traces
PreRecording_h_silico = si.load_extractor(PREPRO_B)
traces_silico = PreRecording_h_silico.get_traces()

# 2. Compute layer-wise noise (13 mins) --------------

# VIVO
# measure site noise of in vivo traces (parallelized, )
with ProcessPoolExecutor() as executor:
    noise_by_trace = executor.map(
        measure_vivo_trace_noise_parallel,
        traces_vivo.T,
        np.arange(0, traces_vivo.shape[1], 1),
    )
vivo_noise_by_trace = list(noise_by_trace)

# SILICO
# measure site noise of fitted silico traces
with ProcessPoolExecutor() as executor:
    silico_noise_by_trace = executor.map(
        measure_silico_trace_noise_parallel,
        traces_silico.T,
        np.arange(0, traces_silico.shape[1], 1),
    )
silico_noise_by_trace = list(silico_noise_by_trace)

# save
os.makedirs(os.path.dirname(NOISE_VIVO_DATAPATH), exist_ok=True)
os.makedirs(os.path.dirname(NOISE_B_DATAPATH), exist_ok=True)
np.save(NOISE_VIVO_DATAPATH, vivo_noise_by_trace)
np.save(NOISE_B_DATAPATH, silico_noise_by_trace)

# check
print("Path preprocessed Buccino:", PREPRO_B)
print("Path Buccino noise", NOISE_B_DATAPATH)