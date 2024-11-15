"""Pipeline to compute data for power spectral density
takes 9 min

Usage:

    sbatch cluster/validation/main/marques/psd_10m_buttw_noise_fitd_raw_welch.sbatch

Returns:
    _type_: _description_
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import spikeinterface as si
import spikeinterface.extractors as se
import scipy
from scipy.optimize import minimize
import spikeinterface.full as si_full
from concurrent.futures import ProcessPoolExecutor
import scipy
import spikeinterface.preprocessing as spre
from scipy import signal

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.study import amplitude
from src.nodes.study import power

# pipeline
COMPUTE_TRACES_IN_CTX = True
COMPUTE_PSD = True

# SETUP PARAMETERS
SFREQ_VIVO = 30000          # sampling frequency
SFREQ_SILICO = 40000        # sampling frequency

# welch PSD plots
FILT_WINDOW = "hann"

# vivo
FILT_WIND_SIZE_VIVO = 30000 # 1Hz freq. resolution
FILT_WIND_OVERLAP_VIVO = int(
    FILT_WIND_SIZE_VIVO // 1.5
)

# silico 
FILT_WIND_SIZE_SILI = 40000 # 1Hz freq. resolution
FILT_WIND_OVERLAP_SILI = int(
    FILT_WIND_SIZE_SILI // 1.5
)

# SETUP DATASET CONFIG
# vivo
data_conf_vivo, param_conf_h_vivo = get_config("vivo_marques", "c26").values()
RAW_PATH_vivo = data_conf_vivo["probe_wiring"]["output"]
PREP_PATH_vivo = data_conf_vivo["preprocessing"]["output"]["trace_file_path"]

# silico
data_conf_sili, param_conf_sili = get_config(
    "silico_neuropixels", "concatenated"
).values()
RAW_PATH_sili = data_conf_sili["probe_wiring"]["output"]
PREP_PATH_sili = data_conf_sili["preprocessing"]["output"]["trace_file_path"]


def get_welch_psd_marques_silico_parallelized(traces: np.ndarray):
    """compute power spectrum density for Marques Silico
    using parallel computing

    Args:
        traces (np.ndarray): timepoints x sites voltage traces

    Returns:
        dict: frequencies x sites powers arraay and frequencies array
    """
    # takes 2 min (instead of 70 min w/o multiprocessing)
    nsites = traces.shape[1]

    # compute power for each site trace
    # in parallel with a pool of workers
    with ProcessPoolExecutor() as executor:
        power_by_site = executor.map(
            get_site_welch_psd_marques_silico,
            traces.T,
            np.arange(0, nsites, 1),
        )
    power_by_sites = list(power_by_site)

    # make an array with powers
    powersd = []
    for site in range(nsites):
        powersd.append(power_by_sites[site][0])
    powers = np.array(powersd)

    # store frequency domain
    freqs = power_by_sites[0][1]
    return {"power": powers, "freq": freqs}


def get_site_welch_psd_marques_silico(trace, site):
    """calculate the welch frequency powers in the input trace

    Args:
        traces (np.ndarray): timepoints x sites voltage trace
        sfreq (_type_): voltage trace sampling frequency
        site: silent, automatically generated by ProcessPoolExecutor()

    Returns:
        _type_: _description_
    """
    (freq, power) = signal.welch(
        trace,
        SFREQ_SILICO,
        window=FILT_WINDOW,
        nperseg=FILT_WIND_SIZE_SILI,
        noverlap=FILT_WIND_OVERLAP_SILI,
    )
    return np.array(power), np.array(freq)


def get_welch_psd_marques_vivo_parallelized(traces: np.ndarray):
    """compute power spectrum density for Marques Silico
    using parallel computing

    Args:
        traces (np.ndarray): timepoints x sites voltage traces

    Returns:
        dict: frequencies x sites powers arraay and frequencies array
    """
    # takes 2 min (instead of 70 min w/o multiprocessing)
    nsites = traces.shape[1]

    # compute power for each site trace
    # in parallel with a pool of workers
    with ProcessPoolExecutor() as executor:
        power_by_site = executor.map(
            get_site_welch_psd_marques_vivo,
            traces.T,
            np.arange(0, nsites, 1),
        )
    power_by_sites = list(power_by_site)

    # make an array with powers
    powersd = []
    for site in range(nsites):
        powersd.append(power_by_sites[site][0])
    powers = np.array(powersd)

    # store frequency domain
    freqs = power_by_sites[0][1]
    return {"power": powers, "freq": freqs}


def get_site_welch_psd_marques_vivo(trace, site):
    """calculate the welch frequency powers in the input trace

    Args:
        traces (np.ndarray): timepoints x sites voltage trace
        sfreq (_type_): voltage trace sampling frequency
        site: silent, automatically generated by ProcessPoolExecutor()

    Returns:
        _type_: _description_
    """
    (freq, power) = signal.welch(
        trace,
        SFREQ_VIVO,
        window=FILT_WINDOW,
        nperseg=FILT_WIND_SIZE_VIVO,
        noverlap=FILT_WIND_OVERLAP_VIVO,
    )
    return np.array(power), np.array(freq)


def save_psd_results(out_raw_sili, out_prep_sili, out_raw_vivo, out_prep_vivo):

    os.makedirs(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/postpro/realism/lfp/",
        exist_ok=True,
    )
    os.makedirs(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/realism/1_vivo/marques/c26_fixed/postpro/",
        exist_ok=True,
    )

    # save sili
    np.save(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/postpro/realism/lfp/full_raw_power_welch_10m",
        out_raw_sili,
    )

    np.save(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/postpro/realism/lfp/full_prep_buttw_power_welch_10m",
        out_prep_sili,
    )

    # save vivo
    np.save(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/realism/1_vivo/marques/c26_fixed/postpro/full_raw_power_welch_10m.npy",
        out_raw_vivo,
    )

    np.save(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/realism/1_vivo/marques/c26_fixed/postpro/full_prep_buttw_power_welch_10m.npy",
        out_prep_vivo,
    )


def load_psd_results():

    # save sili
    out_raw_sili = np.load(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/postpro/realism/lfp/full_raw_power_welch_10m.npy",
        allow_pickle=True,
    )
    out_raw_sili = out_raw_sili.item()

    out_prep_sili = np.load(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/postpro/realism/lfp/full_prep_buttw_power_welch_10m.npy",
        allow_pickle=True,
    )
    out_prep_sili = out_prep_sili.item()

    # save vivo
    out_raw_vivo = np.load(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/realism/1_vivo/marques/c26_fixed/postpro/full_raw_power_welch_10m.npy",
        allow_pickle=True,
    )
    out_raw_vivo = out_raw_vivo.item()

    out_prep_vivo = np.load(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/realism/1_vivo/marques/c26_fixed/postpro/full_prep_buttw_power_welch_10m.npy",
        allow_pickle=True,
    )
    out_prep_vivo = out_prep_vivo.item()
    return out_raw_sili, out_prep_sili, out_raw_vivo, out_prep_vivo


def save_traces_in_ctx(
    raw_traces_vivo, prep_traces_vivo, raw_traces_sili, prep_traces_sili
):

    # vivo (1m)
    np.save(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataeng/1_vivo/marques/campaign/c26_fixed/traces_in_ctx_10m_noise_fitd",
        raw_traces_vivo,
    )
    np.save(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/preprocessed/1_vivo/marques/campaign/c26_fixed/traces_in_ctx_10m_noise_fitd",
        prep_traces_vivo,
    )

    # silico
    np.save(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/dataeng/recording/traces_in_ctx_10m_noise_fitd",
        raw_traces_sili,
    )

    np.save(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/preprocessed/traces_in_ctx_10m_noise_fitd",
        prep_traces_sili,
    )


def load_saved_traces_in_ctx():

    # vivo
    raw_traces_vivo = np.load(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataeng/1_vivo/marques/campaign/c26_fixed/traces_in_ctx_10m_noise_fitd.npy"
    )
    prep_traces_vivo = np.load(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/preprocessed/1_vivo/marques/campaign/c26_fixed/traces_in_ctx_10m_noise_fitd.npy"
    )

    # silico
    raw_traces_sili = np.load(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/dataeng/recording/traces_in_ctx_10m_noise_fitd.npy"
    )

    prep_traces_sili = np.load(
        "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/preprocessed/traces_in_ctx_10m_noise_fitd.npy"
    )
    return raw_traces_vivo, prep_traces_vivo, raw_traces_sili, prep_traces_sili


# Load traces ***********************************
# load raw traces
RawVivo = si.load_extractor(RAW_PATH_vivo)
RawSili = si.load_extractor(RAW_PATH_sili)

# load preprocessed traces
PreprocessedVivo = si.load_extractor(PREP_PATH_vivo)
PreprocessedSili = si.load_extractor(PREP_PATH_sili)

# select first 10 minutes ****************************
RawVivo = RawVivo.frame_slice(start_frame=0, end_frame=SFREQ_VIVO * 10 * 60)
PreprocessedVivo = PreprocessedVivo.frame_slice(
    start_frame=0, end_frame=SFREQ_VIVO * 10 * 60
)
RawSili = RawSili.frame_slice(start_frame=0, end_frame=SFREQ_SILICO * 10 * 60)
PreprocessedSili = PreprocessedSili.frame_slice(
    start_frame=0, end_frame=SFREQ_SILICO * 10 * 60
)

# compress from floats to integers
RawSili = spre.astype(RawSili, "int16")
PreprocessedSili = spre.astype(PreprocessedSili, "int16")
RawVivo = spre.astype(RawVivo, "int16")
PreprocessedVivo = spre.astype(PreprocessedVivo, "int16")

# unit-test
assert RawSili.get_total_duration() == 10 * 60, "not 10 min"
assert RawVivo.get_total_duration() == 10 * 60, "not 10 min"

# Keep cortical sites ***********************************

# silico
layers = ["L1", "L2_3", "L4", "L5", "L6"]
site_layers_sili = RawSili.get_property("layers")
site_layers_sili = [
    "L2_3" if l_i == "L2" or l_i == "L3" else l_i for l_i in site_layers_sili
]
IN_CTX = np.isin(site_layers_sili, layers)
sites_sili = np.where(IN_CTX)[0]

# vivo
site_layers_vivo = RawVivo.get_property("layers")
IN_CTX = np.isin(site_layers_vivo, layers)
sites_vivo = np.where(IN_CTX)[0]


# (17m)Compute
if COMPUTE_TRACES_IN_CTX:

    # Keep only raw traces from sites in cortex
    raw_traces_sili = RawSili.get_traces()
    raw_traces_vivo = RawVivo.get_traces()

    # Keep only preprocessed traces from sites in cortex
    prep_traces_sili = PreprocessedSili.get_traces()
    prep_traces_vivo = PreprocessedVivo.get_traces()

    # get sites in cortex
    # silico
    raw_traces_sili = raw_traces_sili[:, sites_sili]
    prep_traces_sili = prep_traces_sili[:, sites_sili]

    # vivo
    raw_traces_vivo = raw_traces_vivo[:, sites_vivo]
    prep_traces_vivo = prep_traces_vivo[:, sites_vivo]

    #(1m) save
    save_traces_in_ctx(raw_traces_vivo, prep_traces_vivo, raw_traces_sili, prep_traces_sili)

else:
    # or load
    raw_traces_vivo, prep_traces_vivo, raw_traces_sili, prep_traces_sili = (
        load_saved_traces_in_ctx()
    )

# (77m) Compute
if COMPUTE_PSD:
    # raw
    out_raw_sili = get_welch_psd_marques_silico_parallelized(raw_traces_sili)
    out_raw_vivo = get_welch_psd_marques_vivo_parallelized(raw_traces_vivo)

    # preprocessed
    out_prep_sili = get_welch_psd_marques_silico_parallelized(prep_traces_sili)
    out_prep_vivo = get_welch_psd_marques_vivo_parallelized(prep_traces_vivo)

    # save
    save_psd_results(out_raw_sili, out_prep_sili, out_raw_vivo, out_prep_vivo)
else:
    # or load
    out_raw_sili, out_prep_sili, out_raw_vivo, out_prep_vivo = load_psd_results()