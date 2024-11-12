"""Pipeline to compute data for power spectral density for the first 10 
minutes of neuropixels traces

takes 9 min

Usage:

    sbatch cluster/validation/main/marques/psd_10m_buttw_noise_fitd_gain_ftd_10perc_less_raw_welch.sbatch

Returns:
    _type_: _description_
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os
import numpy as np
import spikeinterface as si
from concurrent.futures import ProcessPoolExecutor
import spikeinterface.preprocessing as spre
from scipy import signal

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config


# SETUP PARAMETERS *******************

# pipeline
COMPUTE_TRACES_IN_CTX = True
COMPUTE_PSD = True

# sampling frequency
SF_NV = 30000  # Marques-Smith
SF_NS = 40000  # Biophy. spontaneous
SF_NE = 20000  # Biophy. evoked
SF_NB = 32000  # Synthetic model (Buccino et al., 2020)


# SETUP DATASETS *******************

# neuropixels (Marques-Smith)
cfg_nv, _ = get_config("vivo_marques", "c26").values()
RAW_PATH_NV = cfg_nv["probe_wiring"]["output"]
PRE_PATH_NV = cfg_nv["preprocessing"]["output"]["trace_file_path"]
RAW_PSD_PATH_NV = cfg_nv["validation"]["psd"]["raw"]
PRE_PSD_PATH_NV = cfg_nv["validation"]["psd"]["preprocessed"]

# neuropixels (Biophy. spont.)
cfg_ns, _ = get_config(
    "silico_neuropixels", "concatenated"
).values()
RAW_PATH_NS = cfg_ns["probe_wiring"]["40m"]["output_noise_fitd_gain_fitd_adj10perc_less_int16"]
PRE_PATH_NS = cfg_ns["preprocessing"]["output"]["40m"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]
RAW_PSD_PATH_NS = cfg_ns["validation"]["psd"]["raw"]
PRE_PSD_PATH_NS = cfg_ns["validation"]["psd"]["preprocessed"]

# neuropixels (Biophy. evoked)
cfg_ne, _ = get_config("silico_neuropixels", "stimulus").values()
RAW_PATH_NE = cfg_ne["probe_wiring"]["output"]
PRE_PATH_NE = cfg_ne["preprocessing"]["output"]["trace_file_path"]
RAW_PSD_PATH_NE = cfg_ne["validation"]["psd"]["raw"]
PRE_PSD_PATH_NE = cfg_ne["validation"]["psd"]["preprocessed"]

# neuropixels (synthetic, Buccino)
cfg_nb, param_conf_nb = get_config("buccino_2020", "2020").values()

# Buccino with best fitted gain for layer 5
cfg_nb, _ = get_config("buccino_2020", "2020").values()
RAW_PATH_NB = cfg_nb["probe_wiring"]["10m"]["output_gain_fitd_int16"]
PRE_PATH_NB = cfg_nb["preprocessing"]["output"]["trace_file_path_gain_ftd"]
RAW_PSD_PATH_NB = cfg_nb["validation"]["psd"]["raw"]
PRE_PSD_PATH_NB = cfg_nb["validation"]["psd"]["preprocessed"]


# SETUP WELCH PSD PARAMETERS *******************
FILT_WINDOW = "hann"

# neuropixels (Marques-smith)
FILT_WIND_NV = 30000 # 1Hz resolution
FILT_OVERL_NV = int(
    FILT_WIND_NV // 1.5
)

# neuropixels (Biophy. spontaneous)
FILT_WIND_NS = 40000 # 1Hz resolution
FILT_OVERL_NS = int(
    FILT_WIND_NS // 1.5
)

# neuropixels (Biophy. evoked)
FILT_WIND_NE = 20000 # 1Hz resolution
FILT_OVERL_NE = int(
    FILT_WIND_NE // 1.5
)

# neuropixels (Synth. Buccino)
FILT_WIND_NB = 32000 # 1Hz resolution
FILT_OVERL_NB = int(
    FILT_WIND_NB // 1.5
)


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


def get_welch_psd_ne_parallelized(traces: np.ndarray):
    """compute power spectrum density for neuropixels Biophysical model
    in the evoked condition using parallel computing

    Args:
        traces (np.ndarray): timepoints x sites voltage traces

    Returns:
        dict: frequencies x sites powers array and frequencies array
        
    note: we typically use 72 cores
    """
    # takes 2 min (instead of 70 min w/o multiprocessing)
    nsites = traces.shape[1]

    # compute power for each site trace
    # in parallel with a pool of workers
    with ProcessPoolExecutor() as executor:
        power_by_site = executor.map(
            get_site_welch_psd_ne,
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


def get_welch_psd_nb_parallelized(traces: np.ndarray):
    """compute power spectrum density for neuropixels synthetic model
    (Buccino et al.,) in the spontaneous condition using parallel
    computing

    Args:
        traces (np.ndarray): timepoints x sites voltage traces

    Returns:
        dict: frequencies x sites powers array and frequencies array
        
    note: we typically use 72 cores
    """
    # takes 2 min (instead of 70 min w/o multiprocessing)
    nsites = traces.shape[1]

    # compute power for each site trace
    # in parallel with a pool of workers
    with ProcessPoolExecutor() as executor:
        power_by_site = executor.map(
            get_site_welch_psd_nb,
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
        SF_NV,
        window=FILT_WINDOW,
        nperseg=FILT_WIND_NV,
        noverlap=FILT_OVERL_NV,
    )
    return np.array(power), np.array(freq)


def get_site_welch_psd_marques_silico(trace, site):
    """calculate the welch frequency powers contained in 
    the voltage traces of the Marques-Smith's in vivo dataset

    Args:
        traces (np.ndarray): timepoints x sites voltage trace
        sfreq (_type_): voltage trace sampling frequency
        site: silent, automatically generated by ProcessPoolExecutor()

    Returns:
        _type_: _description_
    """
    (freq, power) = signal.welch(
        trace,
        SF_NS,
        window=FILT_WINDOW,
        nperseg=FILT_WIND_NS,
        noverlap=FILT_OVERL_NS,
    )
    return np.array(power), np.array(freq)


def get_site_welch_psd_ne(trace, site):
    """calculate the welch frequency powers in the input trace
    for the biophysical model in the evoked condition

    Args:
        traces (np.ndarray): timepoints x sites voltage trace
        sfreq (_type_): voltage trace sampling frequency
        site: silent, automatically generated by ProcessPoolExecutor()

    Returns:
        _type_: _description_
    """
    (freq, power) = signal.welch(
        trace,
        SF_NE,
        window=FILT_WINDOW,
        nperseg=FILT_WIND_NE,
        noverlap=FILT_OVERL_NE,
    )
    return np.array(power), np.array(freq)


def get_site_welch_psd_nb(trace, site):
    """calculate the welch frequency powers in the input trace
    for the synthetic model (Buccino et al.,2020)

    Args:
        traces (np.ndarray): timepoints x sites voltage trace
        sfreq (_type_): voltage trace sampling frequency
        site: silent, automatically generated by ProcessPoolExecutor()

    Returns:
        _type_: _description_
    """
    (freq, power) = signal.welch(
        trace,
        SF_NB,
        window=FILT_WINDOW,
        nperseg=FILT_WIND_NB,
        noverlap=FILT_OVERL_NB,
    )
    return np.array(power), np.array(freq)


def save_psd_results(
    out_raw_sili, out_prep_sili, out_raw_vivo, out_prep_vivo, out_raw_ne, out_prep_ne, out_raw_nb, out_prep_nb
    ):

    # save Marques-Smith (vivo)
    parent_path = os.path.dirname(RAW_PSD_PATH_NV)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(RAW_PSD_PATH_NV, out_raw_vivo)
    
    parent_path = os.path.dirname(PRE_PSD_PATH_NV)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(PRE_PSD_PATH_NV, out_prep_vivo)

    # save Biophy. spont. (model)
    parent_path = os.path.dirname(RAW_PSD_PATH_NS)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(RAW_PSD_PATH_NS, out_raw_sili)
    
    parent_path = os.path.dirname(PRE_PSD_PATH_NS)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(PRE_PSD_PATH_NS, out_prep_sili)

    # save Biophy. evoked (model)
    parent_path = os.path.dirname(RAW_PSD_PATH_NE)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(RAW_PSD_PATH_NE, out_raw_ne)
    
    parent_path = os.path.dirname(PRE_PSD_PATH_NE)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(PRE_PSD_PATH_NE, out_prep_ne)

    # save synthetic. (Buccino model)
    parent_path = os.path.dirname(RAW_PSD_PATH_NB)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(RAW_PSD_PATH_NB, out_raw_nb)
    
    parent_path = os.path.dirname(PRE_PSD_PATH_NB)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(PRE_PSD_PATH_NB, out_prep_nb)

# Load traces ***********************************
# raw
RawNV = si.load_extractor(RAW_PATH_NV)
RawNS = si.load_extractor(RAW_PATH_NS)
RawNE = si.load_extractor(RAW_PATH_NE)
RawNB = si.load_extractor(RAW_PATH_NB)
# preprocessed
PreNV = si.load_extractor(PRE_PATH_NV)
PreNS = si.load_extractor(PRE_PATH_NS)
PreNE = si.load_extractor(PRE_PATH_NE)
PreNB = si.load_extractor(PRE_PATH_NB)

# select first 10 minutes of the traces ****************************
# raw
RawNV = RawNV.frame_slice(start_frame=0, end_frame=SF_NV * 10 * 60)
RawNS = RawNS.frame_slice(start_frame=0, end_frame=SF_NS * 10 * 60)
RawNE = RawNS.frame_slice(start_frame=0, end_frame=SF_NE * 10 * 60)
RawNB = RawNS.frame_slice(start_frame=0, end_frame=SF_NB * 10 * 60)
# preprocessed
PreNV = PreNV.frame_slice(
    start_frame=0, end_frame=SF_NV * 10 * 60
)
PreNS = PreNS.frame_slice(
    start_frame=0, end_frame=SF_NS * 10 * 60
)
PreNE = PreNE.frame_slice(
    start_frame=0, end_frame=SF_NE * 10 * 60
)
PreNB = PreNB.frame_slice(
    start_frame=0, end_frame=SF_NB * 10 * 60
)

# compress from floats to integers
# raw
RawNS = spre.astype(RawNS, "int16")
RawNV = spre.astype(RawNV, "int16")
RawNE = spre.astype(RawNE, "int16")
RawNB = spre.astype(RawNB, "int16")
# preprocessed
PreNV = spre.astype(PreNV, "int16")
PreNS = spre.astype(PreNS, "int16")
PreNE = spre.astype(PreNE, "int16")
PreNB = spre.astype(PreNB, "int16")

# unit-test
assert RawNS.get_total_duration() == 10 * 60, "not 10 min"
assert RawNV.get_total_duration() == 10 * 60, "not 10 min"
assert RawNE.get_total_duration() == 10 * 60, "not 10 min"
assert RawNB.get_total_duration() == 10 * 60, "not 10 min"

# Select sites in cortex ***********************************
layers = ["L1", "L2_3", "L4", "L5", "L6"]

# vivo
sites_nv = RawNV.get_property("layers")
IN_CTX = np.isin(sites_nv, layers)
sites_nv = np.where(IN_CTX)[0]

# Biophy. spont.
sites_ns = RawNS.get_property("layers")
sites_ns = [
    "L2_3" if l_i == "L2" or l_i == "L3" else l_i for l_i in sites_ns
]
IN_CTX = np.isin(sites_ns, layers)
sites_ns = np.where(IN_CTX)[0]

# Biophy. evoked
sites_ne = RawNS.get_property("layers")
sites_ne = [
    "L2_3" if l_i == "L2" or l_i == "L3" else l_i for l_i in sites_ne
]
IN_CTX = np.isin(sites_ne, layers)
sites_ne = np.where(IN_CTX)[0]

# raw
raw_traces_nv = RawNV.get_traces()[:, sites_nv]
raw_traces_ns = RawNS.get_traces()[:, sites_ns]
raw_traces_ne = RawNE.get_traces()[:, sites_ne]
raw_traces_nb = RawNB.get_traces() # all sites are L5

# preprocessed
pre_traces_nv = PreNV.get_traces()[:, sites_nv]
pre_traces_ns = PreNS.get_traces()[:, sites_ns]
pre_traces_ne = PreNE.get_traces()[:, sites_ne]
pre_traces_nb = PreNB.get_traces() # all sites are L5

# Compute PSDs ************

# raw
out_raw_nv = get_welch_psd_marques_vivo_parallelized(raw_traces_nv)
out_raw_ns = get_welch_psd_marques_silico_parallelized(raw_traces_ns)
out_raw_ne = get_welch_psd_ne_parallelized(raw_traces_ne)
out_raw_nb = get_welch_psd_ne_parallelized(raw_traces_nb)

# preprocessed
out_prep_nv = get_welch_psd_marques_vivo_parallelized(pre_traces_nv)
out_prep_ns = get_welch_psd_marques_silico_parallelized(pre_traces_ns)
out_prep_ne = get_welch_psd_ne_parallelized(pre_traces_ne)
out_prep_nb = get_welch_psd_ne_parallelized(pre_traces_nb)

# save
save_psd_results(
    out_raw_ns, out_prep_ns, out_raw_nv, out_prep_nv,
    out_raw_ne, out_prep_ne, out_raw_nb, out_prep_nb
    )