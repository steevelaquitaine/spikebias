"""Pipeline to compute data for power spectral density for the first 10 
minutes of neuropixels traces

Method: 

* Welch method
* Buttwerworth temporal filtering
* entire duration of the recordings

Duration: 5 mins

Usage:

    python src/pipes/validation/psd/multiproc_npx_full.py

Returns:
    (.npy): writes power spectral densities 
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import spikeinterface as si
from concurrent.futures import ProcessPoolExecutor
import spikeinterface.preprocessing as spre
from scipy import signal
import time 
import yaml
import logging
import logging.config


# move to PROJECT PATH
PROJ_PATH = '/home/steeve/steeve/epfl/code/spikebias/'
os.chdir(PROJ_PATH)

# add custom package to path
sys.path.append('.')

from src.nodes.utils import demean

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# setup load paths
RAW_PATH_NV = os.path.join(PROJ_PATH, "dataset/00_raw/recording_marques_smith/")
RAW_PATH_NS = os.path.join(PROJ_PATH, "dataset/00_raw/recording_npx_spont/")
RAW_PATH_NE = os.path.join(PROJ_PATH, "dataset/00_raw/recording_npx_evoked/")
RAW_PATH_NB = os.path.join(PROJ_PATH, "dataset/00_raw/recording_buccino/")
RAW_PATH_hv1 = os.path.join(PROJ_PATH, "dataset/00_raw/recording_horvath_probe1/")
RAW_PATH_hv2 = os.path.join(PROJ_PATH, "dataset/00_raw/recording_horvath_probe2/")
RAW_PATH_hv3 = os.path.join(PROJ_PATH, "dataset/00_raw/recording_horvath_probe3/")
RAW_PATH_hs1 = os.path.join(PROJ_PATH, "dataset/00_raw/recording_dense_probe1/")
RAW_PATH_hs2 = os.path.join(PROJ_PATH, "dataset/00_raw/recording_dense_probe2/")
RAW_PATH_hs3 = os.path.join(PROJ_PATH, "dataset/00_raw/recording_dense_probe3/")

# setup save paths
RAW_PSD_PATH_NV = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_marques_smith.npy")
RAW_PSD_PATH_NS = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_npx_spont.npy")
RAW_PSD_PATH_NE = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_npx_evoked.npy")
RAW_PSD_PATH_NB = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_npx_buccino.npy")
RAW_PSD_PATH_HV1 = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_horvath_probe1.npy")
RAW_PSD_PATH_HV2 = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_horvath_probe2.npy")
RAW_PSD_PATH_HV3 = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_horvath_probe3.npy")
RAW_PSD_PATH_HS1 = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_dense_probe1.npy")
RAW_PSD_PATH_HS2 = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_dense_probe2.npy")
RAW_PSD_PATH_HS3 = os.path.join(PROJ_PATH, "dataset/01_intermediate/psd_raw_dense_probe3.npy")

# layers    
layers = ["L1", "L2_3", "L4", "L5", "L6"]
            
# sampling frequency
SF_NV = 30000  # Marques-Smith
SF_NS = 40000  # Biophy. spontaneous
SF_NE = 20000  # Biophy. evoked
SF_NB = 32000  # Synthetic model (Buccino et al., 2020)

# SETUP WELCH PSD PARAMETERS

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


def save_psd(data, write_path:str):
    parent_path = os.path.dirname(write_path)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(write_path, data)


def main():

    # Load datasets
    t0 = time.time()

    logger.info(f"Started pipeline..")

    RawNV = si.load_extractor(RAW_PATH_NV)
    RawNV = spre.astype(RawNV, "int16")
    
    # Select sites in cortex
    sites_nv = RawNV.get_property("layers")
    IN_CTX = np.isin(sites_nv, layers)
    sites_nv = np.where(IN_CTX)[0]
    
    # Remove the DC component by subtracting the means
    raw_traces_nv = demean(RawNV.get_traces()[:, sites_nv])        
    
    # calculate PSD
    out_raw_nv = get_welch_psd_marques_vivo_parallelized(raw_traces_nv)
    
    # save Marques-Smith (vivo)
    save_psd(out_raw_nv, RAW_PSD_PATH_NV)
    # #save_psd(out_prep_nv, PRE_PSD_PATH_NV)        

    # logger.info(f"Saved PSD for Marques-Smith.")

    # # spontaneous neuropixels -------------------

    # RawNS = si.load_extractor(RAW_PATH_NS)
    # #PreNS = si.load_extractor(PRE_PATH_NS)
    # RawNS = spre.astype(RawNS, "int16")
    # #PreNS = spre.astype(PreNS, "int16")
    
    # # Select sites in cortex
    # sites_ns = RawNS.get_property("layers")
    # sites_ns = [
    #     "L2_3" if l_i == "L2" or l_i == "L3" else l_i for l_i in sites_ns
    # ]
    # IN_CTX = np.isin(sites_ns, layers)
    # sites_ns = np.where(IN_CTX)[0]
    
    # # Remove the DC component by subtracting the means
    # raw_traces_ns = demean(RawNS.get_traces()[:, sites_ns])
    # #pre_traces_ns = demean(PreNS.get_traces()[:, sites_ns])
    
    # # Calculate psd
    # out_raw_ns = get_welch_psd_marques_silico_parallelized(raw_traces_ns)
    # #out_prep_ns = get_welch_psd_marques_silico_parallelized(pre_traces_ns)
    
    # # save
    # save_psd(out_raw_ns, RAW_PSD_PATH_NS)
    # #save_psd(out_prep_ns, PRE_PSD_PATH_NS)

    # # evoked neuropixels -------------------
    
    # RawNE = si.load_extractor(RAW_PATH_NE)
    # #PreNE = si.load_extractor(PRE_PATH_NE)
    # RawNE = spre.astype(RawNE, "int16")
    # #PreNE = spre.astype(PreNE, "int16")
    
    # # select sites in cortex
    # sites_ne = RawNE.get_property("layers")
    # sites_ne = [
    #     "L2_3" if l_i == "L2" or l_i == "L3" else l_i for l_i in sites_ne
    # ]
    # IN_CTX = np.isin(sites_ne, layers)
    # sites_ne = np.where(IN_CTX)[0]
    
    # # Remove the DC component by subtracting the means
    # raw_traces_ne = demean(RawNE.get_traces()[:, sites_ne])
    # #pre_traces_ne = demean(PreNE.get_traces()[:, sites_ne])
    
    # # calculate psd
    # out_raw_ne = get_welch_psd_ne_parallelized(raw_traces_ne)
    # #out_prep_ne = get_welch_psd_ne_parallelized(pre_traces_ne)

    # # save
    # save_psd(out_raw_ne, RAW_PSD_PATH_NE)
    # #save_psd(out_prep_ne, PRE_PSD_PATH_NE)
        
    # # Buccino's model -------------------
    
    # RawNB = si.load_extractor(RAW_PATH_NB)
    # #PreNB = si.load_extractor(PRE_PATH_NB)
    # RawNB = spre.astype(RawNB, "int16")
    # #PreNB = spre.astype(PreNB, "int16")
    
    # # Remove the DC component by subtracting the means
    # raw_traces_nb = demean(RawNB.get_traces())
    # #pre_traces_nb = demean(PreNB.get_traces())
    
    # # calculate psd
    # out_raw_nb = get_welch_psd_nb_parallelized(raw_traces_nb)
    # #out_prep_nb = get_welch_psd_nb_parallelized(pre_traces_nb)

    # # save
    # save_psd(out_raw_nb, RAW_PSD_PATH_NB)
    # #save_psd(out_prep_nb, PRE_PSD_PATH_NB)

    # logger.info(f"Completed in {np.round(time.time()-t0,2)} secs")
    # logger.info("PSD data written.")

# run
if __name__== "__main__":
    main()