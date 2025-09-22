"""Computes power spectral densities of biophysically-simulated dense 
recording at depth 1

Uses multiprocessing on a single machine to speed up computations.

Usage:

    conda activate envs/spikebias

    # Biophysical simulations -----------------

    # depth 1
    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_dense_probe1 \
            --save-path dataset/01_intermediate/psds/psd_raw_dense_probe1.npy \
                --layers L1 L2_3 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_dense_probe1 \
            --save-path dataset/01_intermediate/psds/psd_prep_dense_probe1.npy \
                --preprocess True --freq-min 300 --layers L1 L2_3 > out_psds.log

                            
    # depth 2
    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_dense_probe2 \
            --save-path dataset/01_intermediate/psds/psd_raw_dense_probe2.npy \
                --layers L4 L5 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_dense_probe2 \
            --save-path dataset/01_intermediate/psds/psd_prep_dense_probe2.npy \
                --preprocess True --freq-min 300 --layers L4 L5 > out_psds.log
    
                
    # depth 3
    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_dense_probe3 \
            --save-path dataset/01_intermediate/psds/psd_raw_dense_probe3.npy \
                --layers L6 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_dense_probe3 \
            --save-path dataset/01_intermediate/psds/psd_prep_dense_probe3.npy \
                --preprocess True --freq-min 300 --layers L6 > out_psds.log

                
                
    Horvath et al. 2021 --------------

    # depth 1
    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_horvath_probe1 \
            --save-path dataset/01_intermediate/psds/psd_raw_horvath_probe1.npy \
                --layers L1 L2_3 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_horvath_probe1 \
            --save-path dataset/01_intermediate/psds/psd_prep_horvath_probe1.npy \
                --preprocess True --duration 2400 --freq-min 300 --layers L1 L2_3 > out_psds.log

                            
    # depth 2
    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_horvath_probe2 \
            --save-path dataset/01_intermediate/psds/psd_raw_horvath_probe2.npy \
                --layers L4 L5 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_horvath_probe2 \
            --save-path dataset/01_intermediate/psds/psd_prep_horvath_probe2.npy \
                --preprocess True --freq-min 300 --layers L4 L5 > out_psds.log
    
                
    # depth 3
    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_horvath_probe3 \
            --save-path dataset/01_intermediate/psds/psd_raw_horvath_probe3.npy \
                --layers L6 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiproc_dense \
        --recording-path dataset/00_raw/recording_horvath_probe3 \
            --save-path dataset/01_intermediate/psds/psd_prep_horvath_probe3.npy \
                --preprocess True --freq-min 300 --layers L6 > out_psds.log

Returns:
    (.npy): writes power spectral densities

Execution time: 1:30 min

Required:
- >120 GB RAM
- PSD file output: 4 MB

Tested on * Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K @3.2 GHz/5.8 GHz)
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
import yaml
import logging
import logging.config
import time
import argparse

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

# SETUP PARAMETERS
SF = 20000         # voltage trace sampling frequency
GAIN_TO_UV = 0.195 # gain to uV conversion gain

# SETUP WELCH PSD PARAMETERS *******************
FILT_WINDOW = "hann"

# vivo
FILT_WIND_SIZE = SF # 1Hz freq. resolution
FILT_WIND_OVERLAP = int(
    FILT_WIND_SIZE // 1.5
)


def get_welch_psd_parallelized(traces: np.ndarray):
    """compute power spectrum density
    using parallel computing

    Args:
        traces (np.ndarray): timepoints x sites voltage traces

    Returns:
        dict: frequencies x sites powers array and frequencies array
    """
    # takes 2 min (instead of 70 min w/o multiprocessing)
    nsites = traces.shape[1]

    # compute power for each site trace
    # in parallel with a pool of workers
    with ProcessPoolExecutor() as executor:
        power_by_site = executor.map(
            get_site_welch_psd,
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


def get_site_welch_psd(trace, site):
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
        SF,
        window=FILT_WINDOW,
        nperseg=FILT_WIND_SIZE,
        noverlap=FILT_WIND_OVERLAP,
    )
    return np.array(power), np.array(freq)


def save_psd(data, write_path: str):
    parent_path = os.path.dirname(write_path)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(write_path, data)
    
    
if __name__ == "__main__":
    """Entry point
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute psds")
    parser.add_argument("--recording-path", default= './dataset/00_raw/recording_dense_probe1', help="recording path.")
    parser.add_argument("--save-path", default='./dataset/01_intermediate/psds/psd_prep_dense_probe1.npy', help="psd save path")
    parser.add_argument("--preprocess", type=bool, default=False, help="apply highpass filtering and common referencing")
    parser.add_argument("--duration", type=int, default=3600, help="max recording duration in seconds for preprocessing (cheaper)")
    parser.add_argument("--freq-min", type=int, default=300, help="high pass filter cutoff")
    parser.add_argument("--layers", nargs='+', help="list of layers to analyse")    
    args = parser.parse_args()

    # report parameters for visual check
    logger.info(f"recording_path: {args.recording_path}")
    logger.info(f"preprocess: {args.preprocess}")
    logger.info(f"duration: {args.duration}")
    logger.info(f"freq_min: {args.freq_min}")
    logger.info(f"save_path: {args.save_path}")
    logger.info(f"layers: {args.layers}")

    # Load datasets
    t0 = time.time()
    logger.info(f"Started pipeline")

    # load
    Recording = si.load_extractor(args.recording_path)
    logger.info(f"Recording info: {Recording}")

    # convert to uV (consistently with Horvath 2021)
    Recording.set_channel_gains(GAIN_TO_UV)
    logger.info("Converted to uV done")

    # preprocess
    if args.preprocess:
        if args.duration < Recording.get_total_duration():
            Recording = Recording.frame_slice(0, Recording.get_sampling_frequency() * args.duration)
        Recording = spre.highpass_filter(Recording, freq_min=args.freq_min)
        Recording = spre.common_reference(Recording, reference="global", operator="median")
        logger.info(f"Recording for preprocessing: {Recording}")
        logger.info("High-pass filtering and referencing done")

    # compress from floats to integers
    Recording = spre.astype(Recording, "int16")
    logger.info("Compressing done")

    # select sites in cortex    
    sites = Recording.get_property("layers")
    sites = [
        "L2_3" if l_i == "L2" or l_i == "L3" else l_i for l_i in sites
    ]
    sites = np.where(np.isin(sites, args.layers))[0]

    # get traces
    traces_uV = Recording.get_traces(return_scaled=True)

    #- Remove DC component by subtracting the mean
    traces = demean(traces_uV[:, sites])
    logger.info("Detrending done")

    # compute psd
    out_raw = get_welch_psd_parallelized(traces)
    logger.info("Computing PSDs done")

    # save
    save_psd(out_raw, args.save_path)
    logger.info(f"Psds saved in {args.save_path}")
    logger.info(f"Completed in {np.round(time.time()-t0,2)} secs")
