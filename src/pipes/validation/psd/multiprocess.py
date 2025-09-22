"""Computes voltage traces power spectral densities

author: laquitainesteeve@gmail.com 

Uses multiprocessing on a single machine to speed up computations.

Usage:

    # activate virtual environment
    conda activate envs/spikebias

    

    # ----------------- Dense recordings -----------------

    # depth 1 (raw and preprocessed)
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_dense_probe1 --save-path dataset/01_intermediate/psds/psd_raw_dense_probe1.npy \
            --gain-to-uv 0.195 --duration 2400 --layers L1 L2_3 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_dense_probe1 --save-path dataset/01_intermediate/psds/psd_prep_dense_probe1_cutoff_300.npy \
            --gain-to-uv 0.195 --duration 2400 --preprocess True --freq-min 300 --layers L1 L2_3 > out_psds.log

                            
    # depth 2
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_dense_probe2 --save-path dataset/01_intermediate/psds/psd_raw_dense_probe2.npy \
            --gain-to-uv 0.195 --duration 2400 --layers L4 L5 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_dense_probe2 --save-path dataset/01_intermediate/psds/psd_prep_dense_probe2_cutoff_300.npy \
            --gain-to-uv 0.195 --duration 2400 --preprocess True --freq-min 300 --layers L4 L5 > out_psds.log
    
                
    # depth 3
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_dense_probe3 --save-path dataset/01_intermediate/psds/psd_raw_dense_probe3.npy \
                --gain-to-uv 0.195 --duration 2400 --layers L6 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_dense_probe3 --save-path dataset/01_intermediate/psds/psd_prep_dense_probe3_cutoff_300.npy \
            --gain-to-uv 0.195 --duration 2400 --preprocess True --freq-min 300 --layers L6 > out_psds.log

    # Horvath depth 1
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_horvath_probe1 --save-path dataset/01_intermediate/psds/psd_raw_horvath_probe1.npy \
            --gain-to-uv 0.195 --duration 2400 --layers L1 L2_3 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_horvath_probe1 --save-path dataset/01_intermediate/psds/psd_prep_horvath_probe1_cutoff_300.npy \
            --gain-to-uv 0.195 --duration 2400 --preprocess True --freq-min 300 --layers L1 L2_3 > out_psds.log

                            
    # depth 2
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_horvath_probe2 --save-path dataset/01_intermediate/psds/psd_raw_horvath_probe2.npy \
            --gain-to-uv 0.195 --layers L4 L5 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_horvath_probe2 --save-path dataset/01_intermediate/psds/psd_prep_horvath_probe2_cutoff_300.npy \
            --gain-to-uv 0.195 --duration 2400 --preprocess True --freq-min 300 --layers L4 L5 > out_psds.log
    
                
    # depth 3
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_horvath_probe3 --save-path dataset/01_intermediate/psds/psd_raw_horvath_probe3.npy \
            --gain-to-uv 0.195 --duration 2400 --layers L6 > out_psds.log

    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_horvath_probe3 --save-path dataset/01_intermediate/psds/psd_prep_horvath_probe3_cutoff_300.npy \
            --gain-to-uv 0.195 --duration 2400 --preprocess True --freq-min 300 --layers L6 > out_psds.log

                

    # ----------------- Neuropixels ---------------------------------------------------

    # npx-spont (10 min are 9 GB)
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_npx_spont --save-path dataset/01_intermediate/psds/psd_prep_npx_spont_cutoff_100.npy \
            --duration 600 --layers L1 L2_3 L4 L5 L6 --gain-to-uv 1 --preprocess True --freq-min 100 --filter_window hann > out_psds.log

    # npx-evoked
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_npx_evoked --save-path dataset/01_intermediate/psds/psd_prep_npx_evoked_cutoff_100.npy \
            --duration 600 --layers L1 L2_3 L4 L5 L6 --gain-to-uv 1 --preprocess True --freq-min 100 --filter_window hann > out_psds.log

    # marques-smith 
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_marques_smith --save-path dataset/01_intermediate/psds/psd_prep_marques_smith_cutoff_100.npy \
            --duration 600 --layers L1 L2_3 L4 L5 L6 --gain-to-uv 1 --preprocess True --freq-min 100 --filter_window hann > out_psds.log

    # synthetic (Buccino rep)
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_buccino_rep --save-path dataset/01_intermediate/psds/psd_raw_buccino.npy \
            --duration 600 --keep-first-n-sites 200 --gain-to-uv 1 > out_psds.log

    # synthetic (Buccino)
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_buccino_rep --save-path dataset/01_intermediate/psds/psd_prep_buccino_cutoff_100.npy \
            --duration 600 --keep-first-n-sites 200 --gain-to-uv 1 --preprocess True --freq-min 100 --filter_window hann > out_psds.log

            
    # with a 300 Hz cutoff
    nohup python -m src.pipes.validation.psd.multiprocess \
        --recording-path dataset/00_raw/recording_buccino --save-path dataset/01_intermediate/psds/psd_prep_buccino_cutoff_300.npy \
            --duration 600 --keep-first-n-sites 200 --gain-to-uv 1 --preprocess True --freq-min 300 --filter_window hann > out_psds.log
            
                
            
Returns:
    (.npy): writes power spectral densities

Execution time: 1:30 min

Required:

- >170 GB RAM, which corresponds to a max Recording size of 11.44 GiB on my machine
- output power spectrum file: 4 MB

Tested on an Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K @3.2 GHz/5.8 GHz)
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
from scipy import signal
import yaml
import logging
import logging.config
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial


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



def get_welch_psd_parallelized(traces: np.ndarray, sfreq: int, filter_wind: str):
    """compute power spectrum density
    using parallel computing

    Args:
        traces (np.ndarray): timepoints x sites voltage traces

    Returns:
        dict: frequencies x sites powers array and frequencies array
    """
    # create a function that take sampling frequency as an argument
    get_site_welch_psd_with_sfreq = partial(get_site_welch_psd, sfreq=sfreq, filter_wind=filter_wind)

    # takes 2 min (instead of 70 min w/o multiprocessing)
    nsites = traces.shape[1]

    # compute power for each site trace
    # in parallel with a pool of workers
    with ProcessPoolExecutor() as executor:
        power_by_site = executor.map(
            get_site_welch_psd_with_sfreq,
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


def get_site_welch_psd(trace, site, sfreq, filter_wind):
    """calculate the welch frequency powers in the input trace

    Args:
        traces (np.ndarray): timepoints x sites voltage trace
        site: silent, automatically generated by ProcessPoolExecutor()
        sfreq (_type_): voltage trace sampling frequency
        

    Returns:
        _type_: _description_
    """
    (freq, power) = signal.welch(
        trace,
        sfreq,
        window=filter_wind,
        nperseg=sfreq, # filter window size
        noverlap=int(sfreq // 1.5)
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
    parser.add_argument("--gain-to-uv", type=float, default=1, help="gain to uV conversion factor")
    parser.add_argument("--preprocess", type=bool, default=False, help="apply highpass filtering and common referencing")
    parser.add_argument("--duration", type=int, default=3600, help="max recording duration in seconds for preprocessing (cheaper)")
    parser.add_argument("--freq-min", type=int, default=300, help="high pass filter cutoff")
    parser.add_argument("--layers", nargs='+', help="list of layers to analyse")
    parser.add_argument("--keep-first-n-sites", type=int, help="number of sites to keep from the first to N-th site")
    parser.add_argument("--filter_window", type=str, default="hann", help="welch psd filter window")

    args = parser.parse_args()

    # report parameters for visual check
    logger.info(f"recording_path: {args.recording_path}")
    logger.info(f"gain_to_uv: {args.gain_to_uv}")
    logger.info(f"preprocess: {args.preprocess}")
    logger.info(f"duration: {args.duration}")
    logger.info(f"freq_min: {args.freq_min}")
    logger.info(f"save_path: {args.save_path}")
    logger.info(f"layers: {args.layers}")
    logger.info(f"keep_first_n_sites: {args.keep_first_n_sites}")
    logger.info(f"filter_window: {args.filter_window}")

    # Load datasets
    t0 = time.time()
    logger.info(f"Started pipeline")

    # load
    Recording = si.load_extractor(args.recording_path)
    logger.info(f"Recording info: {Recording}")

    # get sampling frequency
    sfreq = Recording.get_sampling_frequency()

    # convert to uV (consistently with Horvath 2021)
    Recording.set_channel_gains(args.gain_to_uv)
    logger.info("Converted to uV done")

    # keep electrodes in cortex
    if args.layers:

        sites = Recording.get_property("layers")
        sites = [
            "L2_3" if l_i == "L2" or l_i == "L3" else l_i for l_i in sites
        ]
        sites = np.where(np.isin(sites, args.layers))[0]
        site_ids = Recording.channel_ids[sites]
        site_ids_to_remove = Recording.get_channel_ids()[~np.isin(Recording.get_channel_ids(), site_ids)]
        logger.info("Keeping site ids in specified layers")

    elif args.keep_first_n_sites:

        site_ids = Recording.get_channel_ids()[np.arange(0, args.keep_first_n_sites, 1)]
        site_ids_to_remove = Recording.get_channel_ids()[~np.isin(Recording.get_channel_ids(), site_ids)]
        logger.info("Setting specified site ids to keep")

    # remove irrelevant sites (eases processing)
    Recording = Recording.remove_channels(site_ids_to_remove)
    logger.info(f"Recording after site curation: {Recording}")        

    # preprocess
    if args.preprocess:
        
        # get recording duration
        if args.duration < Recording.get_total_duration():
            Recording = Recording.frame_slice(0, Recording.get_sampling_frequency() * args.duration)

        # filter
        Recording = spre.highpass_filter(Recording, freq_min=args.freq_min)

        # apply referencing
        Recording = spre.common_reference(Recording, reference="global", operator="median")
        logger.info(f"Recording for preprocessing: {Recording}")
        logger.info("High-pass filtering and referencing done")

    # compress to integers
    Recording = spre.astype(Recording, "int16")
    logger.info("Compressing done")

    # get traces (bottleneck)
    traces_uV = Recording.get_traces(return_scaled=True)

    # - Remove DC component by subtracting the mean
    #traces = demean(traces_uV[:, sites])
    traces = demean(traces_uV)
    logger.info("Detrending done")

    # compute psd    
    out_raw = get_welch_psd_parallelized(traces, sfreq, args.filter_window)
    logger.info("Computing PSDs done")

    # save
    save_psd(out_raw, args.save_path)
    logger.info(f"Psds saved in {args.save_path}")
    logger.info(f"Completed in {np.round(time.time()-t0,2)} secs")
