"""sort james stimulus simulation with Kilosort 3.0
takes 30 min

  author: steeve.laquitaine@epfl.ch
    date: 22.04.2024
modified: 22.04.2024

usage:

    # Download Kilosort 2 release
    wget -c https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.0.tar.gz -O - | tar -xz

    # Submit to cluster
    sbatch cluster/sorting/others/spikewarp/sparse/ks2.sh

We do no preprocessing as Kilosort2 already preprocess the traces with
(see code preprocessDataSub())

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints
"""

import logging
import logging.config
import shutil
from time import time

import spikeinterface as si
import spikeinterface.sorters as ss
import yaml

from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_10m

# parameters (These parameters work for 1 hour recording
# if we reduce the number of recording channel sites by removing
# channels sites outside the cortex, preserving 202 sites)
sorter_params = {
        "detect_threshold": 6,
        "projection_threshold": [10, 5],
        "preclust_threshold": 8,
        "momentum": [20.0, 400.0],
        "car": True,
        "minFR": 0,
        "minfr_goodchannels": 0,
        "freq_min": 150,
        "sigmaMask": 30,
        "lam": 10.0,
        "nPCs": 3,
        "ntbuff": 64,
        "nfilt_factor": 4,
        "NT": 65792 * 2, # works for this batch size for 1 hour recording with 202 sites
        "AUCsplit": 0.9,
        "wave_length": 61,
        "keep_good_only": False,
        "skip_kilosort_preprocessing": False,
        "scaleproc": None,
        "save_rez_to_mat": False,
        "delete_tmp_files": ("matlab_files",),
        "delete_recording_dat": False,
    }

# SETUP CONFIG
exp = "others/spikewarp"
run = "2024_10_08"
sorter = "kilosort2"
cfg, _ = get_config(exp, run).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET KS 2 environment variable (you must have downloaded
# Kilosort 2 release, see usage above)
ss.Kilosort2Sorter.set_kilosort2_path(cfg["sorting"]["sorters"][sorter]["input"])

# sort
sort_and_postprocess_10m(cfg, sorter, sorter_params, duration_sec=600, is_sort=True, is_postpro=False, extract_wvf=False, copy_binary_recording=True, remove_bad_channels=True)

# postprocess
sort_and_postprocess_10m(cfg, sorter, sorter_params, duration_sec=600, is_sort=False, is_postpro=True, extract_wvf=True, copy_binary_recording=False, remove_bad_channels=False)