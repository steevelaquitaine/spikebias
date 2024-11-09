
"""sort 20 min of marques silico with Kilosort 3.0
takes 60 min (with saving, else 20 min)

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 07.05.2024

usage:

    sbatch cluster/sorting/others/spikewarp/sparse/ks.sh

Note:
    - We do no preprocessing as Kilosort already preprocess the traces with
    (see code preprocessDataSub()):
    - we set minFR and minfr_goodchannels to 0

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: 
- to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints
- crashes with "Maximum variable size allowed on the device is exceeded." with "minFR"
and "minfr_goodchannels" set to 0.

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

# SET PARAMETERS
sorter_params = {
    "detect_threshold": 6,
    "car": True,
    "useGPU": True,
    "freq_min": 300,
    "freq_max": 6000,
    "ntbuff": 64,
    "Nfilt": None,
    "NT": None,
    "wave_length": 61,
    "delete_tmp_files": ("matlab_files",),
    "delete_recording_dat": False,
}

# SETUP CONFIG
exp = "others/spikewarp"
run = "2024_10_08"
sorter = "kilosort"
cfg, _ = get_config(exp, run).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET KS environment variable (you must have downloaded
# Kilosort release, see usage above)
ss.KilosortSorter.set_kilosort_path(cfg["sorting"]["sorters"][sorter]["input"])

# sort
sort_and_postprocess_10m(cfg, sorter, sorter_params, duration_sec=600, is_sort=True, is_postpro=False, extract_wvf=False, copy_binary_recording=True, remove_bad_channels=True)

# postprocess
sort_and_postprocess_10m(cfg, sorter, sorter_params, duration_sec=600, is_sort=False, is_postpro=True, extract_wvf=True, copy_binary_recording=False, remove_bad_channels=False)