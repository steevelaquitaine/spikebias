"""sort james stimulus simulation with Kilosort 3.0
takes 30 min

  author: steeve.laquitaine@epfl.ch
    date: 22.04.2024
modified: 22.04.2024

usage:

    sbatch cluster/sorting/others/spikewarp/ks3.sh

We do no preprocessing as Kilosort3 already preprocess the traces with
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

# SET PARAMETERS
sorter_params = {
    'detect_threshold': 6, # 6      # spike detection thresh.
    'projection_threshold': [9, 9], # template-projected detection thresh.
    'preclust_threshold': 8,
    'car': True,
    'minFR': 0.2,
    'minfr_goodchannels': 0.2,
    'nblocks': 5,
    'sig': 20,
    'freq_min': 300,                # high-pass filter cutoff
    'sigmaMask': 30,
    'nPCs': 3,
    'ntbuff': 64,
    'nfilt_factor': 4,
    'do_correction': True,
    'NT': 65792, # None,            # Batch size
    'wave_length': 61,
    'keep_good_only': False
    }


# SETUP CONFIG
exp = "others/spikewarp"
run = "2024_04_13"
sorter = "kilosort3"
cfg, _ = get_config(exp, run).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(cfg["sorting"]["sorters"][sorter]["input"])

# sort and postprocess
sort_and_postprocess_10m(cfg,
                         sorter,
                         sorter_params,
                         duration_sec=600,
                         is_sort=False,
                         is_postpro=True,
                         copy_binary_recording=False)