"""sort buccino with Kilosort 3.0
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 15.01.2023
modified: 08.08.2023

usage: 

    sbatch cluster/sorting/npx_synth/ks3.sh

We do no preprocessing as Kilosort3 already preprocess the traces with 
(see code preprocessDataSub()):

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints

sorter_params:
- we used the default parameters for spikeinterface 0.100.5
But to capture as many units as possible we set:
- minFR=0 instead of 0.2
- minfr_goodchannels=0 instead of 0.2
- batch size was set to 65,792 for memory constrains
We kept default filtering cutoff at 300 Hz as in Pachitariu et al., 2024
"""
import logging
import logging.config
import yaml
import spikeinterface.sorters as ss

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_full

# SET PARAMETERS
# default parameters for spikeinterface 0.100.5.
# to capture as many units as possible we set
# - minFR=0 instead of 0.2
# - minfr_goodchannels=0 instead of 0.2
# we kept default filtering cutoff at 300 Hz as in Pachitariu et al., 2024
sorter_params = {
    "detect_threshold": 6,
    "projection_threshold": [9, 9],
    "preclust_threshold": 8,
    "car": True,
    "minFR": 0, # modified
    "minfr_goodchannels": 0, # modified
    "nblocks": 5,
    "sig": 20,
    "freq_min": 300,
    "sigmaMask": 30,
    "lam": 20.0,
    "nPCs": 3,
    "ntbuff": 64,
    "nfilt_factor": 4,
    "do_correction": True,
    "NT": 65792, # modified
    "AUCsplit": 0.8,
    "wave_length": 61,
    "keep_good_only": False,
    "skip_kilosort_preprocessing": False,
    "scaleproc": None,
    "save_rez_to_mat": False,
    "delete_tmp_files": ("matlab_files",),
    "delete_recording_dat": False,
}

# SETUP CONFIG
exp = "buccino_2020"
run = "2020"
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
sort_and_postprocess_full(cfg, sorter, sorter_params)