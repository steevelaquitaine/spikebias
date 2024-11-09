"""sort and postprocess 10 min of spontaneous with Kilosort 4.0

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 08.08.2024

usage:

    sbatch cluster/sorting/npx_spont/10m/ks4.sh

Note:
    - We do no preprocessing as Kilosort4 already preprocess the traces with
    (see code preprocessDataSub()):
    - we set minFR and minfr_goodchannels to 0
    
duration:
    sorting (21 mins) + postprocessing (16 mins)

% 1) conversion to int16;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: 
- to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints
- crashes with "Maximum variable size allowed on the device is exceeded." with "minFR"
and "minfr_goodchannels" set to 0.
- batch_site must be 10000. Below tensor shape error. Above CUDA memory error.

We set to the default parametersfor spikeinterface 0.100.5
note that there are no minFR and minFR_channels in ks4
- we set batch_size to 10,000 due to memory constrains
- we set dminx to 25.6 um
"""
import logging
import logging.config
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_10m

# SET PARAMETERS
# these are the default parameters
# for spikeinterface 0.100.5
# note that there are no minFR and minFR_channels in ks4
# - we set batch_size to 10,000 instead of 60,0000 due to memory constrains
# - we set dminx to 25.6 um instead of None
sorter_params = {
    "batch_size": 10000,
    "nblocks": 1,
    "Th_universal": 9,
    "Th_learned": 8,
    "do_CAR": True,
    "invert_sign": False,
    "nt": 61,
    "artifact_threshold": None,
    "nskip": 25,
    "whitening_range": 32,
    "binning_depth": 5,
    "sig_interp": 20,
    "nt0min": None,
    "dmin": None,
    "dminx": 25.6,
    "min_template_size": 10,
    "template_sizes": 5,
    "nearest_chans": 10,
    "nearest_templates": 100,
    "templates_from_data": True,
    "n_templates": 6,
    "n_pcs": 6,
    "Th_single_ch": 6,
    "acg_threshold": 0.2,
    "ccg_threshold": 0.25,
    "cluster_downsampling": 20,
    "cluster_pcs": 64,
    "duplicate_spike_bins": 15,
    "do_correction": True,
    "keep_good_only": False,
    "save_extra_kwargs": False,
    "skip_kilosort_preprocessing": False,
    "scaleproc": None,
}

# SETUP CONFIG
exp = "silico_neuropixels"
run = "concatenated"
sorter = "kilosort4"
cfg, _ = get_config(exp, run).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# sort and postprocess
sort_and_postprocess_10m(cfg, sorter, sorter_params)