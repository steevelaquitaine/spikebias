"""sort buccino with Kilosort 2.5
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 11.02.2024

usage: 

    # Download Kilosort 2.5 release
    wget -c https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.5.tar.gz -O - | tar -xz

    # Submit to cluster
    sbatch cluster/sorting/npx_evoked/full/ks2_5.sh
    
duration:
"""
import logging
import logging.config
import spikeinterface.sorters as ss
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_40m

## SET PARAMETERS
# were set to default parameter for KS 2.5 in spikeinterface 0.100.5
# to capture as many units as possible we set:
# - "minFR"=0 instead of 0.1,
# - "minfr_goodchannels"=0 instead of 0.1
sorter_params = {
    "detect_threshold": 6,
    "projection_threshold": [10, 4],
    "preclust_threshold": 8,
    "momentum": [20.0, 400.0],
    "car": True,
    "minFR": 0.1, #-> solved Maximum variable size allowed on the device is exceeded
    "minfr_goodchannels": 0.1, #-> solved Maximum variable size allowed on the device is exceeded
    "nblocks": 1, # solved CUDA_ERROR_ILLEGAL_ADDRESS for long recording (default 5)
    "sig": 20,
    "freq_min": 150,
    "sigmaMask": 30,
    "lam": 10.0,
    "nPCs": 3,
    "ntbuff": 64,
    "nfilt_factor": 4,
    "NT": 10000, # None, -> does not solve Maximum variable size allowed on the device is exceeded
    "AUCsplit": 0.9,
    "do_correction": True,
    "wave_length": 61,
    "keep_good_only": False,
    "skip_kilosort_preprocessing": False,
    "scaleproc": None,
    "save_rez_to_mat": False,
    "delete_tmp_files": ("matlab_files",),
    "delete_recording_dat": False,
}

# SETUP CONFIG
exp = "silico_neuropixels"
run = "stimulus"
sorter = "kilosort2_5"
cfg, _ = get_config(exp, run).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET KS 2 environment variable (you must have downloaded
# Kilosort 2 release, see usage above)
ss.Kilosort2_5Sorter.set_kilosort2_5_path(cfg["sorting"]["sorters"][sorter]["input"])

# sort and postprocess
sort_and_postprocess_40m(cfg, sorter, sorter_params, is_sort=False, is_postpro=True, copy_binary_recording=False, remove_bad_channels=True)