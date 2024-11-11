"""sort and postprocess

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 13.08.2024

usage: 

    # Submit to cluster
    sbatch cluster/sorting/dense_biophy/probe_3/10m/ks2_5.sh
    
duration:
"""
import logging
import logging.config
import spikeinterface.sorters as ss
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_10m

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
    "minFR": 0,
    "minfr_goodchannels": 0,
    "nblocks": 5,
    "sig": 20,
    "freq_min": 150,
    "sigmaMask": 30,
    "lam": 10.0,
    "nPCs": 3,
    "ntbuff": 64,
    "nfilt_factor": 4,
    "NT": None,
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
exp = "silico_horvath"
run = "concatenated/probe_3"
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

# sort and postprocess from existing int16 binary recording
sort_and_postprocess_10m(cfg,
                         sorter,
                         sorter_params,
                         duration_sec=600,
                         is_sort=True,
                         is_postpro=True,
                         copy_binary_recording=True
                         )