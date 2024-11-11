"""sort and postprocess

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 13.08.2024
features:
- duration in sec as arg (4 min only works)
- postprocessing or not as arg
- sorting or not as arg
- copy int16 binary recording or not to speed up sorters as arg

usage:

    # Submit to cluster
    sbatch cluster/sorting/dense_biophy/probe_1/10m/ks2.sh

duration: ??

Parameters:
- we set preserved the default parameters for KS2 in spikeinterface 0.100.5
but to sort as many units as possible we set "minFR" and "minfr_goodchannels"
to 0 instead of 0.1.
"""
import logging
import logging.config
import spikeinterface.sorters as ss
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_10m

# parameters
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
        "NT": 65792,
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
exp = "silico_horvath"
run = "concatenated/probe_1"
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