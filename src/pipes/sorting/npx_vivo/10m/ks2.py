"""sort buccino with Kilosort 2
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 08.08.2024

usage: 

    # Download Kilosort 2 release
    wget -c https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.0.tar.gz -O - | tar -xz

    # Submit to cluster
    sbatch cluster/sorting/npx_vivo/10m/ks2.sh
    
duration: ...
"""
import logging
import logging.config
import spikeinterface.sorters as ss
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_10m

# SET PARAMETERS
# were set to default parameter for KS 2 in spikeinterface 0.100.5
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
        "freq_min": 150,
        "sigmaMask": 30,
        "lam": 10.0,
        "nPCs": 3,
        "ntbuff": 64,
        "nfilt_factor": 4,
        "NT": None,
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
exp = "vivo_marques"
run = "c26"
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

# sort and postprocess
sort_and_postprocess_10m(cfg, sorter, sorter_params)