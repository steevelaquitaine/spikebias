"""sort marques silico with Kilosort 2.5
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 30.01.2024
modified: 30.01.2024

usage: 

    # Download Kilosort 2.5 release
    wget -c https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.5.tar.gz -O - | tar -xz

    # Submit to cluster
    sbatch cluster/sorting/others/spikewarp/ks2_5.sh

note:
    only works if drift correction is disabled (nblocks=0), else CUDA_ERROR_ILLEGAL_ADDRESS
"""
import logging
import logging.config
import shutil
from time import time
import spikeinterface as si
import spikeinterface.sorters as ss
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_10m

# SET PARAMETERS
# were set to default parameter for KS 2.5
# found in spikeinterface 0.96.1
sorter_params = {
        'detect_threshold': 6,              # Threshold for spike detection
        'projection_threshold': [10, 4],    # Threshold on projections
        'preclust_threshold': 8,            # Threshold crossings for pre-clustering (in PCA projection space)
        'car': True,                        # Enable or disable common reference
        'minFR': 0.1,                       # Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
        'minfr_goodchannels': 0.1,          # Minimum firing rate on a 'good' channel
        'nblocks': 0, # 5,                  # blocks for registration. 0 turns it off, 1 does rigid registration. Replaces 'datashift' option.
        'sig': 20,
        'freq_min': 150,
        'sigmaMask': 30,
        'nPCs': 3,
        'ntbuff': 64,
        'nfilt_factor': 4,
        'NT': None,
        'do_correction': True,
        'wave_length': 61,
        'keep_good_only': False,
    }

# SETUP CONFIG
exp = "others/spikewarp"
run = "2024_04_13"
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

# sort
sort_and_postprocess_10m(cfg, sorter, sorter_params, duration_sec=600, is_sort=True, is_postpro=False, extract_wvf=False, copy_binary_recording=True, remove_bad_channels=True)

# postprocess
sort_and_postprocess_10m(cfg, sorter, sorter_params, duration_sec=600, is_sort=False, is_postpro=True, extract_wvf=True, copy_binary_recording=False, remove_bad_channels=False)