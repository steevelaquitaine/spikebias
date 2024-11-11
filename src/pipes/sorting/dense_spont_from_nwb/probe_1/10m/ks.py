"""sort

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 13.08.2024

usage: 

    # Submit to cluster
    sbatch cluster/sorting/dense_biophy/probe_1/10m/ks.sh

takes ...

"""
import logging
import logging.config
import spikeinterface.sorters as ss
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_10m

# SET PARAMETERS
# were set to default parameter for KS
# found in spikeinterface 0.100.5
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
exp = "dense_spont_from_nwb"
run = "probe_1"
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