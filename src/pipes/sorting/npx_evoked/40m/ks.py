"""sort buccino with Kilosort
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 08.08.2024

usage: 

    # Download Kilosort release
    git clone https://github.com/cortex-lab/KiloSort.git

    # Submit to cluster
    sbatch cluster/sorting/npx_evoked/full/ks.sh

takes ...

"""
import logging
import logging.config
import spikeinterface.sorters as ss
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_full

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
exp = "silico_neuropixels"
run = "stimulus"
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

# sort and postprocess
sort_and_postprocess_full(cfg, sorter, sorter_params, is_sort=False, is_postpro=True, copy_binary_recording=False, remove_bad_channels=True)