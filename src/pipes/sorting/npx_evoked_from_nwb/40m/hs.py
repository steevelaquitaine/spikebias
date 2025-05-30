"""sort buccino with Kilosort
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 08.08.2024

usage:

    # install
    pip install herdingspikes

    # Submit to cluster
    sbatch cluster/sorting/npx_evoked/full/hs.sh

takes 51 minutes

"""
import logging
import logging.config
import yaml

# custom package
from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_full

# SET PARAMETERS
# were set to default parameter for KS
# found in spikeinterface 0.100.5
sorter_params = {
    # core params
    "clustering_bandwidth": 5.5,  # 5.0,
    "clustering_alpha": 5.5,  # 5.0,
    "clustering_n_jobs": -1,
    "clustering_bin_seeding": True,
    "clustering_min_bin_freq": 16,  # 10,
    "clustering_subset": None,
    "left_cutout_time": 0.3,  # 0.2,
    "right_cutout_time": 1.8,  # 0.8,
    "detect_threshold": 20,  # 24, #15,
    # extra probe params
    "probe_masked_channels": [],
    "probe_inner_radius": 70,
    "probe_neighbor_radius": 90,
    "probe_event_length": 0.26,
    "probe_peak_jitter": 0.2,
    # extra detection params
    "t_inc": 100000,
    "num_com_centers": 1,
    "maa": 12,
    "ahpthr": 11,
    "out_file_name": "HS2_detected",
    "decay_filtering": False,
    "save_all": False,
    "amp_evaluation_time": 0.4,  # 0.14,
    "spk_evaluation_time": 1.0,
    # extra pca params
    "pca_ncomponents": 2,
    "pca_whiten": True,
    # bandpass filter
    "freq_min": 300.0,
    "freq_max": 6000.0,
    "filter": True,
    # rescale traces
    "pre_scale": True,
    "pre_scale_value": 20.0,
    # remove duplicates (based on spk_evaluation_time)
    "filter_duplicates": True,
}

# SETUP CONFIG
exp = "silico_neuropixels_from_nwb"
run = "npx_evoked"
sorter = "herdingspikes"
cfg, _ = get_config(exp, run).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# sort and postprocess
sort_and_postprocess_full(cfg, sorter, sorter_params, is_sort=False, is_postpro=True, copy_binary_recording=False, remove_bad_channels=True)