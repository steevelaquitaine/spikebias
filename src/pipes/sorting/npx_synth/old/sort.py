"""sort buccino 2020 recordings with Kilosort 3.0

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/sorting/buccino/sort.sbatch

takes 28 min

We do no preprocessing as Kilosort3 already preprocess the traces with 
(see code preprocessDataSub()):

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values
"""
import logging
import logging.config
import os
import shutil
from time import time
import numpy as np
import spikeinterface as si
import spikeinterface.sorters as ss
import yaml

from src.nodes.utils import get_config

# SET PARAMETERS
sorter_params = {
    'detect_threshold': 6, # 6      # spike detection thresh.
    'projection_threshold': [9, 9], # template-projected detection thresh.
    'preclust_threshold': 8,
    'car': True,
    'minFR': 0.2,
    'minfr_goodchannels': 0.2,
    'nblocks': 5,
    'sig': 20,
    'freq_min': 300,                # high-pass filter cutoff
    'sigmaMask': 30,
    'nPCs': 3,
    'ntbuff': 64,
    'nfilt_factor': 4,
    'do_correction': True,
    'NT': 65792, # None,            # Batch size
    'wave_length': 61,
    'keep_good_only': False
    }

# SETUP CONFIG
data_conf, _ = get_config("buccino_2020", "2020").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
WIRED_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
KS3_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["input"]
KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]
KS3_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["ks3_output"]

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# report parameters
logger.info("Sorter params: %s", sorter_params)

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(WIRED_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting
t0 = time()
sorting_KS3 = ss.run_kilosort3(Recording, output_folder=KS3_OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
sorting_KS3 = sorting_KS3.remove_empty_units()
logger.info("Done running kilosort 3.0 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
# os.makedirs(KS3_SORTING_PATH)
sorting_KS3.save(folder=KS3_SORTING_PATH)
logger.info("Done saving kilosort3 in: %s", round(time() - t0, 1))
