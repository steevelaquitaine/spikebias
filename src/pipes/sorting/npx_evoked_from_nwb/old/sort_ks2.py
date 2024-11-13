"""sort james stimulus simulation with Kilosort 3.0
takes 30 min

  author: steeve.laquitaine@epfl.ch
    date: 22.04.2024
modified: 22.04.2024

usage:

    # Download Kilosort 2 release
    wget -c https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.0.tar.gz -O - | tar -xz

    # Submit to cluster
    sbatch cluster/sorting/others/for_james/sort_ks2.sbatch

We do no preprocessing as Kilosort2 already preprocess the traces with
(see code preprocessDataSub())

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints
"""

import logging
import logging.config
import shutil
from time import time

import spikeinterface as si
import spikeinterface.sorters as ss
import yaml

from src.nodes.utils import get_config

# SET PARAMETERS
sorter_params = {
        'detect_threshold': 6,
        'projection_threshold': [10, 4],
        'preclust_threshold': 8,
        'car': True,
        'minFR': 0.1,
        'minfr_goodchannels': 0.1,
        'freq_min': 150,
        'sigmaMask': 30,
        'nPCs': 3,
        'ntbuff': 64,
        'nfilt_factor': 4,
        'NT': None,
        'wave_length': 61,
        'keep_good_only': False,
    }

# SETUP CONFIG
data_conf, _ = get_config("others/for_james", "2024_04_13").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
KS2_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort2"]["input"]
KS2_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort2"]["output"]
KS2_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort2"]["ks2_output"]

# SET KS3 environment variable
ss.Kilosort2Sorter.set_kilosort2_path(KS2_PACKAGE_PATH)

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
sorting_KS2 = ss.run_sorter(sorter_name='kilosort2', recording=Recording, output_folder=KS2_OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
sorting_KS2 = sorting_KS2.remove_empty_units()
logger.info("Done running kilosort2 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS2_SORTING_PATH, ignore_errors=True)
sorting_KS2.save(folder=KS2_SORTING_PATH)
logger.info("Done saving kilosort2 in: %s", round(time() - t0, 1))