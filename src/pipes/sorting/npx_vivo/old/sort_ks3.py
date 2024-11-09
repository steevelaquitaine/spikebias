"""sort marques vivo with Kilosort 3.0
takes 21 min

  author: steeve.laquitaine@epfl.ch
    date: 17.01.2024
modified: 17.01.2024

usage: 

    sbatch cluster/sorting/marques_vivo/sort_ks3.sbatch

We do no preprocessing as Kilosort3 already preprocess the traces with
(see code preprocessDataSub()):

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints

References:
    Solve "Error using gpuArray/subsref": "Subscript indices must either be real positive integers or logicals."
    https://github.com/MouseLand/Kilosort/issues/360
"""
import logging
import logging.config
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
data_conf, _ = get_config("vivo_marques", "c26").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]
KS3_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["input"]
KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]
KS3_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["ks3_output"]

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# report parameters
logger.info("Sorter params: %s", sorter_params)

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading wired recording in: %s", round(time() - t0, 1))

## remove bad channels (outside of the cortex)
# layers = Recording.get_property("layers")
# bad_channels = np.where(layers == "Outside")[0]
# Recording = Recording.remove_channels(bad_channels)

# run sorting (default parameters)
# sorting does not require that a probe be wired
# Recording must just have "channel_ids" and 
# "channel_locations" attributes filled
t0 = time()
sorting_KS3 = ss.run_sorter(sorter_name='kilosort3', recording=Recording, output_folder=KS3_OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
sorting_KS3 = sorting_KS3.remove_empty_units()
logger.info("Done running kilosort 3.0 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
sorting_KS3.save(folder=KS3_SORTING_PATH)
logger.info("Done saving kilosort 3.0 in: %s", round(time() - t0, 1))