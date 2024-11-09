
"""sort 20 min of marques silico with Kilosort 3.0
takes 60 min (with saving, else 20 min)

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 07.05.2024

usage:

    sbatch cluster/sorting/others/spikewarp/ks.sh

Note:
    - We do no preprocessing as Kilosort already preprocess the traces with
    (see code preprocessDataSub()):
    - we set minFR and minfr_goodchannels to 0

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: 
- to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints
- crashes with "Maximum variable size allowed on the device is exceeded." with "minFR"
and "minfr_goodchannels" set to 0.

"""
import logging
import logging.config
import shutil
from time import time

import spikeinterface as si
import spikeinterface.sorters as ss
import yaml

from src.nodes.utils import get_config
from src.nodes.sorting import sort_and_postprocess_10m

# SET PARAMETERS
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

# # SETUP CONFIG
# data_conf, _ = get_config("others/for_james", "2024_04_13").values()

# # setup logging
# with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
#     LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
# logging.config.dictConfig(LOG_CONF)
# logger = logging.getLogger("root")

# # SET PATHS
# # trace
# RECORDING_PATH = data_conf["probe_wiring"]["output"]

# # sorter
# KS_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort"]["input"]
# KS_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort"]["output"]
# KS_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort"]["ks_output"]

# # SET KS3 environment variable
# ss.KilosortSorter.set_kilosort_path(KS_PACKAGE_PATH)

# # get Spikeinterface Recording object
# t0 = time()
# Recording = si.load_extractor(RECORDING_PATH)
# logger.info("Done loading recording in: %s", round(time() - t0, 1))

# # run sorting (default parameters)
# t0 = time()
# sorting_KS = ss.run_sorter(sorter_name='kilosort', recording=Recording, output_folder=KS_OUTPUT_PATH, verbose=True, **sorter_params)

# # remove empty units (Samuel Garcia's advice)
# sorting_KS = sorting_KS.remove_empty_units()
# logger.info("Done running kilosort in: %s", round(time() - t0, 1))

# # write
# shutil.rmtree(KS_SORTING_PATH, ignore_errors=True)
# sorting_KS.save(folder=KS_SORTING_PATH)
# logger.info("Done saving kilosort in: %s", round(time() - t0, 1))


# SETUP CONFIG
exp = "others/spikewarp"
run = "2024_04_13"
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
sort_and_postprocess_10m(cfg,
                         sorter,
                         sorter_params,
                         duration_sec=600,
                         is_sort=False,
                         is_postpro=True,
                         copy_binary_recording=False)