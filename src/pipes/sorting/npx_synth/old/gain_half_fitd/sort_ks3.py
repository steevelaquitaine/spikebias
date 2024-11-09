"""sort buccino with Kilosort 3.0
takes 16 min

  author: steeve.laquitaine@epfl.ch
    date: 15.01.2023
modified: 15.01.2023

usage: 

    sbatch cluster/sorting/buccino/gain_half_fitd/sort_ks3.sbatch

We do no preprocessing as Kilosort3 already preprocess the traces in
source code kilosort.preProcess.preprocessDataSub.m:

% 1) conversion to int16;
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
    "detect_threshold": 6, # alters detection on a per spike basis and is applied to the voltage trace (default=6)
    "projection_threshold": [9, 9],  # threshold for detected projected spikes (energy) during two passes
    "preclust_threshold": 8,
    "car": True,
    "minFR": 0, # 0.2, # Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed"
    "minfr_goodchannels": 0, # 0.2, # Minimum firing rate on a 'good' channel"
    "nblocks": 5, # "blocks for registration. 0 turns it off, 1 does rigid registration. Replaces 'datashift' option.",
    "sig": 20,
    "freq_min": 300,
    "sigmaMask": 30,
    "lam": 20.0,
    "nPCs": 3,
    "ntbuff": 64,
    "nfilt_factor": 4,
    "do_correction": True,
    "NT": None, #16 * 1024 + 64, # #65792*3, # None: automatically computed, # 65792
    "AUCsplit": 0.8, # Threshold on the area under the curve (AUC) criterion for performing a split in the final step
    "wave_length": 61,
    "keep_good_only": False,
    "skip_kilosort_preprocessing": False, # no preprocessing (I commented filtering commands in KS3 gpufilter.m)
    "scaleproc": None, # 200 #is the scaling aaplied after whitening during preprocessing by default (when "None")
    "save_rez_to_mat": False,
    "delete_tmp_files": ("matlab_files",),
    "delete_recording_dat": False,
}


# SETUP CONFIG
data_conf, _ = get_config("buccino_2020", "2020").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["10m"]["output_gain_half_fitd_int16"]

# SET WRITE PATHS
KS3_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["input_buttw"]
KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output_noise_half_fitd_int16"]
KS3_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["ks3_output_noise_half_fitd_int16"]

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# get Spikeinterface Recording object
t0 = time()
WiredInt16 = si.load_extractor(RECORDING_PATH)
assert WiredInt16.dtype == "int16", "RecordingExtractor should be int16"
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
sorting_KS3 = ss.run_sorter(sorter_name='kilosort3',
                            recording=WiredInt16,
                            remove_existing_folder=True,
                            output_folder=KS3_OUTPUT_PATH,
                            verbose=True,
                            **sorter_params)

# remove empty units (Samuel Garcia's advice)
sorting_KS3 = sorting_KS3.remove_empty_units()
logger.info("Done running kilosort3 on 10m half fitted gain Buccino in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
sorting_KS3.save(folder=KS3_SORTING_PATH)
logger.info("Done saving kilosort3 on 10m half fitted gain Buccino in: %s", round(time() - t0, 1))