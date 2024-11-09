
"""sort 10 min of marques silico with Kilosort

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 07.05.2024

usage:

    sbatch cluster/sorting/marques_silico/10m/buttw/ks3/sort_noise_0uV_gain_1e4.sbatch

Note:
    - We do no preprocessing as Kilosort3 already preprocess the traces with
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

# SAVE recording as int16
SAVE_REC_INT16 = False   # once; exists
SECS = 10 * 60 # 20 minutes in seconds

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
    "NT":  None, #2*1024 + 64, # #65792*3, # None: automatically computed, # 65792
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
data_conf, _ = get_config("silico_neuropixels", "concatenated").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET PATHS
# trace
WIRED_40M_INT16_PATH = data_conf["probe_wiring"]["40m"]["output_noise_none_gain_1e4_int16"]
WIRED_INT16_PATH = data_conf["probe_wiring"]["10m"]["output_noise_0uV_gain_1e4_int16"]
# sorter
KS3_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["input_buttw"]
KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["10m"]["output_buttw_noise_0uV_gain_1e4"]
KS3_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["10m"]["ks3_output_buttw_noise_0uV_gain_1e4"]

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# load int16 recording
t0 = time()
WiredInt16 = si.load_extractor(WIRED_40M_INT16_PATH)
logger.info("Done loading int16 recording in: %s", round(time() - t0, 1))

# select 10 min
WiredInt16 = WiredInt16.frame_slice(
    start_frame=0, end_frame=int(SECS * WiredInt16.get_sampling_frequency())
)

# save recording as int16 once
if SAVE_REC_INT16:
    WiredInt16.save(
        folder=WIRED_INT16_PATH,
        format="binary",
        n_jobs=4,
        chunk_memory="40G",
        overwrite=True,
        progress_bar=True
    )
logger.info("Done selecting 10 mins in: %s", round(time() - t0, 1))

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
logger.info("Done running kilosort3 on Marques 10m noise 0uV gain 1e4 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
sorting_KS3.save(folder=KS3_SORTING_PATH)
logger.info("Done saving kilosort3 on Marques 10m noise 0uV gain 1e4 in: %s", round(time() - t0, 1))