"""sort marques silico with Kilosort
takes 60 min (with saving, else 20 min)

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 07.05.2024

usage: 

    sbatch cluster/sorting/marques_silico/40m/ks/sort_buttw_noise_0uV.sbatch

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

"""
import logging
import logging.config
import shutil
from time import time

import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
import yaml

from src.nodes.utils import get_config

# SAVE recording as int16
SAVE_REC_INT16 = False   # once; exists

# SET PARAMETERS
# default params
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
data_conf, _ = get_config("silico_neuropixels", "concatenated").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET PATHS
# trace
WIRED_FLOAT32_PATH = data_conf["probe_wiring"]["output_noise_0uV"]
WIRED_INT16_PATH = data_conf["probe_wiring"]["output_noise_0uV_int16"]
# sorter
KS_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort"]["40m"]["input_buttw"]
KS_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort"]["40m"]["output_buttw_noise_0uV"]
KS_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort"]["40m"]["ks_output_buttw_noise_0uV"]

# SET KS environment variable
ss.KilosortSorter.set_kilosort_path(KS_PACKAGE_PATH)

# save recording as int16 once
if SAVE_REC_INT16:
    WiredFloat32 = si.load_extractor(WIRED_FLOAT32_PATH)
    WiredFloat16 = spre.astype(WiredFloat32, "int16")
    WiredFloat16.save(
        folder=WIRED_INT16_PATH,
        format="binary",
        n_jobs=4,
        chunk_memory="40G",
        overwrite=True,
        progress_bar=True
    )

# load int16 recording
t0 = time()
WiredInt16 = si.load_extractor(WIRED_INT16_PATH)
logger.info("Done loading int16 recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
sorting_KS = ss.run_sorter(sorter_name='kilosort',
                            recording=WiredInt16,
                            remove_existing_folder=True,
                            output_folder=KS_OUTPUT_PATH,
                            verbose=True,
                            **sorter_params)

# remove empty units (Samuel Garcia's advice)
sorting_KS = sorting_KS.remove_empty_units()
logger.info("Done running kilosort in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS_SORTING_PATH, ignore_errors=True)
sorting_KS.save(folder=KS_SORTING_PATH)
logger.info("Done saving kilosort in: %s", round(time() - t0, 1))