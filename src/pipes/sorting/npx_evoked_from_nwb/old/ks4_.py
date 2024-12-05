"""sort 20 min of npx evoked with Kilosort 4.0
takes 60 min (with saving, else 20 min)

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024

usage:

    sbatch cluster/sorting/npx_evoked/ks4.sh

Install: pip install git+https://github.com/SpikeInterface/spikeinterface.git@refs/pull/2827/merge

Note:
    - We do no preprocessing as Kilosort4 already preprocess the traces with
    (see code preprocessDataSub()):
    - we set minFR and minfr_goodchannels to 0

% 1) conversion to int16;
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
import spikeinterface.preprocessing as spre
import yaml

from src.nodes.utils import get_config

# SET PARAMETERS
# these are the default parameters
# for the version of spikeinterface used
sorter_params = {
    "batch_size": 10000, #30000, #60000,
    "nblocks": 1,
    "Th_universal": 9,
    "Th_learned": 8,
    "do_CAR": True,
    "invert_sign": False,
    "nt": 61,
    "artifact_threshold": None,
    "nskip": 25,
    "whitening_range": 32,
    "binning_depth": 5,
    "sig_interp": 20,
    "nt0min": None,
    "dmin": None,
    "dminx": 25.6, #None,
    "min_template_size": 10,
    "template_sizes": 5,
    "nearest_chans": 10,
    "nearest_templates": 100,
    "templates_from_data": True,
    "n_templates": 6,
    "n_pcs": 6,
    "Th_single_ch": 6,
    "acg_threshold": 0.2,
    "ccg_threshold": 0.25,
    "cluster_downsampling": 20,
    "cluster_pcs": 64,
    "duplicate_spike_bins": 15,
    "do_correction": True,
    "keep_good_only": False,
    "save_extra_kwargs": False,
    "skip_kilosort_preprocessing": False,
    "scaleproc": None,
}

# SETUP CONFIG
data_conf, _ = get_config("silico_neuropixels", "npx_evoked").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["full"]["output_noise_fitd_gain_fitd_adj10perc_less_int16"]

# SET WRITE PATHS
KS4_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort4"]["output"]
KS4_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort4"]["ks4_output"]

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)

# convert as int16
Recording = spre.astype(Recording, "int16")  # convert to int16 for KS3
logger.info("Done converting as int16 in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
sorting_KS4 = ss.run_sorter(sorter_name='kilosort4',
                            recording=Recording,
                            remove_existing_folder=True,
                            output_folder=KS4_OUTPUT_PATH,
                            verbose=True,
                            **sorter_params)

# remove empty units (Samuel Garcia's advice)
sorting_KS4 = sorting_KS4.remove_empty_units()
logger.info("Done running kilosort4 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS4_SORTING_PATH, ignore_errors=True)
sorting_KS4.save(folder=KS4_SORTING_PATH)
logger.info("Done saving kilosort4 in: %s", round(time() - t0, 1))