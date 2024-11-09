"""sort 20 min of marques silico with Kilosort 3.0
takes 60 min (with saving, else 20 min)

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 07.05.2024

usage:

    sbatch cluster/sorting/marques_silico/10m/buttw/ks4/sort_noise_ftd_gain_ftd.sbatch

Note:
    - We do no preprocessing as Kilosort3 already preprocess the traces with
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

# SAVE recording as int16
SAVE_REC_INT16 = True   # once; exists
SECS = 10 * 60 # 10 minutes

# SET PARAMETERS
# these are the default parameters
# for the version of spikeinterface used
sorter_params = {
    "batch_size": 30000, #60000,
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
data_conf, _ = get_config("silico_neuropixels", "concatenated").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET PATHS
# trace
WIRED_40M_PATH = data_conf["probe_wiring"]["output_noise_fitd_int16"]
WIRED_INT16_PATH = data_conf["probe_wiring"]["10m"]["output_noise_fitd_int16"]
# sorter
KS4_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort4"]["10m"]["output_buttw_noise_ftd_gain_ftd"]
KS4_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort4"]["10m"]["ks4_output_buttw_noise_ftd_gain_ftd"]

# save 10 min recording as int16 once
if SAVE_REC_INT16:
    # select 10 min
    Wired40m = si.load_extractor(WIRED_40M_PATH)
    Wired10m = Wired40m.frame_slice(
        start_frame=0, end_frame=int(SECS * Wired40m.get_sampling_frequency())
    )
    # compress to int16
    Wired10mInt16 = spre.astype(Wired10m, "int16")
    # save
    Wired10mInt16.save(
        folder=WIRED_INT16_PATH,
        format="binary",
        n_jobs=4,
        chunk_memory="40G",
        overwrite=True,
        progress_bar=True
    )

# load int16 recording
t0 = time()
Wired10mInt16 = si.load_extractor(WIRED_INT16_PATH)
logger.info("Done loading 10 min int16 recording in: %s", round(time() - t0, 1))

# select 20 min and ensure int16
Wired10mInt16 = spre.astype(Wired10mInt16, "int16")
logger.info("Done re-compressing to int16 in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
sorting_KS4 = ss.run_sorter(sorter_name='kilosort4',
                            recording=Wired10mInt16,
                            remove_existing_folder=True,
                            output_folder=KS4_OUTPUT_PATH,
                            verbose=True,
                            **sorter_params)

# remove empty units (Samuel Garcia's advice)
sorting_KS4 = sorting_KS4.remove_empty_units()
logger.info("Done running kilosort4 for 10 min noise fitd gain fitd in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS4_SORTING_PATH, ignore_errors=True)
sorting_KS4.save(folder=KS4_SORTING_PATH)
logger.info("Done saving kilosort4 for 10 min noise fitd gain fitd in: %s", round(time() - t0, 1))