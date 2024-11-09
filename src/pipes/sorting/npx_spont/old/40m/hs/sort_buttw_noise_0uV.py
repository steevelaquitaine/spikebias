"""sort marques silico with herdingspikes
takes 60 min (with saving, else 20 min)

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 07.05.2024

usage: 

    sbatch cluster/sorting/marques_silico/40m/hs/sort_buttw_noise_0uV.sbatch

Note:
    - We do no preprocessing as herdingspikes already preprocess the traces with
    (see code preprocessDataSub()):
    - we set minFR and minfr_goodchannels to 0

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
SAVE_REC_INT16 = False   # once; exists

# SET PARAMETERS
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
KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["herdingspikes"]["40m"]["output_buttw_noise_0uV"]
KS3_OUTPUT_PATH = data_conf["sorting"]["sorters"]["herdingspikes"]["40m"]["hs_output_buttw_noise_0uV"]

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
Sorting = ss.run_sorter(sorter_name='herdingspikes',
                            recording=WiredInt16,
                            remove_existing_folder=True,
                            output_folder=KS3_OUTPUT_PATH,
                            verbose=True,
                            **sorter_params)

# remove empty units (Samuel Garcia's advice)
Sorting = Sorting.remove_empty_units()
logger.info("Done running herdingspikes in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
Sorting.save(folder=KS3_SORTING_PATH)
logger.info("Done saving herdingspikes in: %s", round(time() - t0, 1))