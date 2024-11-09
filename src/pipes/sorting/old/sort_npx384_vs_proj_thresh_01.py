"""sort npx32 simulation with ks3 vs. different detection threshold
takes 28 min

usage: 
    sbatch cluster/sort_npx384_vs_thresh.sbatch

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
import spikeinterface.full as si_full
import spikeinterface.sorters as ss
import yaml

from src.nodes.utils import get_config

# SET PARAMETERS
EXPERIMENT = "silico_neuropixels" # the experiment
SIMULATION_DATE = "2023_08_17"    # the run (date)

# SETUP CONFIG
data_conf, param_conf = get_config(EXPERIMENT, SIMULATION_DATE).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
KS3_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["input"]
KS3_SORTING_VS_PROJ_THRESH_PATHS = param_conf["fig_accuracy_vs_proj_thresh_01"]["sorting"]

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# get Spikeinterface Recording object
t0 = time()
Preprocessed = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
for ix, thresh_i in enumerate(KS3_SORTING_VS_PROJ_THRESH_PATHS):
    
    t0 = time()

    # set ks3 output and SortingExtractor paths
    KS3_OUTPUT_PATH = KS3_SORTING_VS_PROJ_THRESH_PATHS[thresh_i]["ks3_output"]
    KS3_SORTING_PATH = KS3_SORTING_VS_PROJ_THRESH_PATHS[thresh_i]["output"]

    # get threshold
    thresh_01 = KS3_SORTING_VS_PROJ_THRESH_PATHS[thresh_i]["thresh_01"]
    thresh_02 = KS3_SORTING_VS_PROJ_THRESH_PATHS[thresh_i]["thresh_02"]

    # sort
    sorting_KS3 = ss.run_sorter(sorter_name='kilosort3', recording=Preprocessed, output_folder=KS3_OUTPUT_PATH, verbose=True, projection_threshold=[thresh_01, thresh_02])

    # remove empty units (Samuel Garcia's advice)
    sorting_KS3 = sorting_KS3.remove_empty_units()
    logger.info(f"Done running kilosort3 for thresh: {thresh_01} in: %s", round(time() - t0, 1))

    # write
    shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
    sorting_KS3.save(folder=KS3_SORTING_PATH)
    logger.info(f"Done saving kilosort3 for thresh: {thresh_01} in: %s", round(time() - t0, 1))
