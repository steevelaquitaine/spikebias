"""sort probe 2 of silico horvath with ks3
takes 28 min

usage:
    sbatch cluster/sorting/sort_concat_silico_horvath_2.sbatch

We do no preprocessing as Kilosort3 already preprocess the traces with 
(see code preprocessDataSub()):

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
EXPERIMENT = "silico_horvath"
SIMULATION_DATE = "concatenated/probe_2"
BATCH_SIZE = 65792
MIN_FR = 0
MINFR_GOODCHANNELS = 0

# SETUP CONFIG
data_conf, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
KS3_PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["input"]
KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]
KS3_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["ks3_output"]

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# get Spikeinterface Recording object
# - remove 129th "test" channel (actually 128 because starts at 0)
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
if len(Recording.channel_ids)==129:
    Recording = Recording.remove_channels([128])
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
sorting_KS3 = ss.run_sorter(sorter_name='kilosort3', recording=Recording, output_folder=KS3_OUTPUT_PATH, verbose=True, NT=BATCH_SIZE, minFR=MIN_FR, minfr_goodchannels=MINFR_GOODCHANNELS)

# remove empty units (Samuel Garcia's advice)
sorting_KS3 = sorting_KS3.remove_empty_units()
logger.info("Done running kilosort3 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
sorting_KS3.save(folder=KS3_SORTING_PATH)
logger.info("Done saving kilosort3 in: %s", round(time() - t0, 1))