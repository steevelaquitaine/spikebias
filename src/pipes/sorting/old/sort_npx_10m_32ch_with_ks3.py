"""sort npx 10m 384ch recordings with ks3

usage (see script in sbatches/):

    sbatch sbatch/sort_npx_10m_32ch_with_ks3.sbatch

stats: 20 min
"""
import logging
import logging.config
import shutil
from time import time

import spikeinterface as si
# import spikeinterface.full as si_full
import spikeinterface.sorters as ss
import yaml

from src.nodes.utils import get_config

# SET PARAMETERS
EXPERIMENT = "silico_neuropixels"  # specifies the experiment
SIMULATION_DATE = "2023_02_19"  # specifies the run (date)

# SETUP CONFIG
data_conf, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
# recording_path = data_conf["probe_wiring"]["output"]
recording_path = data_conf["preprocessing"]["output"]["trace_file_path"]

# SET WRITE PATHS
GT_SORTING_PATH = data_conf["ground_truth"]["full"]["output"]
KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]
KS3_OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["kilosort3_output"]

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(
    data_conf["sorting"]["sorters"]["kilosort3"]["input"]
)

# get Spikeinterface Recording object
t0 = time()
RX = si.load_extractor(recording_path)
logger.info("loading recording done: %s", round(time() - t0, 1))

# preprocess
# RX = si_full.bandpass_filter(RX, freq_min=300, freq_max=6000)

# run sorting
t0 = time()
sorting_KS3 = ss.run_kilosort3(RX, output_folder=KS3_OUTPUT_PATH, verbose=True)
logger.info(
    "kilosort3 sorting - done: %s",
    round(time() - t0, 1),
)

# write
t0 = time()
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
sorting_KS3.save(folder=KS3_SORTING_PATH)
logger.info(
    "saving kilosort3 sorting - done: %s",
    round(time() - t0, 1),
)