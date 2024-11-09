"""pipeline to concatenate recordings with probe 1 for in silico Horvath
"""

import os
import pandas as pd
import yaml 
import logging
import logging.config
import shutil

proj_dir = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023"
os.chdir(proj_dir)

from src.nodes.utils import get_config
from src.nodes.dataeng.silico import concat_campaigns

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PARAMETERS
EXPERIMENT = "silico_horvath"

# campaign 1
RUN = "campaign_1/probe_1"
data_conf_1, param_conf_1 = get_config(EXPERIMENT, RUN).values()
RAW_SPIKE_PATH_1 = data_conf_1["dataeng"]["campaign"]["output"]["spike_file_path"]
RAW_RECORDING_PATH_1 = data_conf_1["dataeng"]["campaign"]["output"]["trace_file_path"]

# campaign 2
RUN = "campaign_2/probe_1"
data_conf_2, param_conf_2 = get_config(EXPERIMENT, RUN).values()
RAW_SPIKE_PATH_2 = data_conf_2["dataeng"]["campaign"]["output"]["spike_file_path"]
RAW_RECORDING_PATH_2 = data_conf_2["dataeng"]["campaign"]["output"]["trace_file_path"]

# campaign 2
RUN = "campaign_3/probe_1"
data_conf_3, param_conf_3 = get_config(EXPERIMENT, RUN).values()
RAW_SPIKE_PATH_3 = data_conf_3["dataeng"]["campaign"]["output"]["spike_file_path"]
RAW_RECORDING_PATH_3 = data_conf_3["dataeng"]["campaign"]["output"]["trace_file_path"]

# concatenated
RUN = "concatenated/probe_1"
data_conf_concat, _ = get_config(EXPERIMENT, RUN).values()
CONCAT_SPIKE_PATH = data_conf_concat["dataeng"]["campaign"]["output"]["spike_file_path"]
CONCAT_RECORDING_PATH = data_conf_concat["dataeng"]["campaign"]["output"]["trace_file_path"]

def run():

    # - spikes
    spikes_1 = pd.read_pickle(RAW_SPIKE_PATH_1)
    spikes_2 = pd.read_pickle(RAW_SPIKE_PATH_2)
    spikes_3 = pd.read_pickle(RAW_SPIKE_PATH_3)

    # - traces
    traces_1 = pd.read_pickle(RAW_RECORDING_PATH_1)
    traces_2 = pd.read_pickle(RAW_RECORDING_PATH_2)
    traces_3 = pd.read_pickle(RAW_RECORDING_PATH_3)

    duration_recording_1 = traces_1.index[-1]

    # concatenate
    traces = concat_campaigns.concat_raw_recordings(traces_1, traces_2)
    spikes = concat_campaigns.concat_raw_ground_truth_spikes(
        spikes_1, spikes_2, duration_recording_1
        )

    # re/create path and write new files
    parent_path = os.path.dirname(CONCAT_RECORDING_PATH)
    shutil.rmtree(parent_path, ignore_errors=True)
    os.makedirs(parent_path)
    spikes.to_pickle(CONCAT_SPIKE_PATH)
    traces.to_pickle(CONCAT_RECORDING_PATH)

run()