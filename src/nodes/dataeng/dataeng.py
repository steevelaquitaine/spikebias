"""Nodes for data engineering
"""

import pandas as pd
import spikeinterface.extractors as se
import time
import logging
import logging.config
import yaml
import os
import shutil
import numpy as np

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SETUP PARALLEL PROCESSING
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True}


def save_raw_rec_extractor(data_conf):
    """Write raw simulated recording as a spikeinterface
    RecordingExtractor

    Args:
        data_conf (_type_): _description_

    Returns:
        _type_: _description_
    """
    # track time
    t0 = time.time()
    logger.info("Starting ...")
    
    # set traces read write paths
    READ_PATH = data_conf["recording"]["input"]
    WRITE_PATH = data_conf["recording"]["output"]

    # get campaign parameters from one simulation
    simulation = load_campaign_params(data_conf)

    # read and cast raw trace as array (1 min/h recording)
    trace = pd.read_pickle(READ_PATH)
    trace = np.array(trace)
    
    # cast trace as a SpikeInterface Recording object
    Recording = se.NumpyRecording(
        traces_list=[trace],
        sampling_frequency=simulation["lfp_sampling_freq"],
    )
    
    # write (parallel processing works for 10 min recordings, else use 1 node for 1h recording otherwise
    # you get "out of memory error: "slurmstepd: error: Detected 50 oom-kill event(s). 
    # Some of your processes may have been killed by the cgroup out-of-memory handler."")
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Recording.save(folder=WRITE_PATH, format="binary", **job_dict)

    # log
    logger.info("Probe wiring done in  %s secs", round(time.time() - t0, 1))    
    
    
    
def fit_and_cast_as_extractor(data_conf, offset, scale_and_add_noise):
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 

    takes 54 min
    note: RecordingExtractor is not dumpable and can't be processed in parallel
    """
    
    # track time
    t0 = time.time()
    logger.info("Starting ...")

    # cast (30 secs)
    Recording = recording.run(data_conf, offset=offset, scale_and_add_noise=scale_and_add_noise)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    return Recording
