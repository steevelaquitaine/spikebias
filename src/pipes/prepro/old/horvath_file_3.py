"""_summary_
"""

import os
import time
import numpy as np
import spikeinterface.extractors as se
import logging
import logging.config
import yaml

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj68/home/laquitai/bernstein_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth

# SETUP CONFIG
EXPERIMENT = "vivo_horvath"   # specifies the experiment
SIMULATION_DATE = "2021_file_3"      # specifies the run (date)
data_conf, param_conf = get_config(EXPERIMENT, SIMULATION_DATE).values()

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def cast_as_RecordingExtractor():

    # takes 60 mins (6 min recording)
    t0 = time.time()

    # cast (30 secs)
    RecordingExtr = se.NwbRecordingExtractor(data_conf["raw"])

    # write (2 mins)
    recording.write(RecordingExtr, data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")

def check_has_probe(): 

    Recording = recording.load(data_conf)
    try: 
        Recording.get_probe()
        logger.info("A probe is wired as expected")
    except: 
        logger.info("Something is wrong. No probe is wired.")

def preprocess_recording():
    
    # takes 32 min (5min rec)
    t0 = time.time()

    # preprocess (8 min)
    Preprocessed = preprocess.run(data_conf, param_conf)

    # write
    preprocess.write(Preprocessed, data_conf)

    # sanity check is preprocessed
    logger.info(f"Preprocessing done?: {Preprocessed.is_filtered()}")
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")

def sort_ground_truth():

    # takes 27 secs    
    t0 = time.time()

    # get simulation parameters
    simulation = load_campaign_params(data_conf)

    # cast ground truth spikes as a SpikeInterface Sorting Extractor object (1.5h for 534 units)
    SortingTrue = ground_truth.run(simulation, data_conf, param_conf)

    # write
    ground_truth.write(SortingTrue["ground_truth_sorting_object"], data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")

def run():

    t0 = time.time()
    cast_as_RecordingExtractor()
    check_has_probe()
    preprocess_recording()
    sort_ground_truth()
    print(f"done in {np.round(time.time()-t0,2)} secs")

# run 
run()