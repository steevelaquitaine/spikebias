"""pipeline for data processing npx 2023_06_26 experiment

usage: 
    sbatch cluster/processing/process_npx_2023_06_26.sbatch

duration: 26 min on one node
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj68/home/laquitai/sfn_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
from src.nodes.dataeng.lfp_only import stacking

# SETUP PARAMETERS
EXPERIMENT = "silico_neuropixels"   # specifies the experiment 
SIMULATION_DATE = "2023_06_26"      # specifies the run (date)
data_conf, param_conf = get_config(EXPERIMENT, SIMULATION_DATE).values()

GAIN = 1e5

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def stack():

    # takes 7 min (for 5 min rec)
    t0 = time.time()

    # get campaign params and stack
    campaign_params = load_campaign_params(data_conf)
    stacking.run(data_conf, param_conf, campaign_params["blue_config"])
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def cast_as_RecordingExtractor():
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 
    """
    # takes 28 mins
    t0 = time.time()

    # cast (30 secs)
    RecordingExtr = recording.run(data_conf, gain=GAIN, offset=True)

    # write (2 mins)
    recording.write(RecordingExtr, data_conf)
    RecordingExtr = recording.load(data_conf)

    # check is probe
    try: 
        RecordingExtr.get_probe() 
    except: 
        print("there is no probe wired")
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def wire_probe():

    # takes 26 min
    t0 = time.time()

    # run and write
    Recording = probe_wiring.run(data_conf, param_conf)
    probe_wiring.write(Recording, data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def preprocess_recording():

    # takes 32 min
    t0 = time.time()

    # preprocess, write
    Preprocessed = preprocess.run(data_conf, param_conf)
    preprocess.write(Preprocessed, data_conf)

    # sanity check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")

def extract_ground_truth():

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
    stack()
    cast_as_RecordingExtractor()
    wire_probe()
    preprocess_recording()
    extract_ground_truth()
    print(f"done in {np.round(time.time()-t0,2)} secs")

# run pipeline
run()