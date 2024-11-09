"""pipeline for data processing of marques c26 in vivo recordings

author: steeve.laquitaine@epfl.ch

usage: 

    sbatch cluster/processing/process_marques_vivo_c26.sbatch

duration: takes 54 min on a compute node
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
import spikeinterface as si
from src.pipes.metadata.marques_vivo import label_layers


# SETUP PARAMETERS
EXPERIMENT = "vivo_marques"
RUN = "c26"
data_conf, param_conf = get_config(EXPERIMENT, RUN).values()
RECORDING_PATH = data_conf["raw"]
SAMP_FREQ = param_conf["run"]["sampling_frequency"]
N_CONTACTS = param_conf["probe"]["n_contacts"]
DTYPE = param_conf["dtype"]

# SETUP PIPELINE
CAST = True
WIRE = True
SAVE_METADATA = True
PREPROCESS = True

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")



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
    RecordingExtr = si.read_binary(RECORDING_PATH, sampling_frequency=SAMP_FREQ, num_chan=N_CONTACTS, dtype=DTYPE)

    # write (10 mins)
    recording.write(RecordingExtr, data_conf)

    # check is probe
    try:
        RecordingExtr.get_probe()
    except:
        print("no wired probe was found")
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def wire_probe():
    """wire probe (takes 20 min)
    """
    
    # track time
    t0 = time.time()

    # run and write
    Recording = probe_wiring.run(data_conf, param_conf)
    probe_wiring.write(Recording, data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def save_metadata():

    # save layers to probe-wired RecordingExtractor
    label_layers()


def preprocess_recording():

    # takes 32 min
    t0 = time.time()

    # preprocess, write
    Preprocessed = preprocess.run(data_conf, param_conf)
    preprocess.write(Preprocessed, data_conf)

    # sanity check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run():

    t0 = time.time()
    if CAST:
        cast_as_RecordingExtractor()
    if WIRE:
        wire_probe()
    if SAVE_METADATA:
        save_metadata()
    if PREPROCESS:
        preprocess_recording()
    print(f"done in {np.round(time.time()-t0,2)} secs")
    
# run pipeline
run()
