"""pipeline for data processing of horvath concatenated recordings with probe 1, 2 or 3 in silico
from raw bbp-workflow simulation files to single traces and spikes concatenated over
campaigns

usage: 

    sbatch cluster/processing/process_silico_concat_horvath.sbatch

duration: takes 53 min on a compute node
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np

from src.pipes.prepro.horvath_silico import concat, concat_probe_2

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.pipes.prepro.dense_spont import concat_probe_3
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
from src.pipes.metadata.dense_spont import label_layers


# SETUP PARAMETERS


# SET PROCESSING PARAMETERS
# GAIN = 4165.2               # derived from fitting in silico's scale to vivo (see nb)
# MISSING_NOISE_RMS = 16.875  # derived from fitting in silico's noise to vivo (see nb)
GAIN = 1                      # derived from fitting in silico's scale to vivo (see nb)
MISSING_NOISE_RMS = 0         # derived from fitting in silico's noise to vivo (see nb)


# SETUP PIPELINE
CONCAT = True
CAST = True
WIRE = True
SAVE_METADATA = True
PREPROCESS = True
GROUND_TRUTH = True


# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def concat_into_recording(run:str):
    """concatenate bbp-workflow simulations into campaigns/experiments
    and experiments into a single recording

    takes 30 min
    """
    if run == "concatenated/probe_1":
        concat.run()
    elif run == "concatenated/probe_2":
        concat_probe_2.run()
    elif run == "concatenated/probe_3":
        concat_probe_3.run()
    else:
        raise NotImplementedError("This run is not implemented")


def cast_as_RecordingExtractor(data_conf):
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 
    """
    # takes 28 mins
    t0 = time.time()

    # cast (30 secs)
    RecordingExtr = recording.run(data_conf, gain=GAIN, offset=True, noise_std=MISSING_NOISE_RMS)

    # remove 129th "test" channel (actually 128 because starts at 0)
    if len(RecordingExtr.channel_ids)==129:
        RecordingExtr = RecordingExtr.remove_channels([128])

    # write (2 mins)
    recording.write(RecordingExtr, data_conf)

    # check is probe
    try:
        RecordingExtr.get_probe()
    except:
        print("no wired probe was found")
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def wire_probe(data_conf, param_conf):

    # takes 26 min
    t0 = time.time()

    # run and write
    Recording = probe_wiring.run(data_conf, param_conf)
    probe_wiring.write(Recording, data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def save_metadata(experiment: str="silico_horvath", run: str="concatenated/probe_1"):

    # track time
    t0 = time.time()
    label_layers(experiment, run)
    logger.info(f"Saving metadata - done in {np.round(time.time()-t0,2)} secs")


def preprocess_recording(data_conf, param_conf):

    # takes 32 min
    t0 = time.time()

    # preprocess, write
    Preprocessed = preprocess.run(data_conf, param_conf)
    preprocess.write(Preprocessed, data_conf)

    # sanity check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def cast_as_true_spikes_as_SortingExtractor(data_conf, param_conf):

    # takes 27 secs
    t0 = time.time()

    # get simulation parameters
    simulation = load_campaign_params(data_conf)

    # cast ground truth spikes as a SpikeInterface Sorting Extractor object (1.5h for 534 units)
    SortingTrue = ground_truth.run(simulation, data_conf, param_conf)

    # write
    ground_truth.write(SortingTrue["ground_truth_sorting_object"], data_conf)   
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run(experiment:str="silico_horvath", run:str="concatenated/probe_1"):

    # get config
    data_conf, param_conf = get_config(experiment, run).values()

    # track time
    t0 = time.time()
    if CONCAT:
        concat_into_recording(run)
    if CAST:
        cast_as_RecordingExtractor(data_conf)
    if WIRE:
        wire_probe(data_conf, param_conf)
    if SAVE_METADATA:
        save_metadata(experiment, run)
    if PREPROCESS:
        preprocess_recording(data_conf, param_conf)
    if GROUND_TRUTH:
        cast_as_true_spikes_as_SortingExtractor(data_conf, param_conf)
    print(f"done in {np.round(time.time()-t0,2)} secs")