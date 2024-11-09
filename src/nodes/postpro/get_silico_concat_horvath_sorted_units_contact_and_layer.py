"""Pipeline to get in concatenated silico Horvath sorted units' metadata (takes 5 min)
author: steeve.laquitaine@epfl.ch

Usage:
    sbatch cluster/postpro/get_silico_concat_horvath_sorted_units_metadata.sbatch

Returns:
    csv: write a .csv metadata file for depth files 1,2,3

Prerequisites:
    Silico Horvath recordings have been sorted, a SortingExtractors and WaveformExtractors exist

"""

import spikeinterface.postprocessing as spost
import spikeinterface as si
import os 
import pandas as pd
import copy
import numpy as np
import logging
import logging.config
import yaml
import time

# set project path
from src.nodes.utils import get_config
from src.nodes.postpro import waveform
from src.nodes.validation.layer import getAtlasInfo
from src.nodes.dataeng.silico import probe_wiring

# setup waveform parameters
MS_BEFORE = 3
MS_AFTER = 3

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def get_contact_layers(data_conf: dict):
    """get each site layer

    Args:
        data_conf (dict): _description_

    Returns:
        (pd.Series): 
        - index contacts id starting from 0 
        - values: layer
    """
    Recording = probe_wiring.load(data_conf)
    channel_coords = Recording.get_channel_locations(Recording.channel_ids, axes='xyz')
    out = getAtlasInfo(data_conf["dataeng"]["blueconfig"], channel_coords)
    return pd.Series(out[1], index=Recording.channel_ids)


def get_sorted_neuron_metadata(data_conf, recording_path, study_path, MS_BEFORE, MS_AFTER):
    """get sorted units' metadata (contact, layer)

    Requirements:
    - WaveformExtractors must have been created

    Args:
        RAW_HORVATH_1 (_type_): _description_
        recording_path (_type_): _description_
        study_path (_type_): _description_
        MS_BEFORE (_type_): _description_
        MS_AFTER (_type_): _description_

    Returns:
        _type_: _description_
    """
    # load WaveformExtractor
    recording = si.load_extractor(recording_path)
    we = waveform.load(recording, study_path, ms_before=MS_BEFORE, ms_after=MS_AFTER)

    # get channels where spike amplitude is maximal
    # note: channel_ids start from 1 not 0
    max_chids = spost.get_template_extremum_channel(
        we, peak_sign="both"
    )

    # map neurons to contacts
    neuron_contacts = pd.DataFrame.from_dict(max_chids, orient='index')

    # get contact layers
    contact_layers = get_contact_layers(data_conf)

    # map neurons to layers
    neuron_metadata = copy.copy(neuron_contacts)
    neuron_metadata["layer"] = np.nan
    for ix in neuron_contacts.index:    
        contact = neuron_contacts.loc[ix,0]
        layer = contact_layers.loc[contact]
        neuron_metadata.loc[ix, "layer"] = layer

    # construct metadata table
    neuron_metadata.columns = ["contact", "layer"]
    neuron_metadata.index.names = ["neuron"]
    return neuron_metadata
 
 
def run(experiment:str="silico_horvath", run:str="concatenated/probe_1"):

    t0 = time.time()

    # get config
    data_conf_silico_1, _ = get_config(experiment, run).values()
    RECORDING_PATH_silico_1 = data_conf_silico_1["preprocessing"]["output"]["trace_file_path"]
    STUDY_FOLDER_silico_1 = data_conf_silico_1["postprocessing"]["waveform"]["sorted"]["study"]
    SORTED_NEURON_METADATA_FILE_1 = data_conf_silico_1["postprocessing"]["sorted_neuron_metadata"]

    # get metadata (12 secs)
    neuron_metadata_1 = get_sorted_neuron_metadata(data_conf_silico_1, RECORDING_PATH_silico_1, STUDY_FOLDER_silico_1, MS_BEFORE, MS_AFTER)

    # save metadata file
    parent_path = os.path.dirname(SORTED_NEURON_METADATA_FILE_1)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    neuron_metadata_1.to_csv(SORTED_NEURON_METADATA_FILE_1, index=True)
    logger.info(f"Done writing units metadata in {time.time()-t0} secs")
    logger.info("Done")

