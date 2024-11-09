"""Pipeline to get in silico Marques sorted units' metadata (takes 5 min)
author: steeve.laquitaine@epfl.ch

Usage:
    sbatch cluster/postpro/marques_silico/get_sorted_units_site_and_layer.sbatch

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


def get_site_layers(data_conf: dict):
    """get each site's layer

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


def get_sorted_unit_metadata(data_conf, recording_path, study_path, MS_BEFORE, MS_AFTER):
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
        pd.DataFrame: _description_
    """
    # load Waveform Extractor
    recording = si.load_extractor(recording_path)
    we = waveform.load(recording, study_path, ms_before=MS_BEFORE, ms_after=MS_AFTER)

    # get channels where spike amplitude is maximal
    # note: channel_ids start from 1 not 0
    max_chids = spost.get_template_extremum_channel(
        we, peak_sign="both"
    )

    # map units to recording sites
    unit_sites = pd.DataFrame.from_dict(max_chids, orient='index')

    # get site layers
    site_layers = get_site_layers(data_conf)

    # map units to layers
    unit_metadata = copy.copy(unit_sites)
    unit_metadata["layer"] = np.nan
    for ix in unit_sites.index:
        contact = unit_sites.loc[ix,0]
        layer = site_layers.loc[contact]
        unit_metadata.loc[ix, "layer"] = layer

    # construct metadata table
    unit_metadata.columns = ["contact", "layer"]
    unit_metadata.index.names = ["neuron"]
    return unit_metadata


def run(experiment:str="silico_neuropixels", run:str="2023_10_18"):

    # track time
    t0 = time.time()

    # get config
    data_conf, _ = get_config(experiment, run).values()
    PREP_PATH = data_conf["preprocessing"]["output"]["trace_file_path"]
    STUDY_PATH = data_conf["postprocessing"]["waveform"]["sorted"]["study"]
    METADATA_FILE = data_conf["postprocessing"]["sorted_neuron_metadata"]

    # get metadata (12 secs)
    unit_metadata = get_sorted_unit_metadata(data_conf, PREP_PATH, STUDY_PATH, MS_BEFORE, MS_AFTER)

    # save metadata file
    parent_path = os.path.dirname(METADATA_FILE)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    unit_metadata.to_csv(METADATA_FILE, index=True)
    logger.info(f"Done saving units metadata in {time.time()-t0} secs")
    logger.info("Done")