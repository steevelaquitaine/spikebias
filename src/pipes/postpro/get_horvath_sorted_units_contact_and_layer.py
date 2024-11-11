"""Pipeline to get in vivo Horvath sorted units' metadata (10 min)
author: steeve.laquitaine@epfl.ch

Usage:
    sbatch cluster/postpro/get_horvath_sorted_units_metadata.sbatch

Returns:
    csv: write a .csv metadata file for depth files 1,2,3

prerequisites: 
    Horvath recordings have been sorted a SortingExtractors exist
"""

import spikeinterface.postprocessing as spost
import spikeinterface as si 
import os 
import pandas as pd
from pynwb import NWBHDF5IO
import copy
import numpy as np
import logging
import logging.config
import yaml
import time

# set project path
# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.postpro import waveform

# Setup experiment read/write paths
EXPERIMENT = "vivo_horvath"   
SIMULATION_DATE = "2021_file_1"

RAW_HORVATH_1 = "/gpfs/bbp.cscs.ch/project/proj68/scratch/imbeni/sorting/dataset/horvath/Rat01/Insertion1/Depth1/Rat01_Insertion1_Depth1.nwb"
data_conf_horvath_1, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()
RECORDING_PATH_HORVATH_1 = data_conf_horvath_1["preprocessing"]["output"]["trace_file_path"]
SORTED_PATH_HORVATH_1 = data_conf_horvath_1["sorting"]["sorters"]["kilosort3"]["output"]
SORTED_FR_FILE_PATH_HORVATH_1 = data_conf_horvath_1["features"]["sorted_ks3"]["firing_rates"]
STUDY_FOLDER_HORVATH_1 = data_conf_horvath_1["postprocessing"]["waveform"]["study"]
SORTED_NEURON_METADATA_FILE_1 = data_conf_horvath_1["postprocessing"]["sorted_neuron_metadata"]

SIMULATION_DATE = "2021_file_2"
RAW_HORVATH_2 = "/gpfs/bbp.cscs.ch/project/proj68/scratch/imbeni/sorting/dataset/horvath/Rat01/Insertion1/Depth2/Rat01_Insertion1_Depth2.nwb"
data_conf_horvath_2, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()
RECORDING_PATH_HORVATH_2 = data_conf_horvath_2["preprocessing"]["output"]["trace_file_path"]
SORTED_PATH_HORVATH_2 = data_conf_horvath_2["sorting"]["sorters"]["kilosort3"]["output"]
SORTED_FR_FILE_PATH_HORVATH_2 = data_conf_horvath_2["features"]["sorted_ks3"]["firing_rates"]
STUDY_FOLDER_HORVATH_2 = data_conf_horvath_2["postprocessing"]["waveform"]["study"]
SORTED_NEURON_METADATA_FILE_2 = data_conf_horvath_2["postprocessing"]["sorted_neuron_metadata"]

SIMULATION_DATE = "2021_file_3"
RAW_HORVATH_3 = "/gpfs/bbp.cscs.ch/project/proj68/scratch/imbeni/sorting/dataset/horvath/Rat01/Insertion1/Depth3/Rat01_Insertion1_Depth3.nwb"
data_conf_horvath_3, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()
RECORDING_PATH_HORVATH_3 = data_conf_horvath_3["preprocessing"]["output"]["trace_file_path"]
SORTED_PATH_HORVATH_3 = data_conf_horvath_3["sorting"]["sorters"]["kilosort3"]["output"]
SORTED_FR_FILE_PATH_HORVATH_3 = data_conf_horvath_3["features"]["sorted_ks3"]["firing_rates"]
STUDY_FOLDER_HORVATH_3 = data_conf_horvath_3["postprocessing"]["waveform"]["study"]
SORTED_NEURON_METADATA_FILE_3 = data_conf_horvath_3["postprocessing"]["sorted_neuron_metadata"]

# setup waveform parameters
MS_BEFORE = 3
MS_AFTER = 3

# setup pipeline
COMPUTE_WAVEFORM_EXTRACTORS = True

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def read_file_metadata(file_path):
    """read nwb file (provides access to all the metadata in contrast to SpikeInterface's
    NwbRecordingExtractor

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    return NWBHDF5IO(file_path, mode="r").read()


def get_sorted_neuron_metadata(RAW_HORVATH_1, RECORDING_PATH_HORVATH_1, STUDY_FOLDER_HORVATH_1, MS_BEFORE, MS_AFTER):
    """get sorted units' metadata (contact, layer)

    Args:
        RAW_HORVATH_1 (_type_): _description_
        RECORDING_PATH_HORVATH_1 (_type_): _description_
        STUDY_FOLDER_HORVATH_1 (_type_): _description_
        MS_BEFORE (_type_): _description_
        MS_AFTER (_type_): _description_

    Returns:
        _type_: _description_
    """
    # load WaveformExtractor
    recording = si.load_extractor(RECORDING_PATH_HORVATH_1)
    we = waveform.load(recording, STUDY_FOLDER_HORVATH_1, ms_before=MS_BEFORE, ms_after=MS_AFTER)

    # get channels where spike amplitude is maximal
    max_chids = spost.get_template_extremum_channel(
        we, peak_sign="both"
    )

    # map neurons to contacts
    neuron_contacts = pd.DataFrame.from_dict(max_chids, orient='index')

    # map neurons, contacts and layers
    Recording_depth_1 = read_file_metadata(RAW_HORVATH_1)
    df = Recording_depth_1.electrodes.to_dataframe()
    contact_layers_depth_1 = df.location.apply(lambda x: x.decode("utf-8").split(',')[-1])
    neuron_metadata = copy.copy(neuron_contacts)
    neuron_metadata["layer"] = np.nan
    for ix in neuron_contacts.index:    
        contact = neuron_contacts.loc[ix,0]
        layer = contact_layers_depth_1.loc[contact]
        neuron_metadata.loc[ix, "layer"] = layer

    # construct metadata table
    neuron_metadata.columns = ["contact", "layer"]
    neuron_metadata.index.names = ["neuron"]
    return neuron_metadata


# get metadata (12 secs)
t0 = time.time()
logger.info(f"Getting units metadata ...")
neuron_metadata_1 = get_sorted_neuron_metadata(RAW_HORVATH_1, RECORDING_PATH_HORVATH_1, STUDY_FOLDER_HORVATH_1, MS_BEFORE, MS_AFTER)
neuron_metadata_2 = get_sorted_neuron_metadata(RAW_HORVATH_2, RECORDING_PATH_HORVATH_2, STUDY_FOLDER_HORVATH_2, MS_BEFORE, MS_AFTER)
neuron_metadata_3 = get_sorted_neuron_metadata(RAW_HORVATH_3, RECORDING_PATH_HORVATH_3, STUDY_FOLDER_HORVATH_3, MS_BEFORE, MS_AFTER)
logger.info(f"Done getting units metadata  in {time.time()-t0} secs")

# write to file
logger.info(f"Saving to csv ...")
neuron_metadata_1.to_csv(SORTED_NEURON_METADATA_FILE_1, index=True)
neuron_metadata_2.to_csv(SORTED_NEURON_METADATA_FILE_2, index=True)
neuron_metadata_3.to_csv(SORTED_NEURON_METADATA_FILE_3, index=True)
logger.info(f"Done saving to csv in {time.time()-t0} secs")
logger.info("Done")