"""pipeline that saves sorted units' metadata to Sorting Extractors

author: steeve.laquitaine@epfl.ch
date: 17.01.2023
modified: 01.02.2024

Usage:

    # on cluster
    sbatch cluster/postpro/horvath_vivo/save_metadata.sbatch

Prerequisites:
    - Recording Extractor contains metadata (site layers)
    
takes 4 min
"""
import sys
import logging
import logging.config
import yaml
import spikeinterface as si
import spikeinterface.postprocessing as spost
import pandas as pd
import shutil
from src.nodes.utils import get_config
from src.nodes.postpro import spikestats
from src.nodes.postpro.horvath_vivo import get_waveforms
from src.nodes.postpro import waveform

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# Setup waveform parameters
MS_BEFORE = 3
MS_AFTER = 3

# Setup pipeline
EXTRACT_WAVEFORM = True # set to true if you have re-run sorting

def save_sorted_units_site_and_layers(
        Sorting,
        exp:str,
        run:str,
        recording_path:str, 
        study_path:str, 
        sorting_path:str, 
        extract_wfs:bool, 
        save:bool
        ):
    """save sorted units site and layers metadata to Sorting Extractor

    Args:
        Sorting: Sorting extractor
        exp: experiment: "vivo_horvath"
        run: run e.g., probe_1
        recording_path (str): _description_
        study_path (str): _description_
        sorting_path (str): _description_
        extract_wfs (bool): _description_
        save (bool): _description_

    Returns:
        _type_: _description_
    """
    # extract waveforms
    if extract_wfs:
        get_waveforms.run(exp, run)

    # load Waveform Extractor
    recording = si.load_extractor(recording_path)
    we = waveform.load(recording, study_path, ms_before=MS_BEFORE, ms_after=MS_AFTER)

    # get channels where spike amplitude is maximal
    # note: channel_ids start from 1 not 0
    max_chids = spost.get_template_extremum_channel(we, peak_sign="both")

    # map units to recording sites
    unit_sites = pd.DataFrame.from_dict(max_chids, orient="index")
    unit_sites.columns = ["site"]

    # get site layers
    Recording = si.load_extractor(recording_path)
    site_layers = pd.DataFrame(
        Recording.get_property("layers"), index=Recording.channel_ids, columns=["layer"]
    )

    # map sorted units to sites and layers
    # site_and_layer (index: site, columns: layer and unit)
    unit_sites = pd.DataFrame.from_dict(max_chids, orient="index")
    unit_sites.columns = ["site"]
    unit_sites["unit"] = unit_sites.index
    unit_sites.index = unit_sites["site"].values
    unit_sites = unit_sites.drop(columns=["site"])
    site_and_layer = pd.merge(site_layers, unit_sites, left_index=True, right_index=True)
    site_and_layer = site_and_layer.sort_values(by="unit")

    # standardize layer names
    site_and_layer["layer"] = site_and_layer["layer"].replace(
        "Outsideofthecortex", "Outside"
    )

    # Write metadata to Sorting Extractor
    # - Sanity check
    # - Save layer as a property
    # - Save nearest site as a property
    assert all(Sorting.unit_ids == site_and_layer["unit"]), "units should match"
    Sorting.set_property("layer", site_and_layer["layer"].tolist())
    Sorting.set_property("contact", site_and_layer.index.tolist())
    
    # save
    if save:
        shutil.rmtree(sorting_path, ignore_errors=True)    
        Sorting.save(folder=sorting_path)    
    return Sorting


if __name__ == "__main__":
    """
    Usage:
        python3.9 ./save_metadata.py vivo_horvath probe_1
    """

    logger.info("Starting save_metadata for Horvath vivo")

    # get config from function call arguments
    argv = sys.argv[1:]
    EXPERIMENT = argv[0]    # e.g., "vivo_horvath"
    RUN = argv[1]           # e.g., "probe_1"

    # get dataset paths
    data_conf, _ = get_config(EXPERIMENT, RUN).values()
    SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]
    RECORDING_PATH = data_conf["preprocessing"]["output"]["trace_file_path"]
    STUDY_PATH = data_conf["postprocessing"]["waveform"]["sorted"]["study"]

    # load sorting extractor
    Sorting = si.load_extractor(SORTING_PATH)

    # save sorted unit firing rates
    Sorting = spikestats.label_firing_rates(Sorting, RECORDING_PATH, SORTING_PATH, save=False)

    # save sorted unit layers and sites
    # - extract spike waveform
    # - get nearest site and layer
    # - save its layer 
    Sorting = save_sorted_units_site_and_layers(Sorting, EXPERIMENT, RUN, RECORDING_PATH, STUDY_PATH, SORTING_PATH, extract_wfs=EXTRACT_WAVEFORM, save=True)
    logger.info("Done")