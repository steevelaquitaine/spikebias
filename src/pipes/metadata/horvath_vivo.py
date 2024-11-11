"""pipeline to add site layer metadata to vivo Horvath probe-wired RecordingExtractor

author: steeve.laquitaine@epfl.ch
date: 19.12.2023

usage:
    
    sbatch cluster/metadata/label_layers_horvath_vivo.sbatch
"""
import os
import numpy as np
import spikeinterface as si
import shutil
from pynwb import NWBHDF5IO
import yaml 

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config



def read_file_metadata(file_path):
    """read nwb file (provides access to all the metadata in contrast to SpikeInterface's
    NwbRecordingExtractor

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    return NWBHDF5IO(file_path, mode="r").read()


def label_layers(Wired, experiment: str="vivo_horvath", run: str="probe_1"):
    """add site layer metadata to Horvath's 
    probe-wired RecordingExtractor
    """

    data_conf, _ = get_config(experiment, run).values()
    RAW_RECORDING_PATH = data_conf["raw"]
        
    # list metadata
    Recording_depth_1 = read_file_metadata(RAW_RECORDING_PATH)

    # list electrodes metadata
    df = Recording_depth_1.electrodes.to_dataframe()

    # get and curate sites' layer labels
    site_layers = df.location.apply(lambda x: x.decode("utf-8").split(", ")[-1]).tolist()
    site_layers = np.array(site_layers)
    site_layers[site_layers == "Outside of the cortex"] = "Outside"
    site_layers[site_layers == "L2/3"] = "L2_3"

    # sanity check
    assert len(site_layers) == 128, """site count does not match horvath probe'"""

    # add metadata to RecordingExtractor
    Wired.set_property("layers", values=site_layers)
    return Wired



def run(experiment:str="vivo_horvath", run:str="probe_1"):
    """run pipeline

    Args:
        experiment (str, optional): _description_. Defaults to "vivo_horvath".
        run (str, optional): _description_. Defaults to "probe_1".
    """
    label_layers(experiment, run)