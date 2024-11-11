"""pipeline to add site layer metadata

author: steeve.laquitaine@epfl.ch

usage:
    
    sbatch cluster/metadata/dense_spont.sbatch

time: takes 1 hour
"""
import os
import numpy as np
import yaml

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.validation.layer import getAtlasInfo, loadAtlasInfo

def label_layers(data_conf, Recording, blueconfig, load_atlas_metadata=True):
    """record electrode site layer property in RecordingExtractor
    
    Args:
        blueconfig (None): is always None
    """

    # load probe.wired trace
    probe = Recording.get_probe()

    # get site layers and curare
    if load_atlas_metadata:
        _, site_layers = loadAtlasInfo(data_conf)
    else:
        _, site_layers = getAtlasInfo(data_conf, blueconfig, probe.contact_positions)
    site_layers = np.array(site_layers)
    site_layers[site_layers == "L2"] = "L2_3"
    site_layers[site_layers == "L3"] = "L2_3"

    # sanity check
    assert len(site_layers) == 128, """site count does not match horvath's probe'"""

    # add metadata to RecordingExtractor
    Recording.set_property('layers', values=site_layers)
    return Recording