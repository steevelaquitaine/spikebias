"""pipeline to add site layer metadata

author: steeve.laquitaine@epfl.ch

usage:
    
    sbatch cluster/metadata/dense_spont.sbatch

time: takes 1 hour
"""
import os
import numpy as np

# set project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.validation.layer import getAtlasInfo


def label_layers(Recording, blueconfig):
    """add site layer metadata probe wired RecordingExtractor
    """

    # load probe.wired trace
    probe = Recording.get_probe()

    # get site layers and curare
    _, site_layers = getAtlasInfo(blueconfig, probe.contact_positions)
    site_layers = np.array(site_layers)
    site_layers[site_layers == "L2"] = "L2_3"
    site_layers[site_layers == "L3"] = "L2_3"

    # sanity check
    assert len(site_layers) == 128, """site count does not match horvath's probe'"""

    # add metadata to RecordingExtractor
    Recording.set_property('layers', values=site_layers)
    return Recording