"""pipeline to add site layer metadata to vivo Marques probe-wired RecordingExtractor

author: steeve.laquitaine@epfl.ch

"""
import os
import numpy as np
import yaml 

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config

_, param_conf = get_config("vivo_marques", "c26").values()


def save_layer_metadata(Recording):
    """add site layer metadata to Marques 
    probe wired RecordingExtractor
    """
    # infer layer from site depth
    # - flip to get the real depth and site order (with Pial at the top and WM at the bottom) as Marques' sites ids are ordered from the bottom (WM) to the top (Pial)
    probe = Recording.get_probe()
    site_depth = np.flip(probe.contact_positions[:, 1]) # flip depth (y)

    # get layer borders
    layer_end = np.array(
        [
            param_conf["layer_border"]["L1"],
            param_conf["layer_border"]["L2_3"],
            param_conf["layer_border"]["L4"],
            param_conf["layer_border"]["L5"],
            param_conf["layer_border"]["L6"],
        ]
    )

    # set layers borders
    layers = ["L1", "L2_3", "L4", "L5", "L6", "WM"]
    layer_start = np.hstack([0, layer_end])[:-1]

    # find site layers
    site_layers = []
    for s_i in range(len(site_depth)):
        for l_i in range(len(layer_start)):
            if (site_depth[s_i] >= layer_start[l_i]) and (site_depth[s_i] < layer_end[l_i]):
                site_layers.append(layers[l_i])
        # case white matter
        if site_depth[s_i] >= layer_end[-1]:
            site_layers.append(layers[-1])

    # sanity check
    assert len(site_layers) == 384, """site count does not match neuropixels'"""

    # add metadata to Wired Recording Extractor
    Recording.set_property('layers', values=site_layers)
    return Recording