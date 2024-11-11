"""pipeline to add site layer metadata to silico neuropixels as match to Marques et al dataset 
probe-wired RecordingExtractor

author: steeve.laquitaine@epfl.ch
"""
import os
import yaml 

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.validation.layer import getAtlasInfo

#experiment = "silico_neuropixels"
#run = "2023_10_18"
#data_conf, param_conf = get_config(experiment, run).values()
#WIRED_RECORDING_PATH = data_conf["probe_wiring"]["output"]
#BLUECONFIG = data_conf["dataeng"]["blueconfig"]


def label_layers(Recording, blueconfig):
    """add site layer metadata to silico neuropixels run 2023_10_18 
    probe wired RecordingExtractor

    note: multiprocessing crashes when writing extractor here, so we write
    without.
    """

    # load probe.wired trace
    probe = Recording.get_probe()

    # get site layers
    _, site_layers = getAtlasInfo(blueconfig, probe.contact_positions)
    
    # sanity check
    assert len(site_layers) == 384, """site count does not match neuropixels'"""

    # add metadata to RecordingExtractor
    Recording.set_property('layers', values=site_layers)
    
    # rewrite
    #shutil.rmtree(data_conf["probe_wiring"]["output"], ignore_errors=True)
    #Recording.save(folder=data_conf["probe_wiring"]["output"], format="binary")
    return Recording