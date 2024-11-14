"""pipeline to process the "silico_neuropixels" experiment - run "concatenated" 
(simulation of Marques probe) from raw bbp_workflow simulation files to ready-to-sort 
SpikeInterface's Recording and Sorting Extractors

  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023
modified: 05.07.2024
regression-test: OK!

 usage:

    sbatch cluster/prepro/npx_spont/process_dandihub.sh

Note:
    - if preprocessing write crashes because of memory issue. Rerun with all pipeline nodes
    set to False except PREPROCESS=True
    - the trace array requires 240 GB RAM (we typically have 636 GB available RAM on our compute node)

Duration: 2 h 20

"""
import os
import logging
import logging.config
import yaml
import time
import numpy as np

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

# custom package
from src.nodes.utils import get_config
from src.nodes.prepro import preprocess


# SETUP PARAMETERS
data_conf, param_conf = get_config("silico_neuropixels_on_dandihub", "npx_spont").values()

# SETUP PIPELINE
FIT_CAST = True        # done once then set to False (2h18 min)
OFFSET = True
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = True             # done once then set to False (25 mins)
SAVE_METADATA = True    # True to add new metadata to wired probe
PREPROCESS = True      # True to update after adding new metadata (butterworth: 1h40, wavelet: 3h)
GROUND_TRUTH = True    # done once then set to False

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SETUP PARALLEL PROCESSING
# required, else fit and cast as extractor crashes due to lack of 
# memory
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True}
 

def run(filtering: str="wavelet"):
    """
    args:
        filtering: "butterworth" or "wavelet"
    """
    
    # track time
    t0 = time.time()

    # load RecordingExtractor and Ground truth SortingExtractor
    Recording, Sorting = preprocess.load_on_dandihub(data_conf)

    if FIT_CAST:
        Recording = preprocess.fit_and_cast_as_extractor_npx_probe_on_dandihub(data_conf=data_conf,
                                                                 param_conf=param_conf,
                                                                 offset=OFFSET,
                                                                 scale_and_add_noise=SCALE_AND_ADD_NOISE)
        
    if WIRE:
        preprocess.wire_probe(data_conf=data_conf,
                              param_conf=param_conf,
                              Recording=Recording,
                              blueconfig=None,
                              save_metadata=SAVE_METADATA,
                              job_dict=job_dict, 
                              n_sites=384,
                              load_atlas_metadata=True,
                              load_filtered_cells_metadata=True)
    
    if PREPROCESS:
        preprocess.preprocess_recording_npx_probe(data_conf=data_conf,
                                                  param_conf=param_conf,
                                                  job_dict=job_dict,
                                                  filtering=filtering)
        
    if GROUND_TRUTH:
        preprocess.save_ground_truth_on_dandihub(Sorting, data_conf)

    # report time
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")