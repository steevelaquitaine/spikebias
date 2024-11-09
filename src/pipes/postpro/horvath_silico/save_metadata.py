
"""pipeline that saves ground truth and sorted units' metadata to Sorting Extractors
for silico Horvath

author: steeve.laquitaine@epfl.ch

usage:
    
    sbatch cluster/postpro/horvath_silico/postpro_probe1.sbatch
    sbatch cluster/postpro/horvath_silico/postpro_probe2.sbatch
    sbatch cluster/postpro/horvath_silico/postpro_probe3.sbatch

author: steeve.laquitaine@epfl.ch
date: 01.02.2024
"""
import sys
import os
import spikeinterface as si
import logging
import logging.config
import yaml
from src.nodes.utils import get_config
from src.nodes.postpro import biases, spikestats, cell_type, accuracy
from src.nodes.postpro.horvath_silico import get_waveforms, get_sorted_units_site_and_layer
from src.nodes.postpro import layer

# Setup pipeline
EXTRACT_WAVEFORM = True

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


if __name__=="__main__":
    
    # get config
    argv = sys.argv[1:]
    EXPERIMENT = argv[0] # e.g., "silico_horvath"
    RUN = argv[1]        # e.g., "concatenated/probe_1"

    data_conf, _ = get_config(EXPERIMENT, RUN).values()
    SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]
    GT_SORTING_PATH = data_conf["sorting"]["simulation"]["ground_truth"]["output"]
    RECORDING_PATH = data_conf["probe_wiring"]["input"]
    BLUECONFIG_PATH = data_conf["dataeng"]["blueconfig"]
    
    # load Ground truth sorting extractor
    SortingTrue = si.load_extractor(GT_SORTING_PATH)

    # load sorting extractor
    Sorting = si.load_extractor(SORTING_PATH)

    # SORTED UNITS METADATA ---------------
    
    # - saves false positive label metadata
    Sorting = biases.label_false_positives(Sorting, SortingTrue, SORTING_PATH, save=False)

    # - saves firing rates metadata
    Sorting = spikestats.label_firing_rates(Sorting, RECORDING_PATH, SORTING_PATH, save=False)

    # saves sorted units' layer and site metadata
    # - extract spike waveform
    # - get nearest site and layer
    # - save its layer 
    if EXTRACT_WAVEFORM:
        get_waveforms.run(EXPERIMENT, RUN)
    get_sorted_units_site_and_layer.run(EXPERIMENT, RUN)            
    Sorting = layer.label_sorted_unit_layer_simulations(Sorting, EXPERIMENT, RUN, save=True)  
    
    # TRUE UNIT METADATA -----------------

    # - saves true units' properties metadata
    SortingTrue = cell_type.label_true_cell_properties(SortingTrue, BLUECONFIG_PATH, GT_SORTING_PATH, save=False)

    # - saves true units'firing rates metadata
    SortingTrue = spikestats.label_firing_rates(SortingTrue, RECORDING_PATH, GT_SORTING_PATH, save=False)

    # - saves true units' sorting accuracies metadata
    accuracy.label_sorting_accuracies(Sorting, SortingTrue, GT_SORTING_PATH, save=True)