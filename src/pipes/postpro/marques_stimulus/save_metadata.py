"""pipeline that saves ground truth and sorted units' metadata to Sorting Extractors

  author: steeve.laquitaine@epfl.ch
    date: 17.01.2023
modified: 30.01.2024

Usage:

    # on cluster
    sbatch cluster/postpro/marques_stimulus/save_metadata.sbatch

takes 4 min
"""
import sys
import logging
import logging.config
import yaml
import spikeinterface as si
from src.nodes.postpro.marques_stimulus import get_sorted_units_site_and_layer, get_waveforms

from src.nodes.utils import get_config
from src.nodes.postpro import biases, layer, spikestats, cell_type, accuracy
from src.nodes.postpro import layer

# Setup pipeline
EXTRACT_WAVEFORM = True

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")



if __name__ == "__main__":
    """
    Usage:
        python3.9 ./save_metadata.py silico_neuropixels stimulus
    """
    logger.info("Starting save_metadata for marques stimulus")

    # get config from function call arguments
    argv = sys.argv[1:]
    EXPERIMENT = argv[0]    # e.g., "silico_neuropixels"
    RUN = argv[1]           # e.g., "stimulus"

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