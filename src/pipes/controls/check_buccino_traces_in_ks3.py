"""sort buccino 2020 recording dataset with ks3
takes 28 min

We do no preprocessing as Kilosort3 already preprocess the traces with 
(see code preprocessDataSub()):

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values
"""
import os
import shutil
from time import time

import spikeinterface as si
import spikeinterface.full as si_full
import spikeinterface.sorters as ss
import yaml

from src.nodes.utils import get_config

# SET PARAMETERS
EXPERIMENT = "buccino_2020" # the experiment
SIMULATION_DATE = "2020"    # the run (date)

# SETUP CONFIG
data_conf, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
KS3_PACKAGE_PATH = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/controls/sorters/Kilosort3_to_check_traces_buccino/"
KS3_SORTING_PATH = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/controls/sorters/Kilosort3_to_check_traces_buccino/SortingKS3/"
KS3_OUTPUT_PATH = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/controls/sorters/Kilosort3_to_check_traces_buccino/KS3_output/"

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# get Spikeinterface Recording object
Recording = si.load_extractor(RECORDING_PATH)

# run sorting
#sorting_KS3 = ss.run_kilosort3(Recording, output_folder=KS3_OUTPUT_PATH, verbose=True)
sorting_KS3 = ss.run_sorter(sorter_name='kilosort3', recording=Recording, output_folder=KS3_OUTPUT_PATH, verbose=True)

# remove empty units (Samuel Garcia's advice)
sorting_KS3 = sorting_KS3.remove_empty_units()

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
# os.makedirs(KS3_SORTING_PATH)
sorting_KS3.save(folder=KS3_SORTING_PATH)