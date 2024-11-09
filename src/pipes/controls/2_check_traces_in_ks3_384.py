"""sort npx384 simulation with ks3
takes 4:30 min

"""
import os
import shutil

import spikeinterface as si
import spikeinterface.full as si_full
import spikeinterface.sorters as ss

from src.nodes.utils import get_config

# SET PARAMETERS
EXPERIMENT = "silico_neuropixels"
SIMULATION_DATE = "2023_08_17"

# SETUP CONFIG
data_conf, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
KS3_PACKAGE_PATH = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/controls/sorters/Kilosort3_to_check_traces_384/"
KS3_SORTING_PATH = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/controls/sorters/Kilosort3_to_check_traces_384/SortingKS3/"
KS3_OUTPUT_PATH = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/controls/sorters/Kilosort3_to_check_traces_384/KS3_output/"

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# sort
Recording = si.load_extractor(RECORDING_PATH)
sorting_KS3 = ss.run_sorter(sorter_name='kilosort3', recording=Recording, output_folder=KS3_OUTPUT_PATH, verbose=True)

# remove empty units (Samuel Garcia's advice)
sorting_KS3 = sorting_KS3.remove_empty_units()

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
sorting_KS3.save(folder=KS3_SORTING_PATH)