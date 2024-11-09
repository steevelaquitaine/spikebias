"""sort marques silico with mountainsort4
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 12.02.2024
modified: 12.02.2024

usage: 

    # setup mountainsort4 (see envs/mountainsort4.txt)

    # Submit to cluster
    sbatch cluster/sorting/marques_silico/sort_mountainsort4.sbatch
"""
import logging
import logging.config
import shutil
from time import time
import spikeinterface as si
import spikeinterface.sorters as ss
import yaml

# custom package
from src.nodes.utils import get_config

# SET PARAMETERS
# were set to default parameter for mountainsort4
# found in spikeinterface 0.96.1
# sorter_params = {
#         'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
#         'adjacency_radius': -1,  # Use -1 to include all channels in every neighborhood
#         'freq_min': 300,  # Use None for no bandpass filtering
#         'freq_max': 6000,
#         'filter': True,
#         'whiten': True,  # Whether to do channel whitening as part of preprocessing
#         'num_workers': None, # 1
#         'clip_size': 50,
#         'detect_threshold': 3,
#         'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
#         'tempdir': None,
#         'verbose': True
#     }
sorter_params = {
        'detect_sign': -1,
        'clip_size': 50,
        'adjacency_radius': 20,
        'detect_threshold': 3,
        'detect_interval': 10,
        'num_workers': None,
        #'use_recording_directly': False
}

# SETUP CONFIG
data_conf, _ = get_config("silico_neuropixels", "2023_10_18").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
SORTING_PATH = data_conf["sorting"]["sorters"]["mountainsort4"]["output"]
OUTPUT_PATH = data_conf["sorting"]["sorters"]["mountainsort4"]["mountainsort4_output"]

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
Sorting = ss.run_sorter(sorter_name='mountainsort4', recording=Recording, output_folder=OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
Sorting = Sorting.remove_empty_units()
logger.info("Done running mountainsort4 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(SORTING_PATH, ignore_errors=True)
Sorting.save(folder=SORTING_PATH)
logger.info("Done saving mountainsort4 in: %s", round(time() - t0, 1))