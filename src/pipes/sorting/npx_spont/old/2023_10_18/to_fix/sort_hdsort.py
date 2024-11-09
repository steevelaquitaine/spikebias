"""sort marques silico with hdsort
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 11.02.2024

usage: 

    # Download hdsort
    git clone https://git.bsse.ethz.ch/hima_public/HDsort.git

    # Submit to cluster
    sbatch cluster/sorting/marques_silico/sort_hdsort.sbatch
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
# were set to default parameter for KS
# found in spikeinterface 0.96.1
sorter_params = {
        'detect_threshold': 4.2,
        'detect_sign': -1,  # -1 - 1
        'filter': True,
        'parfor': True,
        'freq_min': 300,
        'freq_max': 7000,
        'max_el_per_group': 9,
        'min_el_per_group': 1,
        'add_if_nearer_than': 20,
        'max_distance_within_group': 52,
        'n_pc_dims': 6,
        'chunk_size': 500000,
        'loop_mode': 'local_parfor',
        'chunk_memory': '500M'
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
PACKAGE_PATH = data_conf["sorting"]["sorters"]["hdsort"]["input"]
SORTING_PATH = data_conf["sorting"]["sorters"]["hdsort"]["output"]
OUTPUT_PATH = data_conf["sorting"]["sorters"]["hdsort"]["hdsort_output"]

# SET hdsort environment variable (you must have downloaded hdsort)
ss.HDSortSorter.set_hdsort_path(PACKAGE_PATH)

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
Sorting = ss.run_sorter(sorter_name='hdsort', recording=Recording, output_folder=OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
Sorting = Sorting.remove_empty_units()
logger.info("Done running hdsort in: %s", round(time() - t0, 1))

# write
shutil.rmtree(SORTING_PATH, ignore_errors=True)
Sorting.save(folder=SORTING_PATH)
logger.info("Done saving hdsort in: %s", round(time() - t0, 1))