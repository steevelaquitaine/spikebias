"""sort marques silico with klusta
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 12.02.2024
modified: 12.02.2024

WARNING: CANNOT BE RUN BECAUSE KLUSTAKWIK2 REQUIRES AN OLDER VERSION OF
PYTHON THAN 3.9

usage: 

    # setup klusta
    see envs/klusta.txt

    # Submit to cluster
    sbatch cluster/sorting/marques_silico/sort_klusta.sbatch
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
# were set to default parameter for klusta
# found in spikeinterface 0.96.1
sorter_params = {
        'adjacency_radius': None,
        'threshold_strong_std_factor': 5,
        'threshold_weak_std_factor': 2,
        'detect_sign': -1,
        'extract_s_before': 16,
        'extract_s_after': 32,
        'n_features_per_channel': 3,
        'pca_n_waveforms_max': 10000,
        'num_starting_clusters': 50,
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
SORTING_PATH = data_conf["sorting"]["sorters"]["klusta"]["output"]
OUTPUT_PATH = data_conf["sorting"]["sorters"]["klusta"]["klusta_output"]

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
Sorting = ss.run_sorter(sorter_name='klusta', recording=Recording, output_folder=OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
Sorting = Sorting.remove_empty_units()
logger.info("Done running klusta in: %s", round(time() - t0, 1))

# write
shutil.rmtree(SORTING_PATH, ignore_errors=True)
Sorting.save(folder=SORTING_PATH)
logger.info("Done saving klusta in: %s", round(time() - t0, 1))