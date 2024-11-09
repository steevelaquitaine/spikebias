"""sort marques silico with Kilosort
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 11.02.2024
modified: 14.02.2024

usage: 

    # Download Kilosort release
    git clone https://github.com/cortex-lab/KiloSort.git

    # Submit to cluster
    sbatch cluster/sorting/marques_stimulus/sort_ks.sbatch
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
        'detect_threshold': 6,
        'car': True,
        'useGPU': True,
        'freq_min': 300,
        'freq_max': 6000,
        'ntbuff': 64,
        'Nfilt': None,
        'NT': None,
        'wave_length': 61,
    }

# SETUP CONFIG
data_conf, _ = get_config("silico_neuropixels", "stimulus").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort"]["input"]
SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort"]["output"]
OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort"]["ks_output"]

# SET KS 2 environment variable (you must have downloaded
# Kilosort release, see usage above)
ss.KilosortSorter.set_kilosort_path(PACKAGE_PATH)

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
Sorting = ss.run_sorter(sorter_name='kilosort', recording=Recording, output_folder=OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
Sorting = Sorting.remove_empty_units()
logger.info("Done running kilosort in: %s", round(time() - t0, 1))

# write
shutil.rmtree(SORTING_PATH, ignore_errors=True)
Sorting.save(folder=SORTING_PATH)
logger.info("Done saving kilosort in: %s", round(time() - t0, 1))