"""sort marques stimulus with Kilosort 2.5
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 30.01.2024
modified: 14.02.2024

usage: 

    # Download Kilosort 2.5 release
    wget -c https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.5.tar.gz -O - | tar -xz

    # Submit to cluster
    sbatch cluster/sorting/marques_stimulus/sort_ks2_5.sbatch
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
# were set to default parameter for KS 2.5
# found in spikeinterface 0.96.1
sorter_params = {
        'detect_threshold': 6,  
        'projection_threshold': [10, 4],
        'preclust_threshold': 8,
        'car': True,
        'minFR': 0.1,
        'minfr_goodchannels': 0.1,
        'nblocks': 5,
        'sig': 20,
        'freq_min': 150,
        'sigmaMask': 30,
        'nPCs': 3,
        'ntbuff': 64,
        'nfilt_factor': 4,
        'NT': None,
        'do_correction': True,
        'wave_length': 61,
        'keep_good_only': False,
    }

# SETUP CONFIG
data_conf, _ = get_config("silico_neuropixels", "npx_evoked").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
PACKAGE_PATH = data_conf["sorting"]["sorters"]["kilosort2_5"]["input"]
SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort2_5"]["output"]
OUTPUT_PATH = data_conf["sorting"]["sorters"]["kilosort2_5"]["ks2_5_output"]

# SET KS 2.5 environment variable (you must have downloaded
# Kilosort 2.5 release, see usage above)
ss.Kilosort2_5Sorter.set_kilosort2_5_path(PACKAGE_PATH)

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
Sorting = ss.run_sorter(sorter_name='kilosort2_5', recording=Recording, output_folder=OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
Sorting = Sorting.remove_empty_units()
logger.info("Done running kilosort 2.5 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(SORTING_PATH, ignore_errors=True)
Sorting.save(folder=SORTING_PATH)
logger.info("Done saving kilosort 2.5 in: %s", round(time() - t0, 1))