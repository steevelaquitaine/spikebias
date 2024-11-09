"""sort horvath vivo probe 1 with herdingspikes
takes 28 min

  author: steeve.laquitaine@epfl.ch
    date: 12.02.2024
modified: 14.02.2024

usage: 

    # Download hdsort
    pip install herdingspikes

    # Submit to cluster
    sbatch cluster/sorting/horvath_vivo/probe_1/sort_herdingspikes.sbatch
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
        # core params
        'clustering_bandwidth': 5.5,  # 5.0,
        'clustering_alpha': 5.5,  # 5.0,
        'clustering_n_jobs': -1,
        'clustering_bin_seeding': True,
        'clustering_min_bin_freq': 16,  # 10,
        'clustering_subset': None,
        'left_cutout_time': 0.3,  # 0.2,
        'right_cutout_time': 1.8,  # 0.8,
        'detect_threshold': 20,  # 24, #15,

        # extra probe params
        'probe_masked_channels': [],
        'probe_inner_radius': 70,
        'probe_neighbor_radius': 90,
        'probe_event_length': 0.26,
        'probe_peak_jitter': 0.2,

        # extra detection params
        't_inc': 100000,
        'num_com_centers': 1,
        'maa': 12,
        'ahpthr': 11,
        'out_file_name': "HS2_detected",
        'decay_filtering': False,
        'save_all': False,
        'amp_evaluation_time': 0.4,  # 0.14,
        'spk_evaluation_time': 1.0,

        # extra pca params
        'pca_ncomponents': 2,
        'pca_whiten': True,

        # bandpass filter
        'freq_min': 300.0,
        'freq_max': 6000.0,
        'filter': True,

        # rescale traces
        'pre_scale': True,
        'pre_scale_value': 20.0,

        # remove duplicates (based on spk_evaluation_time)
        'filter_duplicates': True
    }

# SETUP CONFIG
data_conf, _ = get_config("vivo_horvath", "probe_1").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
SORTING_PATH = data_conf["sorting"]["sorters"]["herdingspikes"]["output"]
OUTPUT_PATH = data_conf["sorting"]["sorters"]["herdingspikes"]["herdingspikes_output"]

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))

# run sorting (default parameters)
t0 = time()
Sorting = ss.run_sorter(sorter_name='herdingspikes', recording=Recording, output_folder=OUTPUT_PATH, verbose=True, **sorter_params)

# remove empty units (Samuel Garcia's advice)
Sorting = Sorting.remove_empty_units()
logger.info("Done running herdingspikes in: %s", round(time() - t0, 1))

# write
shutil.rmtree(SORTING_PATH, ignore_errors=True)
Sorting.save(folder=SORTING_PATH)
logger.info("Done saving herdingspikes in: %s", round(time() - t0, 1))