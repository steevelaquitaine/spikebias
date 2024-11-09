#!/usr/bin/env python3
# see https://github.com/magland/mountainsort4/tree/main/examples

import logging
import logging.config
import shutil
from time import time
import spikeinterface as si
import spikeinterface.sorters as ss
import yaml
import mountainsort4 as ms4
#import spikeextractors as se
import spikeinterface.extractors as se

# custom package
#from src.nodes.utils import get_config

# SETUP CONFIG
#data_conf, _ = get_config("silico_neuropixels", "2023_10_18").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
RECORDING_PATH = '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataeng/0_silico/neuropixels_lfp_10m_384ch_hex0_40Khz_2023_10_18/be011315-9555-493e-a59c-27f42d1058ed/campaign/recording/traces' # raw recording to read with spikeinterface #data_conf["probe_wiring"]["output"]

# SET WRITE PATHS
SORTING_PATH = '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/sorting/0_silico/neuropixels_lfp_10m_384ch_hex0_40Khz_2023_10_18/be011315-9555-493e-a59c-27f42d1058ed/SortingMountainsort4/' #data_conf["sorting"]["sorters"]["mountainsort4"]["output"]
OUTPUT_PATH = '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/sorting/0_silico/neuropixels_lfp_10m_384ch_hex0_40Khz_2023_10_18/be011315-9555-493e-a59c-27f42d1058ed/Mountainsort4_output/' #data_conf["sorting"]["sorters"]["mountainsort4"]["mountainsort4_output"]

# get Spikeinterface Recording object
t0 = time()
Recording = si.load_extractor(RECORDING_PATH)
logger.info("Done loading recording in: %s", round(time() - t0, 1))


def main():

    #from ipdb import set_trace; set_trace()
    Recording2 = se.NumpyRecording(
        traces_list=[Recording.get_traces().T], # transposed for that version
        sampling_frequency=Recording.sampling_frequency,
    )
    Recording2.set_channel_locations(Recording.get_channel_locations(), channel_ids=Recording.get_channel_ids())
    #recording, sorting_true = se.example_datasets.toy_example()
    #print(recording.get_traces().shape,"!!!!!!")
    
    logger.info("Started sorting...")

    Sorting = ms4.mountainsort4(
        recording=Recording2,
        detect_sign=-1,
        clip_size=50,
        adjacency_radius=20,
        detect_threshold=3,
        detect_interval=10,
        num_workers=None,
        verbose=True,
        use_recording_directly=False
    )
    logger.info("Terminated sorting...")

    # run sorting (default parameters)
    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
    Sorting = Sorting.remove_empty_units()
    shutil.rmtree(SORTING_PATH, ignore_errors=True)
    Sorting.save(folder=SORTING_PATH)
    logger.info("All done")

if __name__ == '__main__':
    main()