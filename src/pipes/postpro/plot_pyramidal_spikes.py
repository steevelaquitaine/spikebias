
"""save pyramidal spike waveform plots
author: steeve.laquitaine@epfl.ch

Usage:
    
    # run on cluster via the terminal
    sbatch cluster/plot_pyramidal_spikes.sbatch

"""
import logging
import logging.config

# SETUP PACKAGES
import os
import yaml
from matplotlib import pyplot as plt
from src.nodes.prepro import preprocess

# SET PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj68/home/laquitai/bernstein_2023"
os.chdir(PROJ_PATH)

from src.nodes.load import load_campaign_params
from src.nodes.postpro import waveform
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
from src.nodes.utils import get_config

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET PARAMETERS
EXPERIMENT = "silico_neuropixels"  # specifies the experiment
SIMULATION_DATE = "2023_08_17"  # specifies the run (date)
MS_BEFORE = 3  # ms
MS_AFTER = 3  # ms

# SETUP CONFIG
data_conf, param_conf = get_config(EXPERIMENT, SIMULATION_DATE).values()

# SETUP PATHS
SPIKE_FILE_PATH = data_conf["dataeng"]["campaign"]["output"]["spike_file_path"]
RAW_LFP_TRACE_FILE_PATH = data_conf["dataeng"]["campaign"]["output"][
    "trace_file_path"
]

# SET WAVEFORM FOLDER
STUDY_FOLDER = data_conf["postprocessing"]["waveform"]["study"]

# FIGURE
FIG_PATH = data_conf["figures"]["silico"]["all"]["pyramidal_waveforms"]


def run():
    """_summary_
    """
    # load recording
    lfp_recording = preprocess.load(data_conf)

    # load waveform data
    # (this takes 1 hour for 2000 near-contact units)
    we = waveform.load(
        lfp_recording,
        study_folder=STUDY_FOLDER,
        ms_before=MS_BEFORE,
        ms_after=MS_AFTER,
    )
    logger.info("loaded waveforms")
    logger.info(f"unit (count): {len(we.unit_ids)}")

    # select near-contact pyramidal cells
    simulation = load_campaign_params(data_conf)

    # load ground truth sorting extractor
    Truth = ground_truth.load(data_conf)
    logger.info(
        "loaded ground truth sorting",
    )
    
    # unit-test 
    assert len(Truth.unit_ids)==len(we.unit_ids), "unit count mismatch"

    # get morphological classes
    cell_morph = simulation["circuit"].cells.get(
        Truth.unit_ids, properties=["morph_class"]
    )

    # get cell ids
    CELL_IDS = cell_morph[cell_morph["morph_class"] == "PYR"].index.values

    # takes 15 min for 472 units
    # save pyramidal cell plots
    for cell_i in CELL_IDS:
        
        # plot
        fig = waveform.plot(
            WaveformExtractor=we, cell_id=cell_i, 
            colors=[(0.3, 0.3, 0.3), (1, 0, 0)], 
            linewidth_instance=2, linewidth_mean=5
        )

        # save
        if not os.path.isdir(FIG_PATH):
            os.makedirs(FIG_PATH)
        fig.savefig(
            f"{FIG_PATH}cell_{cell_i}", dpi=300, bbox_inches="tight"
        )

        # no display
        plt.close()
    
    # log 
    logger.info(
        "saved all plots",
    )

if __name__ == "__main__":
    we = run()
    logger.info(
        "done",
    )