"""Pipeline that extracts Kilosort3's universal templates

Usage:
   
First setup 

    # activate the python virtual environment
    # load matlab
    # setup GPU for matlab (Kilosort run in Matlab)
    # run sorting (from the terminal)
    
    source env_kilosort_silico/bin/activate
    module load unstable hpe-mpi/2.25.hmpt matlab/r2019b
    cd sorters_packages/Kilosort2/CUDA/
    matlab -batch mexGPUall  # make sure /tmp is cleaned (enough space)
    cd ../../../
    python3.9 -m src.pipes.postpro.univ_temp
    
References:    
    (1) https://spikeinterface.github.io/blog/ground-truth-comparison-and-ensemble-sorting-of-a-synthetic-neuropixels-recording/

notes: 
    matlab code entry point: '/gpfs/bbp.cscs.ch/data/project/proj68/home/laquitai/spike-sorting/kilosort3_output/kilosort3_master.m'
    one needs to delete /kilosort_output
"""

import logging
import logging.config
from time import time

import scipy.io
import spikeinterface.sorters as ss
import yaml

from src.nodes.prepro import preprocess
from src.nodes.utils import get_config

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# Set experiment and run parameters
EXP = "silico_neuropixels"
SIMULATION_DATE = "2023_02_19"
CREATE_RECORDING = False
LOAD_RAW_RECORDING = True
PREP_RECORDING = False
KILOSORT_3 = True

# set plot parameters
TITLE = "virtual neuropixels-32"

# get config
data_conf, param_conf = get_config(EXP, SIMULATION_DATE).values()


def get_KS3_univ_temp(
    trace_recording,
    data_conf: dict,
):
    """get kilosort3's universal templates

    Args:
        trace_recording (_type_): _description_
        data_conf (dict): dataset paths

    Returns:
        np.ndarray: six row x ntimepoints column spike waveforms
    """

    # set kilosort3 path
    ss.Kilosort3Sorter.set_kilosort3_path(
        data_conf["sorting"]["sorters"]["kilosort3"]["input"]
    )
    t0 = time()

    # run sorting
    ss.run_kilosort3(trace_recording, verbose=True, with_output=False)

    # log
    logger.info(
        "inspecting kilosort3 sorting steps - done in %s",
        round(time() - t0, 1),
    )
    t0 = time()

    # return extracted universal templates
    mat = scipy.io.loadmat("kilosort3_output/steeve/data/univ_templates.mat")
    return mat["rez"]["wTEMP"][0][0]


def run(data_conf: dict):
    """run pipeline

    Args:
        data_conf (dict): dataset paths

    Returns:
        _type_: universal templates
    """

    # load preprocessed recording
    trace_recording = preprocess.load(data_conf)

    # get universal templates
    return get_KS3_univ_temp(trace_recording, data_conf)


def load(data_conf: dict):
    mat = scipy.io.loadmat(data_conf["postprocessing"]["universal_templates"])
    return mat["rez"]["wTEMP"][0][0]


if __name__ == "__main__":
    univ_templates = run(data_conf)
