"""Pipeline that preprocesses a campaign traces
- wires traces and probe

Usage:

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run
    python3.9 app.py simulation --pipeline preprocess --conf 2023_01_13
    
Returns:
    _type_: _description_
"""

import logging
import logging.config
import shutil
from sys import argv
from time import time

import spikeinterface.full as si
import yaml

from src.nodes.prepro import probe_wiring
from src.nodes.utils import get_config_silico

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def run(dataset_conf: dict, param_conf: dict):
    """run preprocessing pipeline

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # get write path
    WRITE_PATH = dataset_conf["preprocessing"]["output"]["trace_file_path"]

    # get trace
    trace = probe_wiring.run(dataset_conf, param_conf)

    # bandpass
    bandpassed = si.bandpass_filter(trace, freq_min=300, freq_max=6000)

    # set common reference
    referenced = si.common_reference(
        bandpassed, reference="global", operator="median"
    )

    # write
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    referenced.save(folder=WRITE_PATH, format="binary")
    return referenced


def load(data_conf: dict):
    """Load preprocessed recording from config

    Args:
        data_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    return si.load_extractor(
        data_conf["preprocessing"]["output"]["trace_file_path"]
    )


if __name__ == "__main__":

    # start timer
    t0 = time()

    # parse run parameters
    conf_date = argv[1]

    # get config
    data_conf, param_conf = get_config_silico(conf_date).values()

    # run preprocessing
    output = run(data_conf, param_conf)
    logger.info("Preprocessing done in  %s secs", round(time() - t0, 1))
