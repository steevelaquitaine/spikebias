"""Pipeline that stacks spike and trace chunks from scratch into a campaign
on local node

Usage:

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run pipeline on local node
    python3.9 app.py simulation --exp silico_neuropixels --pipeline stack --conf 2023_02_19

Returns:
    _type_: Stacked spikes over all simulations of a campaign
"""

import logging
import logging.config
import sys
from time import time

import yaml

from src.nodes.dataeng.silico import campaign_stacking, chunk_stacking
from src.nodes.io.load import load_campaign_params
from src.nodes.utils import get_config

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def run(dataset_conf: dict, blue_config: dict):
    """stack all chunks into one campaign file

    Args:
        dataset_conf (dict): _description_
    """
    # start timer
    t_0 = time()

    # run
    chunk_stacking.run(dataset_conf)
    campaign_stacking.run(dataset_conf, blue_config)

    # log
    logger.info("Campaign stacked in %s secs.", round(time() - t_0, 1))


if __name__ == "__main__":
    # get parameters
    for arg_i, argv in enumerate(sys.argv):
        if argv == "--exp":
            EXPERIMENT = sys.argv[arg_i + 1]
        if argv == "--conf":
            conf_date = sys.argv[arg_i + 1]

    # get run config
    data_conf, param_conf = get_config(EXPERIMENT, conf_date).values()

    # load campaign params
    campaign_params = load_campaign_params(data_conf)

    # run
    run(data_conf, campaign_params["blue_config"])
