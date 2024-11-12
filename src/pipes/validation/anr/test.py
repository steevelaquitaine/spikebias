"""Test parallel computing on multiple nodes with mpi

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/validation/main/snr/test.sh
    
Duration: 1 min
"""

# import libs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpi4py import MPI
import logging
import logging.config
import yaml
import time 

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

def main(rank, n_ranks):

    # initialize the data
    x = np.zeros(100)
    y = np.zeros(100)
    
    # create x and y on each node
    if rank == 0:
        logger.info("Calculated x")
        x = np.arange(0, 100, 1)
    elif rank == 1:
        logger.info("Calculated y")
        y = np.arange(0, 100, 1)
    
    # gather x and y on master node
    x = comm.gather(x, root=0)
    y = comm.gather(y, root=0)
    
    # plot y against x on master node
    if rank == 0:
        _, ax = plt.subplots(1, figsize=(5, 5))
        ax.plot(x, y)
        plt.savefig("test.svg")
        logger.info("Saved plot")
        
# track time
t0 = time.time()

# get api
comm = MPI.COMM_WORLD

# get the rank of this node
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# run
main(rank, n_ranks)
logger.info(f"All done in {np.round(time.time()-t0,2)} secs")