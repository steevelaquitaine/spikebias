"""pipeline that applies sorting with kilosort2 and 3 on 
to the run "2023_01_13" of the simulation experiment "silico_neuropixels",
of an LFP recording with a virtual neuropixel probe with 32 contacts 
The experiment's run is configured in conf/silico_reyes/2023_01_13/).

Usage:
   
First setup 

    # clone kilosort2 commit version from Buccino 2020 in sorters_packages/
    git clone https://github.com/MouseLand/Kilosort2
    git checkout -q 48bf2b81d8ad

    # activate spack environment (needed to run Kilosort in Matlab)
    module load unstable hpe-mpi/2.25.hmpt matlab/r2019b
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    export SPACK_SYSTEM_CONFIG_PATH=/gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/config
    export SPACK_USER_CACHE_PATH=/gpfs/bbp.cscs.ch/home/laquitai/spack_install
    spack env create kilosort2_silico_env2 spack_kilosort_silico.yaml
    spack env activate spack_kilosort_silico_env -p
    spack install
    spack load python@3.9.7

    # setup the python virtual environment
    # rm -rf ~/.cache/pip  # clear pip cache
    python3.9 -m venv env_kilosort_silico
    source env_kilosort_silico/bin/activate
    pip3.9 install -r requirements_kilosort_silico.txt

    # install bluepy to read in-silico simulation
    Move to your home directory e.g., /gpfs/bbp.cscs.ch/home/laquitai (where you have have pip install permission).
    Then clone bluepy and checkout branch lfp-reports. Make sure your ssh private key has been setup before for authentication to Gitlab.

    ```bash
    cp -r /gpfs/bbp.cscs.ch/home/tharayil/bluepy-configfile ~/       # copy bluepy-config
    pip3.9 install ~/bluepy-configfile/                              # install
    git clone git@bbpgitlab.epfl.ch:nse/bluepy.git ~/ .              # clone in home path
    cd ~/bluepy/
    git checkout lfp-reports                                         # checkout the lfp package
    pip3.9 install .                                                 # install
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/ # move back to project path
    ```

    # setup GPU for matlab (Kilosort run in Matlab)
    cd sorters_packages/Kilosort2/CUDA/
    matlab -batch mexGPUall                                          # make sure /tmp is cleaned (enough space)
    cd ../../../
    
    # run sorting (from the terminal)
    python3 -m src.pipes.figures.run_population_bias_silico_reyes_2023_01_13

    # or run interactively in Jupyter notebook or ipython
    from src.pipes.figures import run_population_bias_silico_reyes_2023_01_13; fig = run_population_bias_silico_reyes_2023_01_13.run()

After first run: 

    # activate environment (python3.9)
    source env_kilosort_silico/bin/activate

    # sort
    python3 -m src.pipes.sorting.run_population_bias_silico_reyes_2023_01_13

    # or in Jupyter notebook
    from src.pipes.sorting import run_population_bias_silico_reyes_2023_01_13; fig = run_population_bias_silico_reyes_2023_01_13.run()

references:    
    (1) https://spikeinterface.github.io/blog/ground-truth-comparison-and-ensemble-sorting-of-a-synthetic-neuropixels-recording/
"""

import logging
import logging.config
from datetime import datetime

import pandas as pd
import yaml

from src.nodes.postpro import spikestats
from src.nodes.study import bias
from src.nodes.utils import get_config, write_metadata

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# Set experiment and run parameters
EXP = "supp/silico_reyes"
SIMULATION_DATE = "2023_01_13"
CREATE_RECORDING = False
LOAD_RAW_RECORDING = True
PREP_RECORDING = False
KILOSORT_2 = False  # we currently can't use it on simulation data due to matlab mkl version error
LOAD_SORTING_KS2 = False
KILOSORT_3 = True
LOAD_SORTING_KS3 = True

# set plot parameters
TITLE = "virtual reyes-128"

# get config
data_conf, param_conf = get_config(EXP, SIMULATION_DATE).values()


def run():
    """run sorting

    Returns:
        _type_: _description_
    """

    # get recording
    trace_recording = bias.get_recording(
        data_conf,
        param_conf,
        CREATE_RECORDING,
        LOAD_RAW_RECORDING,
        PREP_RECORDING,
    )

    # get ground truth
    GtSorting = bias.get_ground_truth(data_conf, param_conf)

    # get ground truth firing rates
    true_firing_rates = spikestats.get_firing_rates(
        GtSorting, trace_recording
    )
    true_firing_rates["sorter"] = "ground_truth"

    # record firing rates
    firing_rates = [true_firing_rates]

    # get firing rate from Kilosort2
    if KILOSORT_2:
        sorting_KS2 = bias.sort_with_KS2(
            trace_recording,
            data_conf,
            LOAD_SORTING_KS2,
        )
        k2_sorted_firing_rates = spikestats.get_firing_rates(
            sorting_KS2, trace_recording
        )
        k2_sorted_firing_rates["sorter"] = "kilosort2"

        # update firing rates
        firing_rates.append(k2_sorted_firing_rates)

    # get firing rate from Kilosort3
    if KILOSORT_3:
        sorting_KS3 = bias.sort_with_KS3(
            trace_recording, data_conf, LOAD_SORTING_KS3
        )
        k3_sorted_firing_rates = spikestats.get_firing_rates(
            sorting_KS3, trace_recording
        )
        k3_sorted_firing_rates["sorter"] = "kilosort3"

        # update firing rates
        firing_rates.append(k3_sorted_firing_rates)

    # create firing rate comparison dataframe
    df_to_plot = pd.concat(firing_rates, axis=0)

    # plot firing rates
    fig = bias.plot_firing_rates(TITLE, data_conf, df_to_plot)

    # write metadata
    metadata = {
        "fig_path": data_conf["figures"]["silico"]["population_bias_silico"],
        "creation_date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "experiment": SIMULATION_DATE,
        "run_date": SIMULATION_DATE,
        "data_conf": data_conf,
        "param_conf": param_conf,
    }
    write_metadata(
        metadata, data_conf["figures"]["silico"]["population_bias_silico"]
    )
    return fig


if __name__ == "__main__":
    fig = run()
