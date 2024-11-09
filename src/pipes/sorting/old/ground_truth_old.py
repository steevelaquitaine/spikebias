import os
import shutil
import sys

import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import yaml

from src.pipes.silico.dataeng.loading import (
    load_params_for_one_simulation_piece,
)


def load_sim_piece(i: int, simresults):
    """load a simulation chunk
    TODO: move to loading (rename to io) dataeng module

    Args:
        i (int): _description_
        simresults (_type_): _description_

    Returns:
        _type_: _description_
    """
    spike_pieces = pd.read_pickle(simresults + "spikes" + str(i) + ".pkl")
    return spike_pieces


def stack_simulation_pieces(
    spike_write_path: str, simulation: dict, single_piece_sim: bool
):
    """stack a simulation chunks
    TODO: check redundance with dataeng "stacking.py" module

    Args:
        spike_write_path (str): _description_
        simulation (dict): _description_

    Returns:
        _type_: _description_
    """
    spike_list = []
    for piece_i in range(simulation["n_trace_spike_files"]):
        spike_pieces = load_sim_piece(
            piece_i, simulation["paths"]["simulated_traces_and_spikes"]
        )
        if single_piece_sim == True:
            if piece_i == 0:
                spike_list.append(spike_pieces)
        else:
            spike_list.append(spike_pieces)
    stacked_spike = pd.concat(spike_list)

    # write spike trains
    stacked_spike.to_pickle(spike_write_path)

    # create sorting opject
    # note the sampling rate of the recording and not of the spikes
    times = np.int_(
        np.array(stacked_spike.index) * simulation["lfp_sampling_freq"] / 1000
    )
    labels = stacked_spike.values
    return times, labels


def create_sorting_object(
    sorting_object_write_path: str, sampling_freq, times, labels
):
    """_summary_
    TODO: create a format_for_sorting.py module in dataeng and move there

    Args:
        sorting_object_write_path (str): _description_
        sampling_freq (_type_): _description_
        times (_type_): _description_
        labels (_type_): _description_

    Returns:
        _type_: _description_
    """

    # format into a spikeinterface sorting object
    sorting_object = se.NumpySorting.from_times_labels(
        [times], [labels], sampling_freq
    )

    # clear directory
    shutil.rmtree(sorting_object_write_path, ignore_errors=True)

    # write simulated spikes
    sorting_object = sorting_object.save(folder=sorting_object_write_path)
    return sorting_object


def get_spike(simulation: dict, dataset_conf: dict, param_conf: dict):
    """_summary_
    TODO: check redundance with dataeng "stacking.py" module

    Args:
        simulation (dict): _description_
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """
    # set ground truth spike write path
    spike_write_path = dataset_conf["dataeng"]["output"]["spike_file"]
    sorting_object_write_path = dataset_conf["output"][
        "ground_truth_sorting_object"
    ]["write_path"]

    # get simulation parameters
    SINGLE_PIECE_SIM = param_conf["circuit"]["single_piece"]

    # read spikes for all pieces
    times, labels = stack_simulation_pieces(
        spike_write_path, simulation, SINGLE_PIECE_SIM
    )

    # format into a spikeinterface sorting object
    sorting_object = create_sorting_object(
        sorting_object_write_path,
        simulation["lfp_sampling_freq"],
        times,
        labels,
    )

    # store parameters
    params = {"piece_duration": simulation["report"].t_end / 1000}
    return sorting_object, params


def run(simulation: dict, dataset_conf: dict, param_conf: dict):
    """get ground truth spikeinterface sorting object

    Args:
        simulation (dict): _description_

    Returns:
        _type_: _description_
    """

    sorting_object, params = get_spike(simulation, dataset_conf, param_conf)
    return {"ground_truth_sorting_object": sorting_object, "params": params}


if __name__ == "__main__":

    # get simulation date from command call
    conf_date = sys.argv[1]

    # read the pipeline's configuration
    with open(
        f"conf/silico/{conf_date}/dataset.yml", encoding="utf-8"
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)

    with open(
        f"conf/silico/{conf_date}/parameters.yml", encoding="utf-8"
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)

    # load circuit simulation
    # TODO: replace with a function that loads a campaign of many simulations
    simulation_params = load_params_for_one_simulation_piece(
        dataset_conf, param_conf
    )

    # get ground truth spikes
    output = run(simulation_params, dataset_conf, param_conf)
