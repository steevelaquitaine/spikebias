"""toolbox of general utility functions

Returns:
    _type_: _description_
"""
import json
import logging
import logging.config
import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def get_config(exp: str, simulation_date: str):
    """Choose an available experiment pipeline configuration

    Args:
        exp (str): the choosen experiment to run
        - "supp/silico_reyes": in-silico simulated lfp recordings with 8 shanks, 128 contacts reyes probe
        - "buccino_2020": neuropixel probe on buccino 2020's ground truth dataset
        - "silico_neuropixels": in-silico simulated lfp recordings with neuropixel probes
        simulation_date (str): _description_
        ....

    Returns:
        dict: dataset paths and parameter configurations
    """
    logger.info("Reading experiment config.")
    if exp == "supp/silico_reyes":
        data_conf, param_conf = get_config_silico_reyes(
            simulation_date
        ).values()
    elif exp == "buccino_2020":
        data_conf, param_conf = get_config_buccino_2020(
            simulation_date
        ).values()
    elif exp == "silico_neuropixels":    
        data_conf, param_conf = get_config_silico_neuropixels(
            simulation_date
        ).values()
    elif exp == "supp/hybrid_janelia":
        data_conf, param_conf = get_config_hybrid_janelia(
            simulation_date
        ).values()
    elif exp == "synth_monotrode":
        data_conf, param_conf = get_config_synth_monotrode(
            simulation_date
        ).values()
    elif exp == "supp/allen_neuropixels":
        data_conf, param_conf = get_config_allen_npx(
            simulation_date
        ).values()
    elif exp == "supp/buzsaki":
        data_conf, param_conf = get_config_buzsaki(
            simulation_date
        ).values()
    elif exp == "vivo_reyes":
        data_conf, param_conf = get_config_vivo_reyes(
            simulation_date
        ).values()
    elif exp == "vivo_horvath":
        data_conf, param_conf = get_config_vivo_horvath(
            simulation_date
        ).values()
    elif exp == "dense_spont":
        data_conf, param_conf = get_config_silico_horvath(
            simulation_date
        ).values()
    elif exp == "dense_spont_from_nwb":
        data_conf, param_conf = get_config_dense_spont_from_nwb(
            simulation_date
        ).values()
    elif exp == "dense_spont_on_dandihub":
        data_conf, param_conf = get_config_dense_spont_on_dandihub(
            simulation_date
        ).values()        
    elif exp == "silico_neuropixels_from_nwb":
        data_conf, param_conf = get_config_silico_neuropixels_from_nwb(
            simulation_date
        ).values()
    elif exp == "vivo_marques":
        data_conf, param_conf = get_config_vivo_marques(
            simulation_date
        ).values()
    elif exp == "others/spikewarp":
        data_conf, param_conf = get_config_spikewarp(
            simulation_date
        ).values()
    else:
        raise NotImplementedError("""This experiment is not implemented, 
                                  Add it to get_config() in
                                  src/nodes/utils.py""")
    logger.info("Reading experiment config. - done")
    return {"dataset_conf": data_conf, "param_conf": param_conf}


def get_config_vivo_reyes(run_date: str):
    """Get pipeline's configuration for in-vivo Reyes dataset

    Args:
        run_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/vivo_reyes/{run_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "vivo_reyes"
        dataset_conf["date"] = run_date
    with open(
        f"conf/vivo_reyes/{run_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_vivo_horvath(run_date: str):
    """Get pipeline's configuration for in-vivo Horvath dataset
    with neuropixels

    Args:
        run_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/vivo_horvath/{run_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "vivo_horvath"
        dataset_conf["date"] = run_date
    with open(
        f"conf/vivo_horvath/{run_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_silico_horvath(run_date: str):
    """Get pipeline's configuration for in-silico Horvath dataset

    Args:
        run_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/dense_spont/{run_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "dense_spont"
        dataset_conf["date"] = run_date
    with open(
        f"conf/dense_spont/{run_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_dense_spont_from_nwb(simulation_date: str):
    """Get pipeline's configuration for James spike time project

    Args:
        simulation_date (str): _description_
        - probe_1, probe_2, probe_3

    Returns:
        dict: dataset paths and parameter configurations
    """
    logger.info(f"conf/dense_spont/{simulation_date} config")
    
    with open(
        f"conf/dense_spont/{simulation_date}/dataset_nwb.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "dense_spont_from_nwb"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/dense_spont/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}

def get_config_dense_spont_on_dandihub(simulation_date: str):
    """Get pipeline's configuration for James spike time project

    Args:
        simulation_date (str): _description_
        - probe_1, probe_2, probe_3

    Returns:
        dict: dataset paths and parameter configurations
    """
    logger.info(f"conf/dense_spont/{simulation_date} config")
    
    with open(
        f"conf/dense_spont/{simulation_date}/dataset_on_dandihub.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "dense_spont_on_dandihub"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/dense_spont/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_silico_neuropixels_from_nwb(simulation_date: str):
    """Get pipeline's configuration for James spike time project

    Args:
        simulation_date (str): _description_
        - probe_1, probe_2, probe_3

    Returns:
        dict: dataset paths and parameter configurations
    """
    logger.info(f"conf/silico_neuropixels/{simulation_date} config")
    
    with open(
        f"conf/silico_neuropixels/{simulation_date}/dataset_nwb.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "silico_neuropixels_from_nwb"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/silico_neuropixels/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_vivo_marques(run_date: str):
    """Get pipeline's configuration for in-vivo marques dataset
    with neuropixels

    Args:
        run_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/vivo_marques/{run_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "vivo_marques"
        dataset_conf["date"] = run_date
    with open(
        f"conf/vivo_marques/{run_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_silico_reyes(simulation_date: str):
    """Get pipeline's configuration for silico Reyes probe experiment

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/supp/silico_reyes/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "supp/silico_reyes"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/supp/silico_reyes/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_buccino_2020(simulation_date: str):
    """Get pipeline's configuration for buccino 2020 experiment

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/buccino_2020/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "buccino_2020"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/buccino_2020/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_silico_neuropixels(simulation_date: str):
    """Get pipeline's configuration for silico neuropixels probe experiment

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """
    with open(
        f"conf/silico_neuropixels/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "silico_neuropixels"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/silico_neuropixels/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_hybrid_janelia(simulation_date: str):
    """Get pipeline's configuration for hybrid janelia experiment

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/supp/hybrid_janelia/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "supp/hybrid_janelia"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/supp/hybrid_janelia/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_allen_npx(simulation_date: str):
    """Get pipeline's configuration for allen neuropixels dataset

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/supp/allen_neuropixels/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "supp/allen_neuropixels"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/supp/allen_neuropixels/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_buzsaki(simulation_date: str):
    """Get pipeline's configuration for allen neuropixels dataset

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/supp/buzsaki/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "supp/buzsaki"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/supp/buzsaki/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_synth_monotrode(simulation_date: str):
    """Get pipeline's configuration for synth monotrode experiment

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/synth_monotrode/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "synth_monotrode"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/synth_monotrode/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_spikewarp(simulation_date: str):
    """Get pipeline's configuration for James spike time project

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """
    logger.info(f"conf/others/spikewarp/{simulation_date} config")
    with open(
        f"conf/others/spikewarp/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "others/spikewarp"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/others/spikewarp/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def create_if_not_exists(my_path: str):
    """Create a path if it does not exist

    Args:
        my_path (str): _description_
    """
    if not os.path.isdir(my_path):
        os.makedirs(my_path)
        logger.info("The following path has been created %s", my_path)


def write_metadata(metadata: dict, fig_path: str):
    """write figure metadata

    Args:
        metadata (dict): contains the simulation date, the content of its dataset.yml
        and parameters.yml
        fig_path (str): _description_
    """
    # create path if it does not exist
    parent_path = os.path.dirname(fig_path)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
        logger.info("The figure's path did not exist and was created.")

    # write figure
    json_object = json.dumps(metadata, indent=4)
    with open(fig_path + ".json", "w", encoding="utf-8") as file:
        file.write(json_object)


def from_dict_to_df(my_dict: dict, columns: list):
    """Convert dictionary to pandas dataframe

    Args:
        my_dict (dict): dictionary to convert
        columns (list): column names

    Returns:
        pd.DataFrame: _description_
    """
    return pd.DataFrame(list(my_dict.items()), columns=columns)


def euclidean_distance(coord_1: np.array, coord_2: np.array):
    return np.sqrt(np.sum((coord_1 - coord_2)**2))


def write_npy(anr, file_write_path: str):
    """write npy file

    Args:
        anr (_type_): _description_
        file_write_path (str): file path
    """
    parent_path = os.path.dirname(file_write_path)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(file_write_path, anr, allow_pickle=True)

    
def sem(data):
    return np.std(data) / np.sqrt(len(data))


def conf_interv95(data):
    return 1.96 * sem(data)


def demean(trace):
    """subtract the mean of each column

    Args:
        trace (np.array): sample x sites

    Returns:
        _type_: _description_
    """
    site_means = np.mean(trace, axis=0)
    means = site_means.reshape((1, len(site_means)))
    return trace - means


def savefig(fig_path: str):

    # set figure parameters
    savefig_cfg = {
        "transparent": True,
        "dpi": 400
        }
    
    # create parent path if
    # not exists
    create_if_not_exists(
        os.path.dirname(
            os.path.abspath(
                fig_path
                )
            )
        )
    
    # save    
    plt.savefig(fig_path, bbox_inches="tight", **savefig_cfg)


def savefig_with_params(fig_path: str, dpi, transparent):

    # set figure parameters
    savefig_cfg = {
        "transparent": transparent,
        "dpi": dpi
        }
    
    # create parent path if
    # not exists
    create_if_not_exists(
        os.path.dirname(
            os.path.abspath(
                fig_path
                )
            )
        )
    
    # save    
    plt.savefig(fig_path, bbox_inches="tight", **savefig_cfg)


def standardize_layers(layers: list):
    layers = ["L1" if w == "1" else w for w in layers]
    layers = ["L2/3" if w == "3" or w == "2" else w for w in layers]
    layers = ["L2/3" if w == "L2_3" else w for w in layers]
    layers = ["L2/3" if w == "L2" else w for w in layers]
    layers = ["L2/3" if w == "L3" else w for w in layers]
    layers = ["L4" if w == "4" else w for w in layers]
    layers = ["L5" if w == "5" else w for w in layers]
    layers = ["L6" if w == "6" else w for w in layers]
    layers = ["Out" if w == "WM" else w for w in layers]
    return layers


def standardize_gt_layers(layers: list):
    layers = [w.replace("1", "L1") for w in layers]
    layers = ["L2/3" if w == "3" or w == "2" else w for w in layers]
    layers = [w.replace("5", "L5") for w in layers]
    layers = [w.replace("6", "L6") for w in layers]
    layers = [w.replace("4", "L4") for w in layers]
    return layers


def get_kslabel(Sorting):
    """get sorted unit KSLabel property from SortingExtractor

    Args:
        Sorting (SortingExtractor): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    n_units = len(Sorting.unit_ids)
    if isinstance(Sorting.get_property("KSLabel"), type(None)):
        logger.info("No KSLabel property in SortingExtractor")
        return [None]*n_units
    elif len(Sorting.get_property("KSLabel"))==n_units:
        return Sorting.get_property("KSLabel").tolist()
    else:
        raise ValueError("KSLabel has wrong shape")


def get_amplitude(Sorting):
    """get sorted unit Amplitude property from SortingExtractor

    Args:
        Sorting (SortingExtractor): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    n_units = len(Sorting.unit_ids)
    if isinstance(Sorting.get_property("Amplitude"), type(None)):
        logger.info("No Amplitude property in SortingExtractor")
        return [None]*n_units
    elif len(Sorting.get_property("Amplitude"))==n_units:
        return Sorting.get_property("Amplitude").tolist()
    else:
        raise ValueError("Amplitude has wrong shape")