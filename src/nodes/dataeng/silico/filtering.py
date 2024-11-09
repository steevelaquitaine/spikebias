import logging
import logging.config
from copy import copy

import bluepy as bp
import numpy as np
import yaml

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def filter_microcircuit_cells(simulation):
    """Get the microcircuit cells positions
    Setting variable group = {"$target": "hex0"} permits to select the
    target group of 30,190 microcircuit cells.

    Args:
        simulation (dict): simulation parameters and circuit info
        returned by src.nodes.io.silico.load_campaign_params

    Returns:
        dict: _description_
    """
    soma_location = simulation["circuit"].cells.get(
        {"$target": "hex0"}, properties=[bp.Cell.X, bp.Cell.Y, bp.Cell.Z]
    )
    gids = simulation["circuit"].cells.ids({"$target": "hex0"})
    return {"gid": gids, "soma_location": soma_location}


def get_hex_01_cells(simulation):
    """Get the coordinates of 211K cells from a seven-column circuit
    (See 3)

    Args:
        simulation (dict): simulation parameters and circuit info
        returned by src.nodes.io.silico.load_campaign_params

    Returns:
        dict: _description_
    """
    soma_location = simulation["circuit"].cells.get(
        {"$target": "hex_O1"}, properties=[bp.Cell.X, bp.Cell.Y, bp.Cell.Z]
    )
    gids = simulation["circuit"].cells.ids({"$target": "hex0"})
    return {"gid": gids, "soma_location": soma_location}


def get_cell_spiking_above_thresh(gt_sorting0, min_spike: int):
    """Get waveform extractor dictionary for a filtered set of cells
    above a spike count threshold

    Args:
        rec0 (_type_): _description_
        gt_sorting0 (_type_): _description_
        min_spike (int): _description_

    Returns:
        dict: waveform extractor
    """
    # copy object to keep source object unchanged
    gt_sorting0 = copy(gt_sorting0)

    # filter
    gt_dict = gt_sorting0.get_total_num_spikes()
    rare_gids = list(
        dict((k, v) for k, v in gt_dict.items() if v <= min_spike).keys()
    )
    gt_sorting0 = gt_sorting0.remove_units(rare_gids)
    return gt_sorting0


def get_cell_id_spiking_above_thresh(gt_sorting0, min_spike: int):
    """Get global identifier of cells spiking above thresholds

    Args:
        rec0 (_type_): _description_
        gt_sorting0 (_type_): _description_
        min_spike (int): _description_

    Returns:
        dict: waveform extractor
    """
    # copy object to keep source object unchanged
    gt_sorting0 = copy(gt_sorting0)

    # filter
    gt_dict = gt_sorting0.get_total_num_spikes()
    return list(
        dict((k, v) for k, v in gt_dict.items() if v >= min_spike).keys()
    )


def create_study_object(rec0, gt_sorting0):
    """create study object

    Args:
        rec0 (_type_): _description_
        gt_sorting0 (_type_): _description_
    """
    return {
        "rec0": (rec0, gt_sorting0),
    }


def get_most_active_cell(SortingObject, cell_id: np.array):
    """Get the global id of the most active cell among the cell_gids

    Args:
        SortingObject (_type_): Spikeinterface Sorting object
        cell_gid (np.array): _description_

    Returns:
        _type_: global id of the most active cell
    """
    id_spike_dict = SortingObject.get_total_num_spikes()
    pyramidal_id_spike_dict = {key: id_spike_dict[key] for key in cell_id}
    return max(pyramidal_id_spike_dict, key=pyramidal_id_spike_dict.get)
