import numpy as np
import pandas as pd

from src.nodes.io import sorting
from src.nodes.truth.silico import ground_truth


# set detection functions
def get_spike_hits(
    true_ttps: np.array,
    sorted_ttps: np.array,
    sorted_ttps_units: np.array,
    match_windw: int,
):
    """match a unit's true spikes to all sorted spikes

    Args:
        true_timestamps (np.array): _description_
        sorted_timestamps (np.array): _description_

    Returns:
        _type_: _description_
    """
    hits = dict()

    for _, ttp in enumerate(true_ttps):
        # calculate its distance to all sorted timestamps
        distance = sorted_ttps - ttp

        # get the timestamp within match_windw (expect also more than one is possible)
        hits[ttp] = list(sorted_ttps[np.absolute(distance) <= match_windw])
    return {
        "sorted_ttps_hits": hits,
        "all_sorted_ttps": sorted_ttps,
        "unit_labels_for_sorted_ttps": sorted_ttps_units,
    }


def get_matching_sorted_unit(data: dict):
    """get the sorted unit label associated with the sorted timestamps
    that match the timestamps of a single true unit

    Args:
        data (dict): _description_

    Returns:
        _type_: _description_
    """

    # get hit data
    sorted_ttps_hits = data["sorted_ttps_hits"]
    true_ttps = list(sorted_ttps_hits.keys())
    unit_labels_for_sorted_ttps = data["unit_labels_for_sorted_ttps"]
    all_sorted_ttps = data["all_sorted_ttps"]

    # collect the sorted units associated with the sorted timestamps
    # that match the timestamps of the true unit
    out_matched_sorted = dict()
    for true_ttp_i in true_ttps:
        # get the sorted timestamps that match this true timestamp
        matched_ttps = sorted_ttps_hits[true_ttp_i]
        matched_units = []
        for sorted_ttp in matched_ttps:
            matched_units += (
                unit_labels_for_sorted_ttps[all_sorted_ttps == sorted_ttp]
                .astype(int)
                .tolist()
            )
        out_matched_sorted[true_ttp_i] = matched_units
    return {"sorted_unit_hits": out_matched_sorted}


def match_a_true_unit_spikes_to_all_sorted_spikes(
    true_unit_id: int, Truth, Sorting, match_wind: int
):
    """match a true unit spikes to all sorted spikes

    Args:
        true_unit_id (int): _description_
        Truth (_type_): _description_
        Sorting (_type_): _description_
        match_wind (int): _description_

    Returns:
        _type_: _description_
    """
    # get cell's true spike timestamps (timepoints)
    true_ttp = Truth.get_unit_spike_train(unit_id=true_unit_id)

    # get all sorted timestamps and their units
    out = Sorting.get_all_spike_trains()[0]
    all_sorted_ttps = out[0]
    sorted_unit_labels_for_ttps = out[1]

    # detect hits between true and sorted unit timestamps
    out = get_spike_hits(
        true_ttp, all_sorted_ttps, sorted_unit_labels_for_ttps, match_wind
    )

    # label the hits with their sorted units
    out["sorted_unit_hits"] = get_matching_sorted_unit(out)["sorted_unit_hits"]
    return out


def get_true_unit_spikes_detection_status(
    true_unit_id: int,
    data_conf: dict,
    match_wind_ms: float = 0.4,
    sampling_freq: int = 10000,
):
    """get the detection status of a true unit's spikes

    Args:
        true_unit_id (int): _description_
        match_wind_ms (float, optional): _description_. Defaults to 0.4.
        sampling_freq (int, optional): _description_. Defaults to 10000.

    Returns:
        dict: _description_
    """

    # calculate the MATCH_WIND_MS (0.4 ms in SpikeInterface) matching window in timepoints
    match_wind = int(match_wind_ms * sampling_freq / 1000)

    # load precomputed ground truth extractor
    Truth = ground_truth.load(data_conf)

    # load precomputed Sorting extractor
    Sorting = sorting.load(data_conf)

    # match a true unit spikes to all sorted spikes
    out = match_a_true_unit_spikes_to_all_sorted_spikes(
        true_unit_id=true_unit_id,
        Truth=Truth,
        Sorting=Sorting,
        match_wind=match_wind,
    )

    # record each instance of detections (columns)
    # produced by sorting for each true spike event (index)
    detection_instances = cast_hits_dict_as_dataframe(
        out["sorted_ttps_hits"]
    ).T

    # label true unit events that were detected or missed
    # (Boolean)
    detected_df = detection_instances.notnull()

    # sum over detection instances (df's column))
    # to get whether there was at least a detection
    detection_status = detected_df.sum(axis=1) == True

    detection_status.columns = ["detected"]
    detection_status.index.name = "events"
    return {
        "detection_status": detection_status,
        "detection_instances": detection_instances,
    }


# set performance metrics functions
def cast_hits_dict_as_dataframe(hits_dict: dict):
    """Convert dictionary to dataframe

    Args:
        hits_dict (dict): _description_

    Returns:
        _type_: _description_
    """
    return pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in hits_dict.items()])
    )


def test_get_spike_hits():
    """unit-test "match_a_unit_true_spikes_to_all_sorted_spikes" function"""

    # create test dataset
    UNIT = 0
    TEST_TTP = 1
    test_true_ttp = np.array([1, 100, 200, 300])
    test_sorted_spike_trains = np.array([2, 400, 500])
    test_sorted_ttps_units = np.array([0, 0, 1])

    # detect hits
    out = get_spike_hits(
        test_true_ttp,
        test_sorted_spike_trains[UNIT],
        test_sorted_ttps_units,
        match_windw=8,
    )
    assert out["sorted_unit_hits"][TEST_TTP][0] == 2, "wrong output"
