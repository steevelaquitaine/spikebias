"""nodes that analyses sorted units' metadata

Returns:
    _type_: _description_
"""
import pandas as pd
import numpy as np
from src.nodes.metrics.quality import get_scores
from src.nodes import utils


def count_unit_type(
    all_type,
    unique_type,
):

    # count instances
    count = []
    type_nm = []
    for ix in range(len(unique_type)):
        type_nm.append(tuple(unique_type.iloc[ix].values))
        count.append(
            sum(all_type.T.apply(lambda row: all(row.values == unique_type.iloc[ix])))
        )

    df2 = pd.DataFrame()
    df2["type"] = type_nm
    df2["count"] = count
    return df2


def count_unit_type_by_quality(df, sorters, unique_type_feat, unique_type):
    """count unit type distribution by quality
    for all spike sorters

    Args:
        df (_type_): _description_
        sorters (_type_): _description_

    Returns:
        _type_: _description_
    """

    count_g = dict()
    count_oo = dict()

    # loop over spike sorters
    for ix in range(len(df)):

        # type distri good
        count_g[sorters[ix]] = count_unit_type(
            df[ix][df[ix].quality == "mixed: good + overmerger + oversplitter"][
                unique_type_feat
            ],
            unique_type,
        )
        # count unit types by spike sorter
        count_oo[sorters[ix]] = count_unit_type(
            df[ix][df[ix].quality == "mixed: overmerger + oversplitter"][
                unique_type_feat
            ],
            unique_type,
        )
    return count_g, count_oo


def get_ground_truth_match(SortingTrue, Sorting, DT):

    # find ground truth match
    scores = get_scores(SortingTrue, Sorting, DT)
    score = np.max(scores.values, axis=0)
    loc_max = np.argmax(scores.values, axis=0)
    gt = scores.index[loc_max]
    return loc_max, score, scores, gt


def get_sorted_unit_features(Sorting, SortingTrue, DT):
    """_summary_

    Args:
        Sorting (_type_): _description_
        SortingTrue (_type_): _description_
        DT (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = pd.DataFrame()

    # find ground truth match
    loc_max, score, scores, gt = get_ground_truth_match(SortingTrue, Sorting, DT)
    
    # get features
    synapse = SortingTrue.get_property("synapse_class")[loc_max]
    etype = SortingTrue.get_property("etype")[loc_max]
    mtype = SortingTrue.get_property("mtype")[loc_max]

    # build dataset
    df["synapse"] = synapse
    df["etype"] = etype
    df["mtype"] = mtype
    df["score"] = score
    df.index = Sorting.unit_ids
    df["gt"] = gt

    # unit-test
    all(SortingTrue.unit_ids == scores.index), "index must match sorted unit ids"
    return df


def get_unit_features(Sorting):
    """get the selected features of ground truth units

    Args:
        Sorting (_type_): _description_
        DT (_type_): _description_

    Returns:
        df: index: sorted units
    """
    df = pd.DataFrame()

    # build dataset
    layer = utils.standardize_layers(Sorting.get_property("layer"))
    df["layer"] = layer
    df["synapse"] = Sorting.get_property("synapse_class")
    df["etype"] = Sorting.get_property("etype")
    df.index = Sorting.unit_ids
    return df


def get_feature_data_for(sorter: str, SortingK4, SortingTrue, quality_path, DT):

    # features
    feat_df = get_sorted_unit_features(SortingK4, SortingTrue, DT)

    # quality and biases
    q_df = pd.read_csv(quality_path)
    q_df = q_df[q_df.sorter == sorter]
    q_df.index = q_df.sorted
    q_df = q_df.drop(columns=["sorted", "sorter", "experiment"])
    df = q_df.join(feat_df)
    df["sorter"] = sorter

    # unit-test
    assert len(np.unique(q_df.index)) == len(q_df.index), "unit ids should be unique"
    return df


def set_sorted_unit_features(Sorting, SortingTrue, dt: float):
    """_summary_

    Args:
        Sorting (_type_): _description_
        SortingTrue (_type_): _description_
        dt (_type_): _description_

    Returns:
        _type_: _description_
    """

    # get unit features
    feature = get_sorted_unit_features(Sorting, SortingTrue, dt)

    # record as property
    Sorting.set_property("synapse_class", feature["synapse"].values)
    Sorting.set_property("etype", feature["etype"].values)
    Sorting.set_property("mtype", feature["mtype"].values)
    Sorting.set_property("score", feature["score"].values)
    Sorting.set_property("gt", feature["gt"].values)
    return Sorting

# tests -------------------

def test_get_ground_truth_match(SortingTrue, Sorting, DT:float):
    """unit-testing get_ground_truth_match function

    Args:
        SortingTrue (_type_): _description_
        Sorting (_type_): _description_
        DT (_type_): delta_time
    """

    # run function
    loc_max, score, _, _ = get_ground_truth_match(SortingTrue, Sorting, DT)
    
    # sorted unit 0
    sorted_ix = 0
    assert score.iloc[:, sorted_ix].values.max() == score.iloc[loc_max[sorted_ix], sorted_ix], "max should match"
    # sorted unit 1
    sorted_ix = 1
    assert score.iloc[:, sorted_ix].values.max() == score.iloc[loc_max[sorted_ix], sorted_ix], "max should match"
    # sorted unit 3
    sorted_ix = 3
    assert score.iloc[:, sorted_ix].values.max() == score.iloc[loc_max[sorted_ix], sorted_ix], "max should match"
    

def test_get_features_data_for(SortingTrue, dataset):

    # unit-test features
    # layer
    sorted_ix = 0
    assert (
        dataset.iloc[sorted_ix]["layer"][1:]
        == SortingTrue.get_property("layer")[
            SortingTrue.unit_ids == dataset.iloc[sorted_ix]["gt"]
        ]
    ), "layer should match"

    # synapse
    assert (
        dataset.iloc[sorted_ix]["synapse"]
        == np.array(SortingTrue.get_property("synapse_class"))[
            np.where(SortingTrue.unit_ids == dataset.iloc[sorted_ix]["gt"])[0][0]
        ]
    ), "synapse should match"