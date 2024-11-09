"""module that contains sorted unit quality-related metrics

Returns:
    _type_: _description_
"""
import numpy as np
import pandas as pd
from numba import njit, prange
from spikeinterface import comparison
import spikeinterface as si
from src.nodes import utils
from matplotlib import pyplot as plt
pd.options.mode.chained_assignment = None


@njit(parallel=True)
def _compute_chance_score(chance: np.ndarray, gt_ids: np.array,
                   sorted_ids: np.array, fr_g: np.array,
                   fr_s: np.array, duration: float,
                   delta_time: float):
    """calculate chance scores for pairs of ground truth and
    sorted units (parallelized with threading with Numba)

    Args:
        chance (np.ndarray): initialized array of np.nan
        - with shape len(sorted_ids) x len(gt_ids)
        gt_ids (np.array): ground truth unit ids
        sorted_ids (np.array): sorted unit ids
        fr_g (list): ground truth unit firing rates
        fr_s (list): sorted unit firing rates
        duration (float): recording duration
        delta_time (float): coincidence interval surrounding a ground truth spike 
        that defines a hit from a coinciding sorted unit spike

    Returns:
        np.array: chance scores for pairs of sorted and ground truth units
    """
    # loop over ground truth units
    for g_ix in prange(len(gt_ids)):
        
        # loop over sorted units
        for s_ix in prange(len(sorted_ids)):

            # time interval in ms
            interval_ms = 2 * delta_time

            # expected nb of coincidences
            n_sp = interval_ms * min(fr_g[g_ix], fr_s[s_ix]) / 1000

            # poisson for k=0 hits
            poisson = np.exp(-n_sp)

            # probability of hit by chance
            p_chance_hit = 1.0 - poisson

            # nb of spikes
            n_gt = fr_g[g_ix] * duration
            n_s = fr_s[s_ix] * duration

            # nb of hits, false positives, misses
            # - the smallers spike train min(n_gt, n_s) determines
            # the maximum possible number of hits
            n_h = p_chance_hit * min(n_gt, n_s)
            n_fp = n_s - n_h
            n_m = n_gt - n_h
            
            # calculate chance score for this pair
            chance[s_ix, g_ix] = n_h / (n_h + n_m + n_fp)
    return chance


def _get_all_matched_true_unit(s_id, scores):
    """get all true unit matchs

    Args:
        match (_type_): _description_
        s_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.array(scores.columns[scores.loc[s_id, :] > 0])


def _is_false_positive(s_id, gt_id, scores, chances):
    return all(scores.loc[s_id, gt_id].values <= chances.loc[s_id, gt_id].values)


def _is_false_positive_for_numba(s_ix, g_ixs, scores, chances):
    """_summary_

    Args:
        s_ix (int): _description_
        g_ixs (int): _description_
        scores (np.array): _description_
        chances (np.array): _description_

    Returns:
        _type_: _description_
    """
    return (scores[s_ix, g_ixs] <= chances[si, g_ixs]).all()


def _is_good(s_id, gt_id, scores, det: float):
    """is sorted unit good: it matches one ground truth with 80% and no
    other ground truth with

    Args:
        s_id (_type_): _description_
        det (float): _description_

    Returns:
        _type_: _description_
    """
    return any(scores.loc[s_id, gt_id].values >= det)


def _is_good_for_numba(s_ix, g_ixs, scores: np.array, det: float):
    """is sorted unit good: it matches one ground truth with 80% and no
    other ground truth with

    Args:
        s_ix (int): _description_
        g_ixs (int): _description_
        scores (np.array): _description_
        det (float): "good" class threshold

    Returns:
        _type_: _description_
    """
    return (scores[s_ix, g_ixs] >= 0.8).any()


def _is_overmerger(s_id, gt_id, scores, chances):
    """is an overmerger: the sorted unit matches at least two ground
    truths with strictly above chance scores"""
    return sum(scores.loc[s_id, gt_id].values > chances.loc[s_id, gt_id].values) > 1


def _is_overmerger_for_numba(s_ix, g_ixs, scores: np.array, chances: np.array):
    """is an overmerger: the sorted unit matches at least two ground
    truths with strictly above chance scores"""
    return (scores[s_ix, g_ixs] > chances[s_ix, g_ixs]).sum() > 1


def _make_1darray(gt_id):
    if isinstance(gt_id, np.ndarray):
        return gt_id
    else:
        return np.array([gt_id])
    
    
def _is_oversplitter(s_id, gt_id, scores, chance):
    """oversplitter (redundant): the sorted unit's matched
    ground truths are matched with other sorted units
    above chance"""
    gt_id = _make_1darray(gt_id)

    # drop target sorted unit
    # check if any detection > chance for each matched ground truth
    is_detected = []
    for _, g_i in enumerate(gt_id):
        other_score = scores.drop(index=s_id).loc[:, g_i].values
        other_chance = chance.drop(index=s_id).loc[:, g_i].values
        is_detected.append(any(other_score > other_chance))

    # any detection across all the matched ground truths
    return any(is_detected)


def _is_oversplitter_for_numba(s_ix, g_ixs, scores, chances):
    """oversplitter (redundant): the sorted unit's matched
    ground truths are matched with other sorted units
    above chance

    Args:
        s_ix (_type_): _description_
        g_ixs (_type_): _description_
        scores (_type_): _description_
        chances (_type_): _description_

    Returns:
        _type_: _description_
    """
    scores2 = np.delete(scores, s_ix, axis=0)
    chances2 = np.delete(chances, s_ix, axis=0)
    other_scores = scores[:, g_ixs]
    other_chances = chances[:, g_ixs]
    return (other_scores > other_chances).any()


def _set_df(df: pd.DataFrame, sorted: int, quality: str):
    """record sorted unit quality
    append new quality if found many"""
    df.loc[sorted, "sorted"] = int(sorted)
    qual = df.loc[sorted, "quality"]
    if isinstance(qual, str):
        df.loc[sorted, "quality"] += quality
    else:
        df.loc[sorted, "quality"] = quality
    return df


def get_score_single_unit(sorting_path: str, gtruth_path: str, delta_time: float):
    """get single unit scores (if exists "KSLabel" metadata)
    
    Args:
    
    sorting_path (str): SortingExtractor path
    gtruth_path (str): Ground truth path

    Returns:
        _type_: _description_
    """
    
    # load SortingExtractors
    Sorting_ns = si.load_extractor(sorting_path)
    SortingTrue_ns = si.load_extractor(gtruth_path)
    
    # get scores (N sorted units rows x N true units columns)
    scores_ns = get_scores(SortingTrue_ns, Sorting_ns, delta_time)
    scores_ns = scores_ns.T

    # curate (get single-unit only if exists)
    if "KSLabel" in Sorting_ns.get_property_keys():
        kslabel = Sorting_ns.get_property("KSLabel")
        loc_single_u = Sorting_ns.unit_ids[kslabel=="good"]
        scores_ns = scores_ns.loc[loc_single_u, :]
    return (scores_ns, Sorting_ns, SortingTrue_ns)
    
    
def get_scores(
    SortingTrue,
    Sorting,
    delta_time: float,
):
    """get agreement scores between 
    ground truth and sorted units

    Args:
        SortingTrue (_type_): _description_
        Sorting (_type_): _description_
        delta_time (float): _description_

    Returns:
        pd.DataFrame: agreemen scores
    """
    comp = comparison.compare_sorter_to_ground_truth(
        SortingTrue,
        Sorting,
        exhaustive_gt=True,
        delta_time=delta_time,
    )
    return comp.agreement_scores


def precompute_chance_score(
    recording_path, scores, Sorting, SortingTrue, duration, delta_time
):
    """parallelized computing of chance scores for pairs of ground truths
    and sorted units

    Args:
        recording_path (_type_): _description_
        scores (_type_): _description_
        Sorting (_type_): _description_
        SortingTrue (_type_): _description_
        duration (_type_): specified duration
        delta_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    Rec = si.load_extractor(recording_path)
    duration = min(Rec.get_total_duration(), duration)
    return get_parallel_chance_score(
        scores, Sorting, SortingTrue, duration, delta_time
    )
    
    
def get_parallel_chance_score(scores: pd.DataFrame, Sorting, SortingTrue,
                         duration: float, delta_time: float):
    """parallelized computing of chance scores for pairs of ground truths
    and sorted units

    Args:
        scores (pd.DataFrame): agreement scores
        Sorting (SortingExtractor): sorted
        SortingTrue (SortingExtractor): ground truth
        duration (float): actual recording duration
        delta_time (float): coincidence interval surrounding a ground truth spike 
        that defines a hit from a coinciding sorted unit spike

    Returns:
        pd.DataFrame: chance scores per sorted and ground truth pairs
    """

    # initialize chance array
    chance_array = np.ones((scores.shape[0], scores.shape[1])) * np.nan

    # get firing rates
    # sorted units
    fr_s = []
    for s_id in scores.index:
        fr_s.append(Sorting.count_num_spikes_per_unit()[s_id] / duration)

    # ground truth units
    fr_g = []
    for g_id in scores.columns:
        fr_g.append(SortingTrue.count_num_spikes_per_unit()[g_id] / duration)

    # compute chance scores with Numba (parallelized on multiple cores)
    chance_array = _compute_chance_score(
        chance_array,
        np.array(scores.columns),
        np.array(scores.index),
        fr_g,
        fr_s,
        duration,
        delta_time
    )
    return pd.DataFrame(chance_array, index=scores.index, columns=scores.columns)


def qualify_sorted_units_slow(scores, chance_df, thr_good):
    """qualify sorted units as false positives, good,
    overmerger, oversplitter and mixtures of these
    
    This function is time consuming and can be parallelized.

    Args:
        scores (pd.DataFrame): agreement scores
        - index: sorted unit ids 
        chance_df (_type_): theoretically derived chance scores
        - per pairs of ground truth and sorted units
        thr_good (float): threshold for good unit

    Returns:
        pd.DataFrame: _description_
        
    Note:
        Their colors:
        cl = {"good": [0.7, 0.1, 0.1], # strong red
            "oversplitter": [0.6, 0.9, 0.6], # blue
            "overmerger": [0, 0.7, 1], # green
            "mixed: good + overmerger": np.array([[0.7, 0.1, 0.1], [0, 0.7, 1]]).mean(axis=0),
            "mixed: good + oversplitter": np.array([[0.7, 0.1, 0.1], [0.6, 0.9, 0.6]]).mean(axis=0),
            "mixed: overmerger + oversplitter": np.array([[0.6, 0.9, 0.6], [0, 0.7, 1]]).mean(axis=0),
            "mixed: good + overmerger + oversplitter": np.array([[0.7, 0.1, 0.1], [0, 0.7, 1],[0.6, 0.9, 0.6]]).mean(axis=0),
            "false positive": [0, 0, 0] # black
        }        
    """
    # initialize dataframe to record quality
    df = pd.DataFrame({"sorted": [], "quality": np.nan})

    # qualify each sorted single-unit
    for _, s_id in enumerate(scores.index):

        # get its matched ground truth units
        gt_id = _get_all_matched_true_unit(s_id, scores)

        if _is_false_positive(s_id, gt_id, scores, chance_df):
            df = _set_df(df, s_id, "+ false_positive ")

        else:
            if _is_good(s_id, gt_id, scores, thr_good):
                df = _set_df(df, s_id, "+ good ")

            if _is_overmerger(s_id, gt_id, scores, chance_df):
                df = _set_df(df, s_id, "+ overmerger ")

            if _is_oversplitter(s_id, gt_id, scores, chance_df):
                df = _set_df(df, s_id, "+ oversplitter ")

    # standardize the labels
    # independently of their ordering 
    # in each entry
    df = standardize_sorting_quality(df)
                    
    # unit-test
    assert len(df) == len(scores.index), "some sorted units were not qualified"
    return df


def qualify_sorted_units(scores, s_ids, chances, thr_good):
    """qualify sorted units as false positives, good,
    overmerger, oversplitter and mixtures of these
    
    Writing entirely with numpy array accelerated 100X,
    compared to pandas dataframe.
    
    Args:
        scores (pd.DataFrame): agreement scores
        - index: sorted unit ids 
        chances (_type_): theoretically derived chance scores
        - per pairs of ground truth and sorted units
        thr_good (float): threshold for good unit

    Returns:
        pd.DataFrame: _description_
        
    Note:
        Their colors:
        cl = {"good": [0.7, 0.1, 0.1], # strong red
            "oversplitter": [0.6, 0.9, 0.6], # blue
            "overmerger": [0, 0.7, 1], # green
            "mixed: good + overmerger": np.array([[0.7, 0.1, 0.1], [0, 0.7, 1]]).mean(axis=0),
            "mixed: good + oversplitter": np.array([[0.7, 0.1, 0.1], [0.6, 0.9, 0.6]]).mean(axis=0),
            "mixed: overmerger + oversplitter": np.array([[0.6, 0.9, 0.6], [0, 0.7, 1]]).mean(axis=0),
            "mixed: good + overmerger + oversplitter": np.array([[0.7, 0.1, 0.1], [0, 0.7, 1],[0.6, 0.9, 0.6]]).mean(axis=0),
            "false positive": [0, 0, 0] # black
        }        
    """
    # initialize dataframe to record quality
    sorted = []
    qualities = []

    # parallelized loop over sorted single-units
    for s_ix in range(len(s_ids)):
        
        quality = ""
        
        # get all matched true units
        g_ixs = np.where(scores[s_ix, :] > 0)[0]
        
        # false positives?
        if (scores[s_ix, g_ixs] <= chances[s_ix, g_ixs]).all():
            quality += "+ false_positive "
        else:
            # compute data to check oversplitter
            scores2 = np.delete(scores, s_ix, axis=0)
            chances2 = np.delete(chances, s_ix, axis=0)
            other_scores = scores2[:, g_ixs]
            other_chances = chances2[:, g_ixs]
            # good?
            if (scores[s_ix, g_ixs] >= thr_good).any():
                quality += "+ good "
            # overmerger?
            if (scores[s_ix, g_ixs] > chances[s_ix, g_ixs]).sum() > 1:
                quality += "+ overmerger "
            # oversplitter?
            if (other_scores > other_chances).any():
                quality += "+ oversplitter "

        sorted.append(s_ids[s_ix])
        qualities.append(quality)
    
    # make dataframe
    df = pd.DataFrame([sorted, qualities], index=["sorted", "quality"]).T
    
    # standardize the labels
    # independently of their ordering 
    # in each entry
    df = standardize_sorting_quality(df)
                    
    # unit-test
    assert len(df) == len(scores), "some sorted units were not qualified"
    return df


def qualify_sorted_units2(scores, s_ids, chances, thr_good, sorting_path: str):
    """qualify sorted units as false positives, good,
    overmerger, oversplitter and mixtures of these
    
    Writing entirely with numpy array accelerated 100X,
    compared to pandas dataframe.
    
    Args:
        scores (pd.DataFrame): agreement scores
        - index: sorted unit ids 
        chances (_type_): theoretically derived chance scores
        - per pairs of ground truth and sorted units
        thr_good (float): threshold for good unit

    Returns:
        pd.DataFrame: _description_
        
    Note:
        Their colors:
        cl = {"good": [0.7, 0.1, 0.1], # strong red
            "oversplitter": [0.6, 0.9, 0.6], # blue
            "overmerger": [0, 0.7, 1], # green
            "mixed: good + overmerger": np.array([[0.7, 0.1, 0.1], [0, 0.7, 1]]).mean(axis=0),
            "mixed: good + oversplitter": np.array([[0.7, 0.1, 0.1], [0.6, 0.9, 0.6]]).mean(axis=0),
            "mixed: overmerger + oversplitter": np.array([[0.6, 0.9, 0.6], [0, 0.7, 1]]).mean(axis=0),
            "mixed: good + overmerger + oversplitter": np.array([[0.7, 0.1, 0.1], [0, 0.7, 1],[0.6, 0.9, 0.6]]).mean(axis=0),
            "false positive": [0, 0, 0] # black
        }        
    """
    # initialize dataframe to record quality
    sorted = []
    qualities = []

    # parallelized loop over sorted single-units
    for s_ix in range(len(s_ids)):
        
        quality = ""
        
        # get all matched true units
        g_ixs = np.where(scores[s_ix, :] > 0)[0]
        
        # false positives?
        if (scores[s_ix, g_ixs] <= chances[s_ix, g_ixs]).all():
            quality += "+ false_positive "
        else:
            # compute data to check oversplitter
            scores2 = np.delete(scores, s_ix, axis=0)
            chances2 = np.delete(chances, s_ix, axis=0)
            other_scores = scores2[:, g_ixs]
            other_chances = chances2[:, g_ixs]
            # good?
            if (scores[s_ix, g_ixs] >= thr_good).any():
                quality += "+ good "
            # overmerger?
            if (scores[s_ix, g_ixs] > chances[s_ix, g_ixs]).sum() > 1:
                quality += "+ overmerger "
            # oversplitter?
            if (other_scores > other_chances).any():
                quality += "+ oversplitter "

        sorted.append(s_ids[s_ix])
        qualities.append(quality)
    
    # make dataframe
    df = pd.DataFrame([sorted, qualities], index=["sorted", "quality"]).T
    
    # standardize the labels
    # independently of their ordering 
    # in each entry
    df = standardize_sorting_quality(df)

    # set layer metadata
    df = set_layers(df, sorting_path)
    
    # unit-test
    assert len(df) == len(scores), "some sorted units were not qualified"
    return df


def standardize_sorting_quality(df):
    """standardize the labels used for qualities
    This lists all the possible combinations
    (eight) of qualities for a sorted neuron

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
        
    Notes:
    Their colors:
        cl = {"good": [0.7, 0.1, 0.1], # strong red
        "oversplitter": [0.6, 0.9, 0.6], # blue
        "overmerger": [0, 0.7, 1], # green
        "mixed: good + overmerger": np.array([[0.7, 0.1, 0.1], [0, 0.7, 1]]).mean(axis=0),
        "mixed: good + oversplitter": np.array([[0.7, 0.1, 0.1], [0.6, 0.9, 0.6]]).mean(axis=0),
        "mixed: overmerger + oversplitter": np.array([[0.6, 0.9, 0.6], [0, 0.7, 1]]).mean(axis=0),
        "mixed: good + overmerger + oversplitter": np.array([[0.7, 0.1, 0.1], [0, 0.7, 1],[0.6, 0.9, 0.6]]).mean(axis=0),
        "false positive": [0, 0, 0] # black
    }
    """

    is_q = df.quality.str.contains

    # 1. good
    df["quality"][
        (
            (is_q("good"))
            & (~is_q("overmerger"))
            & (~is_q("oversplitter"))
            & (~is_q("false_positive"))
        )
    ] = "good"

    # 2. "overmerger"
    df["quality"][
        (
            (~is_q("good"))
            & (is_q("overmerger"))
            & (~is_q("oversplitter"))
            & (~is_q("false_positive"))
        )
    ] = "overmerger"

    # 3. "oversplitter"
    df["quality"][
        (
            (~is_q("good"))
            & (~is_q("overmerger"))
            & (is_q("oversplitter"))
            & (~is_q("false_positive"))
        )
    ] = "oversplitter"

    # 4. mixed: "good + overmerger"
    df["quality"][
        (
            (is_q("good"))
            & (is_q("overmerger"))
            & (~is_q("oversplitter"))
            & (~is_q("false_positive"))
        )
    ] = "mixed: good + overmerger"
    
    # 5. mixed: "good + oversplitter"
    df["quality"][
        (
            (is_q("good"))
            & (~is_q("overmerger"))
            & (is_q("oversplitter"))
            & (~is_q("false_positive"))
        )
    ] = "mixed: good + oversplitter"
    
    # 6. mixed: "overmerger + oversplitter"
    df["quality"][
        (
            (~is_q("good"))
            & (is_q("overmerger"))
            & (is_q("oversplitter"))
            & (~is_q("false_positive"))
        )
    ] = "mixed: overmerger + oversplitter"

    # 7. mixed: "good + overmerger + oversplitter"
    df["quality"][
        (
            (is_q("good"))
            & (is_q("overmerger"))
            & (is_q("oversplitter"))
            & (~is_q("false_positive"))
        )
    ] = "mixed: good + overmerger + oversplitter"

    # 8. false positive
    df["quality"][
        (
            (~is_q("good"))
            & (~is_q("overmerger"))
            & (~is_q("oversplitter"))
            & (is_q("false_positive"))
        )
    ] = "false positive"
    return df


def set_layers(quality_df, sorting_path):
    """Add layers of sorted units in quality dataframe
    created by qualify_sorted_units() function

    Args:
        quality_df (_type_): _description_
        sorting_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    Sorting = si.load_extractor(sorting_path)
    layer = utils.standardize_layers(Sorting.get_property("layer"))
    loc = np.where(np.isin(Sorting.unit_ids, quality_df["sorted"]))[0]
    quality_df["layer"] = np.array(layer)[loc]
    return quality_df


def get_scores_for_layer(layer: list, scores: pd.DataFrame, Sorting):
    """filter the rows of the dataframe to
    get the scores for the sorted units which
    inferred location is in the list "layer"

    Args:
        layer (_type_): _description_
        scores (_type_): _description_
        Sorting (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # sorted units
    units = scores.index
    # all sorted units
    all_units = Sorting.unit_ids
    # get all units layers
    layers = utils.standardize_layers(Sorting.get_property("layer"))
    # find the selected sorted units
    loc = np.where(np.isin(all_units, units))[0]
    # return the scores of sorted units in 
    # the layer list
    layers = np.array(layers)[loc]
    return scores.iloc[np.isin(layers, layer), :]


def combine_quality_across_dense_probe(thr_good, sorting_path1, sorting_path2, sorting_path3, scores1, scores2, scores3, chance1, chance2, chance3):
    """_summary_

    Returns:
        _type_: _description_
    """
    # qualify all sorted single-units
    # depth 1 - qualify and add metadata
    ds1 = qualify_sorted_units(scores1.values, scores1.index.values, chance1.values, thr_good)
    ds1["Depth"] = 1
    ds1 = set_layers(ds1, sorting_path1)
    
    # depth 2 - qualify and add metadata
    ds2 = qualify_sorted_units(scores2.values, scores2.index.values, chance2.values, thr_good)
    ds2["Depth"] = 2
    ds2 = set_layers(ds2, sorting_path2)
    
    # depth 3 - qualify and add metadata
    ds3 = qualify_sorted_units(scores3.values, scores3.index.values, chance3.values, thr_good)
    ds3["Depth"] = 3
    ds3 = set_layers(ds3, sorting_path3)
    
    # concatenate
    df = pd.concat([ds1, ds2, ds3])
    return df


def get_scores_for_dense_probe(
    sort_path1: str,
    sort_path2: str,
    sort_path3: str,
    GT_path1,
    GT_path2,
    GT_path3,
    DELTA_TIME: float,
):

    # get scores for layer 1 and 2/3 from depth 1
    scores1, Sorting1, SortingTrue1 = get_score_single_unit(
        sort_path1, GT_path1, DELTA_TIME
    )
    scores1 = get_scores_for_layer(["L1", "L2/3"], scores1, Sorting1)

    # get scores for layer 4 and 5 from depth 1
    scores2, Sorting2, SortingTrue2 = get_score_single_unit(
        sort_path2, GT_path2, DELTA_TIME
    )
    scores2 = get_scores_for_layer(["L4", "L5"], scores2, Sorting2)

    # get scores for layer 6 from depth 1
    scores3, Sorting3, SortingTrue3 = get_score_single_unit(
        sort_path3, GT_path3, DELTA_TIME
    )
    scores3 = get_scores_for_layer(["L6"], scores3, Sorting3)
    return (
        {
            "scores1": scores1,
            "scores2": scores2,
            "scores3": scores3
        },
        {
            "Sorting1": Sorting1,
            "Sorting2": Sorting2,
            "Sorting3": Sorting3
        },
        {
            "True1": SortingTrue1,
            "True2": SortingTrue2,
            "True3": SortingTrue3,
         }
    )
    
 
def get_chance_for_dense_probe(
    dur,
    dt,
    R_d1,
    R_d2,
    R_d3,
    scores1,
    scores2,
    scores3,
    Sorting1,
    Sorting2,
    Sorting3,
    True1,
    True2,
    True3
):
    """get the chance scores for the dense probe
    at the three depths

    Args:
        dur (_type_): _description_
        dt (_type_): _description_
        R_d1 (_type_): _description_
        R_d2 (_type_): _description_
        R_d3 (_type_): _description_
        scores1 (_type_): _description_
        scores2 (_type_): _description_
        scores3 (_type_): _description_
        Sorting1 (_type_): _description_
        Sorting2 (_type_): _description_
        Sorting3 (_type_): _description_
        True1 (_type_): _description_
        True2 (_type_): _description_
        True3 (_type_): _description_

    Returns:
        _type_: _description_
    """
    chance1 = precompute_chance_score(
        R_d1, scores1, Sorting1, True1, dur, dt
    )
    chance2 = precompute_chance_score(
        R_d2, scores2, Sorting2, True2, dur, dt
    )
    chance3 = precompute_chance_score(
        R_d3, scores3, Sorting3, True3, dur, dt
    )
    return {"chance1": chance1, "chance2": chance2, "chance3": chance3}


def plot_ratio_by_exp(ax, df, cl, legend, legend_cfg):

    # count classified unit qualities
    plot_df = (
        df[["quality", "experiment"]]
        .groupby(["quality", "experiment"])
        .value_counts()
        .reset_index(name="count")
    )

    # convert count to percentages
    pivot_df = plot_df.pivot_table(
        index="experiment", columns="quality", values="count", fill_value=0
    )
    pivot_df_percentage = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    # assign a unique color to each unique quality
    # in the dataframe
    ordered_cl = [cl[q] for q in np.sort(df["quality"].unique())]

    # stacked bar plot
    ax = pivot_df_percentage.plot(
        ax=ax,
        kind="bar",
        stacked=True,
        color=ordered_cl,
        rot=0,
        width=0.9,
    )

    # Annotate counts
    for experiment_index, experiment in enumerate(pivot_df.index):
        bottom = 0  # Initialize the bottom for stacking

        for quality in pivot_df.columns:
            count = pivot_df.loc[
                experiment, quality
            ]  # Raw count from the original data
            percentage = pivot_df_percentage.loc[
                experiment, quality
            ]  # Corresponding percentage

            if count > 0:  # Only annotate if count > 0
                ax.text(
                    experiment_index,  # x position (the bar index)
                    bottom
                    + percentage / 2,  # y position (middle of the stacked bar segment)
                    str(int(count)),  # Text annotation (convert count to string)
                    ha="center",
                    va="center",
                    color="black",
                )
            bottom += percentage  # Update the bottom for the next stack

    # legend
    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1.5),
            ncol=1,
            **legend_cfg,
        )
    else:
        ax.get_legend().remove()
        
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.spines["right"].set_visible(False)

    # label axis
    ax.set_xlabel(df["sorter"].unique()[0])
    ax.set_ylabel("Single-unit ratio")
    ax.set_yticks([0, 25, 50, 75, 100], [0, 0.25, 0.50, 0.75, 1])
    return ax


def _plot_ratio_by_sorter(ax, df, cl, legend_cfg, lgd: bool, ylabel: bool):
    """stacked bar plot of the ratio of
    each unit quality class by experiment

    Args:
        df (_type_): _description_
        cl (_type_): _description_
        legend_cfg (_type_): _description_

    Returns:
        _type_: _description_
    """

    # count classified unit qualities
    plot_df = (
        df[["quality", "sorter"]]
        .groupby(["quality", "sorter"])
        .value_counts()
        .reset_index(name="count")
    )

    # convert count to percentages
    pivot_df = plot_df.pivot_table(
        index="sorter", columns="quality", values="count", fill_value=0
    )
    pivot_df_percentage = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    # assign a unique color to each unique quality
    # in the dataframe
    ordered_cl = [cl[q] for q in np.sort(df["quality"].unique())]

    # stacked bar plot
    ax = pivot_df_percentage.plot(
        ax=ax,
        kind="bar",
        stacked=True,
        color=ordered_cl,
        rot=0,
        width=0.9,
    )

    # Annotate counts
    for experiment_index, experiment in enumerate(pivot_df.index):
        bottom = 0  # Initialize the bottom for stacking

        for quality in pivot_df.columns:
            count = pivot_df.loc[
                experiment, quality
            ]  # Raw count from the original data
            percentage = pivot_df_percentage.loc[
                experiment, quality
            ]  # Corresponding percentage

            if count > 0:  # Only annotate if count > 0
                ax.text(
                    experiment_index,  # x position (the bar index)
                    bottom
                    + percentage / 2,  # y position (middle of the stacked bar segment)
                    str(int(count)),  # Text annotation (convert count to string)
                    ha="center",
                    va="center",
                    color="black",
                )
            bottom += percentage  # Update the bottom for the next stack

    # legend
    if lgd:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(8, 1),
            ncol=1,
            **legend_cfg,
        )
    else: 
        ax.get_legend().remove()

        
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.spines["right"].set_visible(False)

    # label axis
    #ax.set_xlabel(df["sorter"].unique()[0])
    if ylabel:
        ax.set_ylabel("Single-unit ratio")
        ax.set_yticks([0, 25, 50, 75, 100], [0, 0.25, 0.50, 0.75, 1])
    else:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
    return ax


def plot_ratio_by_sorter(ax, df, color, lgd_cfg):

    ax[0] = _plot_ratio_by_sorter(
        ax[0], df[df["sorter"] == "KS4"], color, lgd_cfg, True, True
    )
    ax[1] = _plot_ratio_by_sorter(
        ax[1], df[df["sorter"] == "KS3"], color, lgd_cfg, False, False
    )
    ax[2] = _plot_ratio_by_sorter(
        ax[2], df[df["sorter"] == "KS2.5"], color, lgd_cfg, False, False
    )
    ax[3] = _plot_ratio_by_sorter(
        ax[3], df[df["sorter"] == "KS2"], color, lgd_cfg, False, False
    )
    ax[4] = _plot_ratio_by_sorter(
        ax[4], df[df["sorter"] == "KS"], color, lgd_cfg, False, False
    )
    ax[5] = _plot_ratio_by_sorter(
        ax[5], df[df["sorter"] == "HS"], color, lgd_cfg, False, False
    )    
    return ax


def _plot_ratio_by_layer(ax, df, cl, legend_cfg, lgd: bool, ylabel: bool, xlabel=False):
    """stacked bar plot of the ratio of
    each unit quality class

    Args:
        df (_type_): _description_
        cl (_type_): _description_
        legend_cfg (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # count classified unit qualities
    plot_df = (
        df[["quality", "layer"]]
        .groupby(["quality", "layer"])
        .value_counts()
        .reset_index(name="count")
    )

    # convert count to percentages
    pivot_df = plot_df.pivot_table(
        index="layer", columns="quality", values="count", fill_value=0
    )
    pivot_df_percentage = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    # assign a unique color to each unique quality
    # in the dataframe
    ordered_cl = [cl[q] for q in np.sort(df["quality"].unique())]

    # stacked bar plot
    ax = pivot_df_percentage.plot(
        ax=ax,
        kind="bar",
        stacked=True,
        color=ordered_cl,
        rot=0,
        width=0.75,
        linewidth=0.1,
        edgecolor="k",
    )

    # Annotate counts
    # - only above 5% are visible
    # - pink: contains "good" in quality, else does not.
    for experiment_index, experiment in enumerate(pivot_df.index):
        bottom = 0  # Initialize the bottom for stacking

        for quality in pivot_df.columns:
            count = pivot_df.loc[
                experiment, quality
            ]  # Raw count from the original data
            percentage = pivot_df_percentage.loc[
                experiment, quality
            ]  # Corresponding percentage

            if percentage > 5:  # Only annotate if percentage > 0.05
                if "good" in quality:
                    ax.text(
                        experiment_index,  # x position (the bar index)
                        bottom
                        + percentage / 2,  # y position (middle of the stacked bar segment)
                        str(int(count)),  # Text annotation (convert count to string)
                        ha="center",
                        va="center",
                        color="pink",
                        fontsize=5,
                        fontweight=1000
                    )
                else:
                    ax.text(
                        experiment_index,  # x position (the bar index)
                        bottom
                        + percentage / 2,  # y position (middle of the stacked bar segment)
                        str(int(count)),  # Text annotation (convert count to string)
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=5,                        
                    )
            bottom += percentage  # Update the bottom for the next stack

    # legend
    if lgd:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(6, 1),
            ncol=1,
            **legend_cfg,
        )
    else: 
        ax.get_legend().remove()
        
    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # label axis
    if ylabel:
        ax.set_ylabel("Single-unit ratio")
        ax.set_yticks([0, 25, 50, 75, 100], [0, 0.25, 0.50, 0.75, 1])
    else:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
    if xlabel:
       ax.set_xlabel(df["sorter"].unique()[0])
    else:
        ax.set_xlabel("")
    return ax


def plot_ratio_by_layer(ax, df, color, lgd_cfg):
    """_summary_

    Args:
        ax (_type_): _description_
        df (_type_): _description_
        color (_type_): _description_
        lgd_cfg (_type_): _description_
    
    Usage:
        fig, ax = plt.subplots(1,5)
        ax = plot_ratio_by_layer(ax, df, color, lgd_cfg)
        

    Returns:
        _type_: _description_
    """
    ax[0] = _plot_ratio_by_layer(
        ax[0], df[df["layer"] == "L1"], color, lgd_cfg, True, True
    )
    ax[1] = _plot_ratio_by_layer(
        ax[1], df[df["layer"] == "L2/3"], color, lgd_cfg, False, False
    )
    ax[2] = _plot_ratio_by_layer(
        ax[2], df[df["layer"] == "L4"], color, lgd_cfg, False, False
    )
    ax[3] = _plot_ratio_by_layer(
        ax[3], df[df["layer"] == "L5"], color, lgd_cfg, False, False
    )
    ax[4] = _plot_ratio_by_layer(
        ax[4], df[df["layer"] == "L6"], color, lgd_cfg, False, False
    )
    return ax