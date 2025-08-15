

import spikeinterface as si
from spikeinterface import comparison
import numpy as np


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


def get_accuracy_for(
    KS4_nb_10m,
    GT_nb_10m,
    DT,
):
    # ground truth
    SortingTrue_nb = si.load_extractor(GT_nb_10m)

    SortingTrue_nb = SortingTrue_nb.frame_slice(
        start_frame=0, end_frame=10 * 60 * SortingTrue_nb.sampling_frequency
    )

    # KS4
    Sorting = si.load_extractor(KS4_nb_10m)
    scores = get_scores(SortingTrue_nb, Sorting, DT)
    scores = scores.loc[:, Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]]
    ac_k4 = np.array(scores.max(axis=1).sort_values(ascending=False))
    return ac_k4


def get_accuracy(
    KS4_nb_10m,
    KS3_nb_10m,
    KS2_5_nb_10m,
    KS2_nb_10m,
    KS_nb_10m,
    HS_nb_10m,
    GT_nb_10m,
    DT,
):
    # ground truth
    SortingTrue_nb = si.load_extractor(GT_nb_10m)
    SortingTrue_nb = SortingTrue_nb.frame_slice(
        start_frame=0, end_frame=10 * 60 * SortingTrue_nb.sampling_frequency
    )

    # KS4
    Sorting = si.load_extractor(KS4_nb_10m)
    scores = get_scores(SortingTrue_nb, Sorting, DT)
    scores = scores.loc[:, Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]]
    ac_k4 = np.array(scores.max(axis=1).sort_values(ascending=False))

    # KS3
    Sorting = si.load_extractor(KS3_nb_10m)
    scores = get_scores(SortingTrue_nb, Sorting, DT)
    scores = scores.loc[:, Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]]
    ac_k3 = np.array(scores.max(axis=1).sort_values(ascending=False))

    # KS2.5
    Sorting = si.load_extractor(KS2_5_nb_10m)
    scores = get_scores(SortingTrue_nb, Sorting, DT)
    scores = scores.loc[:, Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]]
    ac_k25 = np.array(scores.max(axis=1).sort_values(ascending=False))

    # KS2.5
    Sorting = si.load_extractor(KS2_5_nb_10m)
    scores = get_scores(SortingTrue_nb, Sorting, DT)
    scores = scores.loc[:, Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]]
    ac_k25 = np.array(scores.max(axis=1).sort_values(ascending=False))

    # KS2
    Sorting = si.load_extractor(KS2_nb_10m)
    scores = get_scores(SortingTrue_nb, Sorting, DT)
    scores = scores.loc[:, Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]]
    ac_k2 = np.array(scores.max(axis=1).sort_values(ascending=False))

    # KS
    Sorting = si.load_extractor(KS_nb_10m)
    scores = get_scores(SortingTrue_nb, Sorting, DT)
    ac_k = np.array(scores.max(axis=1).sort_values(ascending=False))

    # HS
    Sorting = si.load_extractor(HS_nb_10m)
    scores = get_scores(SortingTrue_nb, Sorting, DT)
    ac_h = np.array(scores.max(axis=1).sort_values(ascending=False))

    return np.vstack([ac_k4, ac_k3, ac_k25, ac_k2, ac_k, ac_h])