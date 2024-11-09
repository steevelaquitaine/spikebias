from scipy.stats import poisson
# import spikeinterface as si
# import shutil
# from spikeinterface import comparison
# from src.nodes.postpro.cell_matching import get_SpikeInterface_matching_object


def get_p_chance_hit(fr: float, delta_time: float):
    """derive the chance probability of hits
    (coincidences between two independent sorted and
    ground truth unit spike trains)
    
    We should use the firing rate of the less firing
    of the two. It determines the expected maximum
    possible number of coincidences.

    Args:
        fr (float): firing rate in spikes/ms
        delta_time (float): SpikeInterface delta_time interval in ms

    Returns:
        _type_: _description_
    """
    k = 0  # we want the probability of k=0 coincidences
    interval_ms = 2 * delta_time  # time interval in ms
    n_sp = interval_ms * fr  # expected nb of coincidences
    return 1.0 - poisson.pmf(k=k, mu=n_sp)


def get_unit_chance_agreement_score(fr_gt: float, fr_s: float, rec_dur: float, p_chance_hit: float):
    """get unit chance scorey
    
    The chance scorey metrics should change with the ground truth firing rate.
    It is not the case with the current calculation.
    Intuition: the more a ground truth unit spikes within the duration of recording (say 600 secs),
    the more spikes will be missed when compared a sorting unit of a fixed firing rate.
    The increasing number of misses should decrease the value of the chance score metrics,
    which is currently not the case.

    Args:
        fr_gt (float): ground truth firing rate (spikes/secs)
        fr_s (float): sorted unit firing rate (spikes/secs)
        p_chance_hit (float): chance probability of hits
        rec_dur (float): recording duration
    """
    # nb of spikes
    n_gt = fr_gt * rec_dur
    n_s = fr_s * rec_dur

    # nb of hits, false positives, misses
    # - the smallers spike train min(n_gt, n_s) determines
    # the maximum possible number of hits
    n_h = p_chance_hit * min(n_gt, n_s)
    n_fp = n_s - n_h
    n_m = n_gt - n_h
    return n_h / (n_h + n_m + n_fp)



# def get_sorting_accuracies(GT_SORTING_PATH, KS3_SORTING_PATH):
#     """returns max accuracy across sorted units for each ground truth
#     """
#     matching = get_SpikeInterface_matching_object(GT_SORTING_PATH, KS3_SORTING_PATH)
#     return matching.agreement_scores.max(axis=1).sort_values(ascending=False)


# def label_sorting_accuracies(Sorting, SortingTrue, GT_SORTING_PATH:str, save:bool):
#     """Saves "sorting_accuracy" property to Ground truth Sorting Extractor

#     Args:
#         Sorting
#         SortingTrue
#     Returns:
#         SortingTrue
#     """
#     # best match sorted and true unist
#     matching = comparison.compare_sorter_to_ground_truth(
#         SortingTrue, Sorting, exhaustive_gt=True
#     )
    
#     # calculate matching accuracy
#     accuracy = matching.agreement_scores.max(axis=1)

#     # unit-test
#     assert all(SortingTrue.unit_ids == accuracy.index.tolist()), "unit ids do not match"

#     # add unit accuracies to properties
#     SortingTrue.set_property("sorting_accuracy", accuracy.tolist())

#     # save SortingExtractor
#     if save:
#         shutil.rmtree(GT_SORTING_PATH, ignore_errors=True)
#         SortingTrue.save(folder=GT_SORTING_PATH)
#     return SortingTrue
