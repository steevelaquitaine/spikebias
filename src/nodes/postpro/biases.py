"""postprocessing functions
"""
import numpy as np
import shutil
from spikeinterface import comparison


def label_false_positives(Sorting, SortingTrue, SORTING_PATH: str, save:bool):
    """label false positive sorted neurons and save to
    SortingExtractor properties as "false_positive"

    Args:
        GT_SORTING_PATH_1 (_type_): _description_
        KS3_SORTING_PATH_1 (_type_): _description_
        save (bool)

    Returns: 
        Sorting: Sorting extractor
    """
    # get true/sorted matching object
    matching = comparison.compare_sorter_to_ground_truth(
        SortingTrue, Sorting, exhaustive_gt=True
    )
    
    # label false positives
    false_positives = (
        (np.sum(matching.agreement_scores, axis=0) == 0)
        .values.astype(int)
        .tolist()
    )

    # check that unit ids are correct
    assert all(
        Sorting.unit_ids == matching.agreement_scores.columns
    ), "should be the same unit ids"

    # add false positive labels to properties
    Sorting.set_property("false_positives", false_positives)

    # save
    if save:
        shutil.rmtree(SORTING_PATH, ignore_errors=True)
        Sorting.save(folder=SORTING_PATH)
    return Sorting