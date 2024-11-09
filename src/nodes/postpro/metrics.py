import pandas as pd


def cast_hits_dict_as_dataframe(hits_dict: dict):
    """_summary_

    Args:
        hits_dict (dict): _description_

    Returns:
        _type_: _description_
    """
    return pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in hits_dict.items()])
    )


def get_hit_counts_for_a_true_units(out: dict):
    """count hits by sorted cell (index)

    Returns:
        pd.Series: _description_
    """
    df = cast_hits_dict_as_dataframe(out["sorted_unit_hits"])

    # count hits per sorted unit
    hit_count = df.T.value_counts()

    # format sorted units indices
    formatted_index = [int(ix[0]) for ix in hit_count.index]
    hit_count.index = formatted_index
    return hit_count.to_dict()


def get_event_count_truth(unit_id: int, Truth):
    """_summary_

    Args:
        unit_id (int): _description_
        Truth (_type_): _description_

    Returns:
        _type_: _description_
    """

    return len(Truth.get_unit_spike_train(unit_id=unit_id))


def get_event_count_sorting(unit_id: int, Sorting):
    """_summary_

    Args:
        unit_id (int): _description_
        Sorting (_type_): _description_

    Returns:
        _type_: _description_
    """
    return len(Sorting.get_unit_spike_train(unit_id=unit_id))


def get_agreement_score(
    true_unit: int,
    sorted_unit: int,
    hit_count: dict,
    event_counts_truth: dict,
    event_counts_sorting: dict,
):
    """_summary_

    Args:
        true_unit (int): _description_
        sorted_unit (int): _description_
        hit_count (dict): _description_
        event_counts_truth (dict): _description_
        event_counts_sorting (dict): _description_

    Returns:
        _type_: _description_
    """
    return hit_count[sorted_unit] / (
        event_counts_truth[true_unit]
        + event_counts_sorting[sorted_unit]
        - hit_count[sorted_unit]
    )
