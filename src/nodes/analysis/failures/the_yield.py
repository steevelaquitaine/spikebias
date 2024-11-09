import spikeinterface as si
import pandas as pd
import numpy as np
from src.nodes import utils

def get_yield(sorters: list, exps: list, sort_paths: list):
    """get yield per sorter, experiment and layer

    Args:
        sorters (list): _description_
        exps (list): _description_
        sort_paths (list): paths of the SortingExtractors

    Usage:
    
        # setup
        sorters = ["ks4", "ks3", "ks2_5", "ks2", "ks", "hs"]
        exps = ["S"] * len(sorters)
        sort_paths = [KS4_nb, KS3_nb, KS2_5_nb, KS2_nb, KS_nb, HS_nb]

        # get yields dataframe
        df = get_yield(sorters, exps, sort_paths)
        df
        
    Returns:
        pd.DataFrame: yield data by layer, experiment and sorter
    """
    # note: for ks and hs no quality metrics was found (except "quality" will all values
    # being "unsorted" without definition)
    df = pd.DataFrame()

    for s_i, sort_path in enumerate(sort_paths):

        df2 = pd.DataFrame()

        # load SortingExtractor
        Sorting = si.load_extractor(sort_path)

        # record metadata
        df2["layer"] = utils.standardize_layers(Sorting.get_property("layer"))

        # case KSLabel exists
        if not Sorting.get_property("KSLabel") is None:

            # calculate yield per layer
            df2["KSLabel"] = Sorting.get_property("KSLabel") == "good"
            df2 = df2.groupby(["layer"], as_index=False).sum()
            df2 = df2.rename({"KSLabel": "yield"}, axis="columns")

        else:
            df2 = df2.groupby(["layer"], as_index=False).size()
            df2 = df2.rename({"size": "yield"}, axis="columns")

        # record metadata
        df2["experiments"] = exps[s_i]
        df2["sorters"] = sorters[s_i]
        df = pd.concat([df, df2], axis=0)
    return df


def get_yield_per_site(sorters: list, exps: list, sort_paths: list, rec_path: str):
    """get yield per site for each sorter, experiment and layer

    Args:
        sorters (list): _description_
        exps (list): _description_
        sort_paths (list): paths of the SortingExtractors

    Usage:
    
        # setup
        sorters = ["ks4", "ks3", "ks2_5", "ks2", "ks", "hs"]
        exps = ["S"] * len(sorters)
        sort_paths = [KS4_nb, KS3_nb, KS2_5_nb, KS2_nb, KS_nb, HS_nb]

        # get yields dataframe
        df = get_yield(sorters, exps, sort_paths)
        df
        
    Returns:
        pd.DataFrame: yield data by layer, experiment and sorter
    """
    # note: for ks and hs no quality metrics was found (except "quality" will all values
    # being "unsorted" without definition)
    df = pd.DataFrame()

    # loop over spike sorters
    for s_i, sort_path in enumerate(sort_paths):

        df2 = pd.DataFrame()

        # load SortingExtractor
        Sorting = si.load_extractor(sort_path)

        # record metadata
        df2["layer"] = utils.standardize_layers(Sorting.get_property("layer"))

        # case KSLabel exists
        if not Sorting.get_property("KSLabel") is None:

            # calculate yield per layer
            df2["KSLabel"] = Sorting.get_property("KSLabel") == "good"
            df2 = df2.groupby(["layer"], as_index=False).sum()
            df2 = df2.rename({"KSLabel": "yield"}, axis="columns")

        else:
            df2 = df2.groupby(["layer"], as_index=False).size()
            df2 = df2.rename({"size": "yield"}, axis="columns")
         
        # record metadata
        df2["experiments"] = exps[s_i]
        df2["sorters"] = sorters[s_i]

        # record number of site metadata
        rec_ns = si.load_extractor(rec_path)
        layers = utils.standardize_layers(rec_ns.get_property("layers"))
        n_sites_df = (
            pd.DataFrame(layers, columns=["layer"]).groupby("layer", as_index=False).size()
        )
        n_sites_df = n_sites_df.rename({"size": "n_sites"}, axis="columns")
        df2 = n_sites_df.merge(df2)
        
        # concatenate
        df = pd.concat([df, df2], axis=0)        
        
    # calculate yield per site
    df["yield"] /= df["n_sites"]
    df.reset_index(drop=True, inplace=True)
    df2 = df.rename({"yield": "yield per site"}, axis="columns")
    return df2


def get_gt_yield_per_site(exp: str, gt_path: str, rec_path: str):

    # load SortingExtractor
    SortingTrue = si.load_extractor(gt_path)

    # layers of each true unit withing 50 ums of the probe
    df3 = pd.DataFrame()
    df3["layer"] = utils.standardize_layers(SortingTrue.get_property("layer"))
    print(df3.shape[0])
    
    # drop silent units (can't be detected)
    n_spikes = SortingTrue.get_total_num_spikes()
    not_silent = np.where(pd.DataFrame.from_dict(n_spikes, orient="index") > 0)[0]
    df3 = df3.iloc[not_silent]
    print("nb of silent gt units:", len(df3) - len(not_silent))
    
    # count units per layer
    df3 = df3.groupby("layer", as_index=False).size()
    df3 = df3.rename({"size": "yield_theory"}, axis="columns")

    # record nb of site metadata
    rec_ns = si.load_extractor(rec_path)
    layers = utils.standardize_layers(rec_ns.get_property("layers"))
    n_sites_df = (
        pd.DataFrame(layers, columns=["layer"]).groupby("layer", as_index=False).size()
    )
    n_sites_df = n_sites_df.rename({"size": "n_sites"}, axis="columns")
    df3 = n_sites_df.merge(df3)

    # record metadata
    df3["experiments"] = exp
    df3["yield_theory"] /= df3["n_sites"]
    df3 = df3.rename({"yield_theory": "yield per site (theory)"}, axis="columns")
    return df3


def count_ctx_sites(record_path: str):
    """count the number of electrode sites in the cortex

    Args:
        REC_ns (str): path of the RecordingExtractor

    Returns:
        int: _description_
    """
    Recording = si.load_extractor(record_path)
    layers = utils.standardize_layers(Recording.get_property("layers"))
    return sum(np.array(layers) != "Outside")
