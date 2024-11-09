import numpy as np
import pandas as pd

def compute_raster(stim_epochs: pd.DataFrame, ttp: np.array):
    """compute raster for a single unit as a list of list of 
    timestamps expressed as sample indices within stim_epochs 
    start and end columns for each epoch (stim_epoch indices)

    Args:
        stim_epochs (pd.DataFrame): _description_
        ttp (np.array): all timestamps

    Returns:
        raster(list(list)): list of list of timestamps within 
        each stimulus epoch
    """
    
    # create raster data
    raster = []
    # loop over stimulus epochs
    for ix in range(len(stim_epochs)):
        # get epoch start
        start_s = stim_epochs["start_sample"].iloc[ix]
        # get epoch end
        end_s = stim_epochs["end_sample"].iloc[ix]
        # locate all spikes in between
        loc_ix = np.where((start_s <= ttp) & (ttp < end_s))[0]
        # timestamp in sample count since the stimulus
        ttp_to_stim = ttp[loc_ix] - start_s
        raster.append(ttp_to_stim.tolist())
    return raster