"""nodes to concatenate campaign files

Returns:
    _type_: _description_
"""
import pandas as pd

def concat_raw_traces_and_spikes(trace_paths:list, spike_paths:list):
    """Starts from pickle raw traces and spike files and concatenate 
    them into a recording and spike file

    Args:
        trace_paths (list): _description_
        spike_paths (list): _description_

    Returns:
        _type_: _description_
    """

    # read first trace and spikes 
    trace_concat = pd.read_pickle(trace_paths[0])
    spike_concat = pd.read_pickle(spike_paths[0])

    for ix in range(1, len(trace_paths)):
        
        # concat traces
        # get subsequent trace
        trace_next = pd.read_pickle(trace_paths[ix])
        
        # update indices of subsequent trace
        concat_duration = trace_concat.index[-1]
        trace_next.index += concat_duration
        trace_concat = pd.concat([trace_concat, trace_next])

        # concat spikes
        spike_next = pd.read_pickle(spike_paths[ix])
        spike_next.index += concat_duration
        spike_concat = pd.concat([spike_concat, spike_next])
    return trace_concat, spike_concat