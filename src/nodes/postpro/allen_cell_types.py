import numpy as np
import xarray as xr
import pandas as pd 
from matplotlib import pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


def load_session_data(session_id, manifest_path):

    # get high level overview of the Neuropixels Visual Coding dataset
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # download and read session data (not too heavy, 1.68 GB)
    session = cache.get_session_data(session_id)

    # check laser experiments only 
    assert session.ecephys_session_id >= 789848216, "Make sure the experiment used a laser, which is the most reliable optotagging method"
    return session


def optotagging_spike_counts(session, time_resolution=0.0005, cortex_area='VIS'):
    """_summary_

    Args:
        session (_type_): _description_
        time_resolution (float, optional): _description_. Defaults to 0.0005. # 0.5 ms bins

    Returns:
        _type_: _description_
    """
    # align spikes to light pulses
    trials = session.optogenetic_stimulation_epochs[(session.optogenetic_stimulation_epochs.duration > 0.009) & \
                                                    (session.optogenetic_stimulation_epochs.duration < 0.02)]

    # get units from cortical areas
    units = session.units[session.units.ecephys_structure_acronym.str.match(cortex_area)]
    
    bin_edges = np.arange(-0.01, 0.025, time_resolution)

    time_resolution = np.mean(np.diff(bin_edges))
    spike_matrix = np.zeros((len(trials), len(bin_edges), len(units)))
    for unit_idx, unit_id in enumerate(units.index.values):
        spike_times = session.spike_times[unit_id]
        for trial_idx, trial_start in enumerate(trials.start_time.values):
            in_range = (spike_times > (trial_start + bin_edges[0])) * \
                       (spike_times < (trial_start + bin_edges[-1]))
            binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
            spike_matrix[trial_idx, binned_times, unit_idx] = 1
    return xr.DataArray(
        name='spike_counts',
        data=spike_matrix,
        coords={
            'trial_id': trials.index.values,
            'time_relative_to_stimulus_onset': bin_edges,
            'unit_id': units.index.values
        },
        dims=['trial_id', 'time_relative_to_stimulus_onset', 'unit_id']
    )


def plot_optotagging_response(da, units, time_resolution, bin_edges, ):
    """_summary_

    Args:
        da (_type_): _description_
        units (_type_): _description_
        time_resolution (_type_): _description_
        bin_edges (_type_): _description_
    """
    # setup figure
    plt.figure(figsize=(5,10))
    
    # plot
    plt.imshow(da.mean(dim='trial_id').T / time_resolution,
               extent=[np.min(bin_edges), np.max(bin_edges),
                       0, len(units)],
               aspect='auto', vmin=0, vmax=200)
    
    for bound in [0.0005, 0.0095]:
        plt.plot([bound, bound],[0, len(units)], ':', color='white', linewidth=1.0)
    
    # add legend
    plt.xlabel('Time (s)')
    plt.ylabel('Unit #')
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('Mean firing rate (Hz)')


def get_optotagged_neurons(session, is_plot=False):

    # align
    da = optotagging_spike_counts(session, time_resolution=0.0005, cortex_area='VIS')

    # compare evoked to baseline
    baseline = da.sel(time_relative_to_stimulus_onset=slice(-0.01,-0.002))
    baseline_rate = baseline.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / 0.008
    evoked = da.sel(time_relative_to_stimulus_onset=slice(0.001,0.009))
    evoked_rate = evoked.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / 0.008

    # get parvalbumin (positive) interneurons
    # add 1 to prevent divide-by-zero errors
    cre_pos_units = da.unit_id[(evoked_rate / (baseline_rate + 1)) > 2].values 

    # plot_optotagging_response
    if is_plot:
        plot_optotagging_response(da)
    return cre_pos_units


def get_spont_firing_rates(session, neurons:list):
    """get spontaneous firing rates for selected 
    neuron ids in "neurons"

    Args:
        session (_type_): _description_
        neurons (list): _description_

    Returns:
        _type_: _description_
    """
    # get spike times from the first block of drifting gratings presentations 
    spontaneous_ids = session.stimulus_presentations.loc[
        (session.stimulus_presentations['stimulus_name'] == 'spontaneous')
    ].index.values

    # count presentations
    n_presentations = len(spontaneous_ids)

    # initialize dataframe (will be dropped)
    firing_rate_all = session.conditionwise_spike_statistics(
        stimulus_presentation_ids=[spontaneous_ids[0]],
        unit_ids=neurons
    )["spike_count"]

    # count spike per unit (rows) per presentation (col)
    for pres_i in np.arange(0, n_presentations):

        # this presentation
        this_pres = spontaneous_ids[pres_i]

        # spike count per neuron
        spike_count = session.conditionwise_spike_statistics(
            stimulus_presentation_ids=[this_pres],
            unit_ids=neurons
        )["spike_count"]

        # get this presentation duration
        duration = session.stimulus_presentations.loc[this_pres]["duration"]

        # get neurons' firing rate
        firing_rate = spike_count / duration

        # stack firing rates per presentation (column-wise)
        firing_rate_all = pd.merge(firing_rate_all, firing_rate, left_index=True, right_index=True)

    # drop first placeholder column
    firing_rate_all = firing_rate_all.iloc[:,1:]

    # average fr over presentations
    mean_fr = firing_rate_all.mean(axis=1)
    mean_fr.index = mean_fr.index.droplevel("stimulus_condition_id")
    return mean_fr


def load_all_sorted_spont_firing_rates(session, cortex_area='VIS'):
    """get all neurons spontaneous firing rates

    Args:
        session (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get spike times from the first block of drifting gratings presentations 
    spontaneous_ids = session.stimulus_presentations.loc[
        (session.stimulus_presentations['stimulus_name'] == 'spontaneous')
    ].index.values

    # count presentations
    n_presentations = len(spontaneous_ids)

    # get units from "VIS"
    all_units = session.units[session.units.ecephys_structure_acronym.str.match(cortex_area)]
    all_units = list(all_units.index)

    # initialize dataframe (will be dropped)
    firing_rate_all = session.conditionwise_spike_statistics(
        stimulus_presentation_ids=[spontaneous_ids[0]],
        unit_ids=all_units
    )["spike_count"]

    # count spike per unit (rows) per presentation (col)
    for pres_i in np.arange(0, n_presentations):

        # this presentation
        this_pres = spontaneous_ids[pres_i]

        # spike count per neuron
        spike_count = session.conditionwise_spike_statistics(
            stimulus_presentation_ids=[this_pres],
            unit_ids=all_units
        )["spike_count"]

        # get this presentation duration
        duration = session.stimulus_presentations.loc[this_pres]["duration"]

        # get neurons' firing rate
        firing_rate = spike_count / duration

        # stack firing rates per presentation (column-wise)
        firing_rate_all = pd.merge(firing_rate_all, firing_rate, left_index=True, right_index=True)

    # drop first placeholder column
    firing_rate_all = firing_rate_all.iloc[:,1:]

    # average fr over presentations
    mean_fr = firing_rate_all.mean(axis=1)
    mean_fr.index = mean_fr.index.droplevel("stimulus_condition_id")
    return mean_fr