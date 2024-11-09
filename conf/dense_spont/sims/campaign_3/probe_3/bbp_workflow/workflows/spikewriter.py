"""
Functions to generates TC spike trains used as input to the SSCx circuit as part of bbp-workflow
(e.g. lognormally distributed spike trains to mimic whisker flicks
and inhomogenous adapting Markov processes to mimic adaptation to longer stimuli)
last update: András Ecker, 11.2021, James Isbister, 18.11.2022
"""


import os
import numpy as np


def generate_stim_series_num_stims(stim_start, num_stims, inter_stimulus_interval):
    """Generates stimulus time series (as delays in ms)"""
    return list(range(stim_start, stim_start + num_stims * inter_stimulus_interval, inter_stimulus_interval))
    # return np.arange(stim_start, stim_end, stim_rate)


def generate_stim_series(stim_start, stim_end, stim_rate):
    """Generates stimulus time series (as delays in ms)"""
    return np.arange(stim_start, stim_end, 1000./stim_rate)


def _random_pattern_order(stim_times, pattern_names, seed):
    """Randomizes pattern order (more equal representation of individual patterns than `np.random.choicce()`)"""
    q, r = np.divmod(len(stim_times), len(pattern_names))
    stim_order = [pattern_name for _ in range(q) for pattern_name in pattern_names]
    if r:
        stim_order.extend(pattern_names[-r:])
    # shuffle every 30 sec independently
    chunk_size = np.searchsorted(stim_times, stim_times[0]+30000)
    idx = np.arange(len(stim_times))
    split_idx = np.split(idx, np.arange(chunk_size, len(stim_times), chunk_size))
    for i, chunk_idx in enumerate(split_idx):
        np.random.seed(seed+i)
        np.random.shuffle(chunk_idx)
    return np.asarray(stim_order)[np.concatenate(split_idx)]


def generate_rnd_pattern_stim_series(stim_start, stim_end, stim_rate, pattern_names,
                                     seed, save_name, init_transient="J"):
    """Generates random series of patterns (and corresponding stimulus times).
    It has the option to activate a pattern (default to "J" which is 25% of all the base patterns in the pyramid scheme)
    as initial transient (high activity state outside of the analysed window)."""
    stim_times = generate_stim_series(stim_start, stim_end, stim_rate)
    if len(stim_times) == 10:
        stim_order = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])  # quick debug...
    else:
        stim_order = _random_pattern_order(stim_times, pattern_names, seed)
    if init_transient is not None:
        stim_times = np.insert(stim_times, 0, stim_start - 750)
        stim_order = np.insert(stim_order, 0, init_transient)
    if save_name is not None:
        _ensure_dir(os.path.dirname(save_name))
        with open(save_name, "w") as f:
            for t, pattern_name in zip(stim_times, stim_order):
                f.write("%i %s\n" % (t, pattern_name))
    # create a dict format which is handled easier in the rest of the functions
    stim_times_dict = {pattern_name: stim_times[stim_order == pattern_name] for pattern_name in pattern_names}
    return stim_times_dict


def load_rnd_pattern_stim_series(f_name):
    """Loads in random series of patterns from .txt file (it would be the option to save them to pickle...
    and then it would be easier to load... but this feels more general)"""
    stim_times, stim_order = [], []
    with open(f_name, "r") as f:
        for line in f:
            tmp = line.strip().split()
            stim_times.append(float(tmp[0]))
            stim_order.append(tmp[1])
    stim_times, stim_order = np.asarray(stim_times), np.asarray(stim_order)
    pattern_names = np.unique(stim_order)
    return {pattern_name: stim_times[stim_order == pattern_name] for pattern_name in pattern_names}


def extract_stim_times(stim_times_dict):
    """Extracts stimulus times (used for POm fibers) from pattern dict. (used in multi pattern setups)"""
    pattern_names = list(stim_times_dict.keys())
    stim_times = stim_times_dict[pattern_names[0]]
    for pattern_name in pattern_names[1:]:
        stim_times = np.concatenate((stim_times, stim_times_dict[pattern_name]))
    return np.sort(stim_times)


def _generate_rate_signal(t_start, t_end, stim_times, stim_duration, min_rate, max_rate, dt):
    """Generates rate signal: `stim_duration` long linear decrease from `max_rate` to `min_rate` at `stim_times`)"""
    assert (t_start < stim_times).all() and (stim_times < t_end).all(), "Stim times are outside the simulation interval"
    assert stim_duration < np.min(np.diff(stim_times, append=t_end)), "Stim times are too close!"
    t = np.arange(t_start, t_end+dt, dt)
    rate = min_rate * np.ones_like(t)
    for stim_time in stim_times:
        idx = np.where((t > stim_time) & (t < stim_time + stim_duration))[0]
        rate[idx] = np.linspace(max_rate, min_rate, len(idx))
    return t, rate


def generate_rate_signal(t_start, t_end, stim_times, stim_duration, min_rate, max_rate, dt=1.):
    """Generates (time dependent) firing rate signal used for spike generation"""
    if type(stim_times) == np.ndarray:
        return _generate_rate_signal(t_start, t_end, stim_times, stim_duration, min_rate, max_rate, dt)
    elif type(stim_times) == dict:
        rates_dict = {}
        for pattern_name, pattern_stim_times in stim_times.items():
            t, rate = _generate_rate_signal(t_start, t_end, pattern_stim_times, stim_duration,
                                            min_rate/len(stim_times), max_rate, dt)
            rates_dict[pattern_name] = rate
        return t, rates_dict
    else:
        raise RuntimeError("Unexpected type: %s. Please pass np.ndarray or dict of np.ndarrays" % type(stim_times))


def _generate_exp_rand_numbers(lambda_, n_rnds, seed):
    """MATLAB's random exponential number"""
    np.random.seed(seed)
    return -1.0 / lambda_ * np.log(np.random.rand(n_rnds))


def generate_hom_poisson_spikes(rate, t_start, t_end, seed):
    """Generates Poisson process: (interval times are exponentially distributed:
    X_i = -ln(U_i)/lambda_, where lambda_ is the rate and U_i ~ Uniform(0,1))"""
    expected_n = (t_end-t_start)/1000. * rate
    n_rnds = np.ceil(expected_n+3 * np.sqrt(expected_n))  # NeuroTools' way of determining the number of random ISIs
    rnd_isis = _generate_exp_rand_numbers(rate, int(n_rnds), seed) * 1000.  # ISIs in ms
    poisson_proc = np.cumsum(rnd_isis) + t_start
    if poisson_proc[-1] > t_end:
        return poisson_proc[poisson_proc <= t_end]
    else:
        i, extra_spikes = 1, []
        t_last = poisson_proc[-1] + _generate_exp_rand_numbers(rate, 1, seed+i)[0]
        while t_last < t_end:
            extra_spikes.append(t_last)
            i += 1
            t_last += _generate_exp_rand_numbers(rate, 1, seed+i)[0]
        return np.concatenate((poisson_proc, extra_spikes))


def merge_spike_trains(spike_times, spiking_gids):
    """Concatenates lists of `spike_times` and `spiking_gids` and sorts them (based on spike times)"""
    spike_times = np.concatenate(spike_times)
    spiking_gids = np.concatenate(spiking_gids)
    idx = np.argsort(spike_times)
    return spike_times[idx], spiking_gids[idx]


def generate_hom_poisson_spike_train(gids, rate, t_start, t_end, seed):
    """Generates Poisson spike trains for all gids"""
    spike_times, spiking_gids = [], []
    for gid in gids:
        spike_times_ = generate_hom_poisson_spikes(rate, t_start, t_end, seed+gid)
        spike_times.append(spike_times_)
        spiking_gids.append(gid * np.ones_like(spike_times_))
    return merge_spike_trains(spike_times, spiking_gids)


def generate_inh_adaptingmarkov_spikes(a, bq, tau, t, seed):
    """
    Generates inhomogenous adapting Markov process from Muller et al. 2007
    Method is based on the paper and the implementation is borrowed from NeuroTools
    :params a, bq: t shaped arrays with the time varying parameters of the hazard function
                   (a in Hz typically from 5-80Hz, bq is typically 3-1)
    :param tau: time constant of the adaptation (in ms)
    """
    assert a.shape == bq.shape == t.shape, "a, bq, and t must have the same shape"
    max_rate = np.max(a)
    poisson_spikes = generate_hom_poisson_spikes(max_rate, t_start=t[0], t_end=t[-1], seed=seed)
    isis = np.diff(poisson_spikes, prepend=0)
    np.random.seed(int(seed + max_rate))
    rnds = np.random.rand(len(isis))  # pre-generate Uniform(0, 1) for thinning
    i, t_i = 0, 0
    t_s = 1000 * tau  # initial state is unadapted (i.e. large t_s)
    keep = np.zeros_like(poisson_spikes, dtype=bool)
    while i < len(poisson_spikes):
        t_i = np.searchsorted(t[t_i:], poisson_spikes[i], "right") - 1 + t_i
        t_s += isis[i]
        hazard = a[t_i] * np.exp(-bq[t_i] * np.exp(-t_s / tau))  # see eq. (2.10) in Muller et al. 2007
        if rnds[i] < hazard / max_rate:
            keep[i] = True
            t_s = -tau * np.log(np.exp(-t_s / tau) + 1)  # see eq. (2.7) in Muller et al. 2007
        i += 1
    return poisson_spikes[keep]


def _generate_inh_adaptingmarkov_spike_train(gids, t, rate_signal, bq, tau, seed):
    """Generates spikes drawn from an inhomogenous adapting Markov process (see Muller et al. 2007) for all gids"""
    bq = bq * np.ones_like(rate_signal)
    spike_times, spiking_gids = [], []
    for gid in gids:
        spike_times_ = generate_inh_adaptingmarkov_spikes(rate_signal, bq, tau, t, seed+gid)
        spike_times.append(spike_times_)
        spiking_gids.append(gid * np.ones_like(spike_times_))
    return merge_spike_trains(spike_times, spiking_gids)


def generate_inh_adaptingmarkov_spike_train(gids, t, rate_signal, bq, tau, seed):
    """Generates spike traains used as input to the circuit
    see docs of: `_generate_inh_adaptingmarkov_spike_train()` and `generate_inh_adaptingmarkov_spikes()`"""
    if type(gids) == np.ndarray and type(rate_signal) == np.ndarray:
        return _generate_inh_adaptingmarkov_spike_train(gids, t, rate_signal, bq, tau, seed)
    elif type(gids) == dict and type(rate_signal) == dict:
        pattern_names = list(gids.keys())
        spike_times, spiking_gids = [], []
        for i, pattern_name in enumerate(pattern_names):
            spike_times_tmp, spiking_gids_tmp = _generate_inh_adaptingmarkov_spike_train(gids[pattern_name], t,
                                                rate_signal[pattern_name], bq, tau, seed+(i*1000))
            spike_times.append(spike_times_tmp)
            spiking_gids.append(spiking_gids_tmp)
        return merge_spike_trains(spike_times, spiking_gids)
    else:
        raise RuntimeError("Gids and rate both have to be either np.arrays or dicts")

import pandas as pd
from scipy.interpolate import interp1d
def generate_yu_svoboda_spikes(gids, vpm_input_df, base_seed):
    
    # psth = vpm_input_df.etl.q(layer=0, creline="VPM").iloc[0]['psth_mean']
    psth = vpm_input_df[(vpm_input_df['layer']==0) & (vpm_input_df['creline']=="VPM")].iloc[0]['psth_mean']
    psth_cut = psth[40:51]
    psth_cut = psth_cut - np.min(psth_cut)
    normalised_psth_cut = psth_cut / np.sum(psth_cut)
    f = interp1d(np.arange(len(psth_cut)), normalised_psth_cut)
    x = np.linspace(0.0, 10.0, num=10000, endpoint=True)
    fine_p = f(x)
    fine_p = fine_p / np.sum(fine_p)
    np.random.seed(base_seed + gids[-1])
    spike_times = np.random.choice(x, len(gids), p=fine_p)
    return spike_times, gids


def generate_yu_svoboda_spike_trains(stim_times, gids, data_for_vpm_input, base_seed):

    vpm_input_df = pd.read_parquet(data_for_vpm_input)

    spike_times, spiking_gids = [], []
    for i, stim_time in enumerate(stim_times):
        spike_times_, spiking_gids_ = generate_yu_svoboda_spikes(gids, vpm_input_df, base_seed)
        spike_times_ += stim_time
        spike_times.append(spike_times_)
        spiking_gids.append(spiking_gids_)

    return merge_spike_trains(spike_times, spiking_gids)


def ji_diamond_estimate_scaled_spikes(gids, base_seed):
    
    # x = [0.666, 1.998, 3.33, 4.662, 5.994]
    x = [1.0, 3.0, 5.0, 6.333, 7.666]
    y = [0., 0.10185185, 1., 0.11111111, 0.07407407]
    f = interp1d(x, y)
    new_x = np.linspace(np.min(x), np.max(x), num=10000, endpoint=True)
    fine_p = f(new_x)
    fine_p = fine_p / np.sum(fine_p)
    np.random.seed(base_seed + gids[-1])
    spike_times = np.random.choice(new_x, len(gids), p=fine_p)
    return spike_times, gids


def generate_ji_diamond_estimate_scaled_spike_trains(stim_times, gids, base_seed):

    spike_times, spiking_gids = [], []
    for i, stim_time in enumerate(stim_times):
        spike_times_, spiking_gids_ = ji_diamond_estimate_scaled_spikes(gids, base_seed)
        spike_times_ += stim_time
        spike_times.append(spike_times_)
        spiking_gids.append(spiking_gids_)

    return merge_spike_trains(spike_times, spiking_gids)

def generate_lognormal_spikes(gids, mu, sigma, spike_rate, base_seed):
    """Generates spike times drawn from a lognormal distribution"""
    size = int(len(gids) * spike_rate)
    np.random.seed(base_seed + gids[0])
    spike_times = np.random.lognormal(mu, sigma, size=size)
    np.random.seed(base_seed + gids[-1])
    spiking_gids = np.random.choice(gids, size=size, replace=True)
    return np.sort(spike_times), spiking_gids

def generate_lognormal_spike_train(stim_times, gids, mu, sigma, spike_rate, base_seed):
    """Generates lognormal spike trains at given stimulus times"""
    spike_times, spiking_gids = [], []
    for i, stim_time in enumerate(stim_times):
        spike_times_, spiking_gids_ = generate_lognormal_spikes(gids, mu, sigma, spike_rate, base_seed)
        spike_times_ += stim_time
        spike_times.append(spike_times_)
        spiking_gids.append(spiking_gids_)
    return merge_spike_trains(spike_times, spiking_gids)


def _ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def write_spikes(spike_times, spiking_gids, f_name):
    """Writes spikes to .dat file format expected by Neurodamus"""
    assert len(spike_times) == len(spiking_gids), "The length of spike times and gids don't match" \
                                                  "thus can't written to file."
    _ensure_dir(os.path.dirname(f_name))
    with open(f_name, "w") as f:
        f.write("/scatter\n")
        for t, gid in zip(spike_times, spiking_gids):
            f.write("%.2f %i\n" % (t, gid))
