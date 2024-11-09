# Description:   Simplified re-implementation for Python 3 of simwriter's 'random_dot_flash'
#                stimulus generator using NeuroTools' adapting Markov process
# Author:        C. Pokorny
# Date:          08-09/2021

import os
import numpy as np
import matplotlib.pyplot as plt


def write_spike_file(out_map, out_file):
    """
    Writes output spike trains to file
    """
    out_path = os.path.split(out_file)[0]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(out_file, 'w') as f:
        f.write('/scatter\n')
        for gid, spike_times in out_map.items():
            if spike_times is not None:
                for t in spike_times:
                    f.write(f'{t:f}\t{gid:d}\n')


def map_groups_to_fibers(spike_map, grp_gids):
    """
    Maps spikes of groups of fibers to individual fiber GIDs
    """
    out_map = {}
    for grp_idx, spike_times in spike_map.items():
        if spike_times is not None:
            for g in grp_gids[grp_idx]:
                out_map[g] = spike_times

    return out_map


def plot_PSTHs(out_map, stim_train, time_windows, bin_size=10, save_path=None):
    """
    Plots PSTHs of generated stimulus spikes
    bin_size: Time resolution (ms)
    """
    num_fibers = len(out_map)
    num_patterns = max(stim_train) + 1

    pattern_spikes = [[]] * num_patterns
    for sidx, pidx in enumerate(stim_train):
        spikes = np.hstack([out_map[gid][np.logical_and(out_map[gid] >= time_windows[sidx], out_map[gid] < time_windows[sidx + 1])] for gid in out_map.keys()]).flatten()
        spikes = spikes - time_windows[sidx] # Re-align to stimulus onset
        pattern_spikes[pidx] = pattern_spikes[pidx] + [spikes]

    num_stim_per_pattern = [len(p) for p in pattern_spikes]
    pattern_spikes = [np.hstack(p) for p in pattern_spikes]

    t_max = np.max(np.diff(time_windows))
    num_bins = np.round(t_max / bin_size).astype(int)
    bins = np.arange(num_bins + 1) * bin_size

    pattern_PSTHs = [1e3 * np.histogram(pattern_spikes[pidx], bins=bins)[0] / (bin_size * num_fibers * num_stim_per_pattern[pidx]) for pidx in range(num_patterns)]

    plt.figure(figsize=(2 * num_patterns, 4))
    pat_colors = plt.cm.jet(np.linspace(0, 1, num_patterns))
    for pidx in range(num_patterns):
        plt.subplot(1, num_patterns, pidx + 1)
        plt.bar(bins[:-1], pattern_PSTHs[pidx], width=bin_size, align='edge', color=pat_colors[pidx, :])
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing rate (Hz)')
        plt.title(f'Pattern {pidx} (N={num_stim_per_pattern[pidx]})', fontweight='bold')
    plt.suptitle('Stimulus PSTHs', fontweight='bold')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'PSTHs.png'), dpi=300)


def plot_spikes(spike_map, time_axis, stim_train, time_windows, grp_label, reindex=False, save_path=None):
    """
    Plots spike trains for groups of spikes
    """
    dt = np.mean(np.diff(time_axis))
    num_patterns = max(stim_train) + 1
    pat_colors = plt.cm.jet(np.linspace(0, 1, num_patterns))

    if reindex: # Re-assign consecutive indices starting at 0
        index_table = {}
        count_idx = 0
        for grp_idx in sorted(spike_map.keys()):
            index_table.update({grp_idx: count_idx})
            count_idx += 1
    else: # Keep original indices
        index_table = {k: k for k in sorted(spike_map.keys())}

    plt.figure(figsize=(10, 5))
    for grp_idx, spike_times in spike_map.items():
        if spike_times is not None:
            plt.plot(spike_times, np.full_like(spike_times, index_table[grp_idx]), 'k|', markersize=2)
    plt.xlim((-0.5 * dt, time_windows[-1] + 0.5 * dt))
    plt.ylim((min(index_table.values()) - 0.5, max(index_table.values()) + 0.5))
    for stim_idx, pat_idx in enumerate(stim_train):
        plt.plot(np.full(2, time_windows[stim_idx]), plt.ylim(), color=pat_colors[pat_idx, :], zorder=0)
    plt.xlabel('Time (ms)')
    plt.ylabel(grp_label)
    plt.title('Spike signals')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'spikes_per_{grp_label.lower().replace(" ", "_")}.png'), dpi=300)


def generate_spikes(rate_map, time_axis, spike_seed, bq, tau):
    """
    Generates random spike trains for each group of fibers
    """
    t_max = max(time_axis) + np.mean(np.diff(time_axis))

    np.random.seed(spike_seed)
    spike_map = {}
    for grp_idx, rate_sig in rate_map.items():
        if rate_sig is not None:
            if rate_sig.sum() == 0:
                spike_map[grp_idx] = np.array([])
            else:
                bq_sig = bq * np.ones_like(rate_sig)
                spike_map[grp_idx] = inh_adaptingmarkov_generator(rate_sig, bq_sig, tau, time_axis, t_max)
        else:
            spike_map[grp_idx] = None

    return spike_map


def plot_rate_signals(rate_map, time_axis, stim_train, time_windows, save_path=None):
    """
    Plots analog rate signals (per group of fibers)
    """
    num_samples_total = len(time_axis)
    dt = np.mean(np.diff(time_axis))
    num_groups = len(rate_map.keys())
    num_patterns = max(stim_train) + 1
    pat_colors = plt.cm.jet(np.linspace(0, 1, num_patterns))

    rate_signals = []
    for grp_idx, rate_sig in rate_map.items():
        if rate_sig is None:
            rate_signals.append(np.zeros(num_samples_total))
        else:
            rate_signals.append(rate_sig)
    rate_signals = np.vstack(rate_signals)

    plt.figure(figsize=(10, 5))
    plt.imshow(rate_signals, aspect='auto', interpolation='nearest', cmap='gray_r', origin='lower', extent=(-0.5 * dt, time_windows[-1] + 0.5 * dt, -0.5, num_groups - 0.5))
    plt.ylim(plt.ylim())
    for stim_idx, pat_idx in enumerate(stim_train):
        plt.plot(np.full(2, time_windows[stim_idx]), plt.ylim(), color=pat_colors[pat_idx, :])
    plt.colorbar(label='Rate (Hz)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Group idx')
    plt.title('Rate signals')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'rate_signals.png'), dpi=300)


def generate_rate_signals(stim_train, pattern_grps, num_groups, start, duration_stim, duration_blank, rate_min, rate_max, dt=1.0):
    """
    Generates analog rate signals map (per group of fibers)
    dt: Time resolution (ms) of analog signal generation (default: 1.0 ms)
    """
    # Initialize rate signals
    num_samples_init = np.round(start / dt).astype(int)
    rate_map = {grp_idx: [rate_min + np.zeros(num_samples_init)] for grp_idx in range(num_groups)}

    # Construct series of stimuli
    stim_len = duration_stim + duration_blank # (ms)
    num_samples_stim = np.round(stim_len / dt).astype(int)
    num_samples_sig = np.round(duration_stim / dt).astype(int)
    num_samples_blank = num_samples_stim - num_samples_sig

    single_stim = np.concatenate((rate_max * np.ones(num_samples_sig), rate_min + np.zeros(num_samples_blank)))
    single_stim_empty = rate_min + np.zeros_like(single_stim)

    for pat_idx in stim_train:
        for grp_idx in range(num_groups):
            if grp_idx in pattern_grps[pat_idx]:
                rate_map[grp_idx].append(single_stim)
            else:
                rate_map[grp_idx].append(single_stim_empty)

    # Concatenate 
    for grp_idx in range(num_groups):
        rate_map[grp_idx] = np.concatenate(rate_map[grp_idx])

    num_samples_total = len(stim_train) * num_samples_stim + num_samples_init
    time_axis = np.arange(num_samples_total) * dt # (ms)
    time_windows = np.arange(num_samples_init, num_samples_total + 1, num_samples_stim) * dt # (ms)

    return rate_map, time_axis, time_windows


def plot_stim_series(stim_train, save_path=None):
    """
    Plots the series of stimuli (stim train) and a histogram
    of pattern occurrences
    """
    num_patterns = max(stim_train) + 1
    pat_colors = plt.cm.jet(np.linspace(0, 1, num_patterns))
    
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 2, 1)
    for stim_idx, pat_idx in enumerate(stim_train):
        plt.plot(np.full(2, stim_idx), [0, pat_idx], '-', color=pat_colors[pat_idx, :])
        plt.plot(stim_idx, pat_idx, 'o', color=pat_colors[pat_idx, :])
    plt.xlim(plt.xlim())
    plt.plot(plt.xlim(), np.zeros(2), color='grey', zorder=0)
    plt.title(f'Stimulus train')
    plt.yticks(np.unique(stim_train))
    plt.xlabel('Stimulus number')
    plt.ylabel('Pattern index')

    plt.subplot(1, 2, 2)
    plt.hist(stim_train, bins=np.arange(-0.5, num_patterns), edgecolor='k')
    plt.xticks(range(num_patterns))
    plt.xlabel('Pattern index')
    plt.ylabel('Count')
    plt.title('Pattern histogram')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'stim_train.png'), dpi=300)


def generate_stim_series(num_patterns, num_stimuli, series_seed, strict_enforce_p=True, p_seeds=None, overexpressed_tuples=None):
    """
    Generates random series of stimuli (stim train)
    """
    if p_seeds is None:
        p_seeds = np.ones(num_patterns, dtype=float)
    p_seeds = np.array(p_seeds) / np.sum(p_seeds)
    assert len(p_seeds) == num_patterns, 'ERROR: Pattern probabilities do not match number of stimulus patterns!'

    pattern_idx = np.arange(num_patterns)
    if overexpressed_tuples is not None:
        tl_dict, pattern_idx, p_seeds = gen_overexpression_dict(pattern_idx, overexpressed_tuples, p_seeds)
    else:
        tl_dict = dict([(i, [i]) for i in pattern_idx])

    p_seeds = np.array(p_seeds) / np.sum(p_seeds)
    assert len(p_seeds) == len(pattern_idx), 'ERROR: Pattern probabilities mismatch!'
    np.random.seed(series_seed)
    if strict_enforce_p:
        over_fac = np.mean([len(v) for v in list(tl_dict.values())])
        n_per_seed = np.ceil((num_stimuli / over_fac) * p_seeds)
        select_from = np.hstack([_seed * np.ones(int(_num), dtype=int)
                                    for _seed, _num in zip(pattern_idx, n_per_seed)])
        stim_train = np.random.permutation(select_from)
    else:
        stim_train = np.random.choice(pattern_idx, num_stimuli, p=p_seeds)
    stim_train = np.hstack([tl_dict[i] for i in stim_train])[:num_stimuli]
    assert len(stim_train) == num_stimuli, 'ERROR: Stim train mismatch!'

    return stim_train


def plot_spatial_patterns(pattern_pos2d, pattern_pos3d, pos2d, pos3d, pos2d_all, pos3d_all, save_path=None):
    """
    Plots spatial patterns consisting of groups of fibers
    """
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    if pattern_pos2d is not None:
        num_patterns = len(pattern_pos2d)
    elif pattern_pos3d is not None:
        num_patterns = len(pattern_pos3d)
    else: # No pattern to plot
        return
    pat_colors = plt.cm.jet(np.linspace(0, 1, num_patterns))

    if pattern_pos2d is not None:
        plt.figure(figsize=(15, 5))
        for p in range(num_patterns):
            plt.subplot(1, num_patterns, p + 1)
            plt.plot(pos2d_all[:, 0], pos2d_all[:, 1], '.', color='lightgrey', markersize=1)
            plt.plot(pos2d[:, 0], pos2d[:, 1], '.', color='darkgrey', markersize=1)
            plt.plot(pattern_pos2d[p][:, 0], pattern_pos2d[p][:, 1], '.', color=pat_colors[p, :], markersize=1)
            plt.axis('image')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Pattern {p}')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'patterns_2d.png'), dpi=300)

    if pattern_pos3d is not None:
        plt.figure(figsize=(15, 5))
        for p in range(num_patterns):
            plt.subplot(1, num_patterns, p + 1, projection='3d')
            plt.plot(pos3d_all[:, 0], pos3d_all[:, 1], pos3d_all[:, 2], '.', color='lightgrey', markersize=1)
            plt.plot(pos3d[:, 0], pos3d[:, 1], pos3d[:, 2], '.', color='darkgrey', markersize=1)
            plt.plot(pattern_pos3d[p][:, 0], pattern_pos3d[p][:, 1], pattern_pos3d[p][:, 2], '.', color=pat_colors[p, :], markersize=1)
            plt.gca().set_xlabel('x')
            plt.gca().set_ylabel('y')
            plt.gca().set_zlabel('z')
            plt.title(f'Pattern {p}')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'patterns_3d.png'), dpi=300)


def generate_spatial_pattern(gids, pos2d, pos3d, grp_idx, stimuli_seeds, sparsity):
    """
    Generates spatial patterns by randomly selecting groups of fibers
    """
    num_groups = max(grp_idx) + 1
    num_patterns = len(stimuli_seeds)
    num_groups_per_pattern = np.round(sparsity * num_groups).astype(int)
    pattern_grps = []
    for seed in stimuli_seeds:
        np.random.seed(seed)
        pattern_grps.append(np.sort(np.random.choice(num_groups, num_groups_per_pattern, replace=False)).tolist())

    pattern_gids = [np.hstack([gids[grp_idx == grp] for grp in pattern_grps[p]]) for p in range(num_patterns)]
    pattern_pos3d = [np.vstack([pos3d[grp_idx == grp, :] for grp in pattern_grps[p]]) for p in range(num_patterns)]
    if pos2d is None:
        pattern_pos2d = None
    else:
        pattern_pos2d = [np.vstack([pos2d[grp_idx == grp, :] for grp in pattern_grps[p]]) for p in range(num_patterns)]    

    return pattern_grps, pattern_gids, pattern_pos2d, pattern_pos3d


# Modified code from simwriter 1.1.0 (simwriter/libraries/generators/generators.py)
def gen_overexpression_dict(base_seeds, overexpressed, p_seeds):
    """
    Generates dict with overexpressed stimulus tuples always occurring in sequence
    """
    assert np.all([i in base_seeds for i in np.hstack(overexpressed)])
    all_ov = np.hstack(overexpressed)
    p_dict = dict([(i, p) for i, p in zip(base_seeds, p_seeds)])
    out_seeds = np.setdiff1d(base_seeds, all_ov).tolist()
    out_p = [p_dict[i] for i in out_seeds]
    tl_dict = dict([(i, [i]) for i in out_seeds])
    i = -1
    for ov in overexpressed:
        while i in base_seeds:
            i -= 1
        tl_dict[i] = ov
        out_seeds.append(i)
        out_p.append(np.mean([p_dict[o] for o in ov]))
        i -= 1
    return tl_dict, np.array(out_seeds), np.array(out_p)


# Modified code from NeuroTools 0.2.0
def poisson_generator(rate, t_start=0.0, t_stop=1000.0, debug=False):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if 
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
    -------
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)

    Examples:
    --------
        >> gen.poisson_generator(50, 0, 1000)
        >> gen.poisson_generator(20, 5000, 10000)

    See also:
    --------
        inh_poisson_generator, inh_gamma_generator, inh_adaptingmarkov_generator
    """

    #number = int((t_stop-t_start)/1000.0*2.0*rate)

    # less wasteful than double length method above
    n = (t_stop-t_start)/1000.0*rate
    number = np.ceil(n+3*np.sqrt(n))
    if number<100:
        number = min(5+np.ceil(2*n),100)

    if number > 0:
        isi = np.random.exponential(1.0/rate, number.astype(int))*1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes+=t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i==len(spikes):
        # ISI buf overrun

        t_last = spikes[-1] + np.random.exponential(1.0/rate, 1)[0]*1000.0

        while (t_last<t_stop):
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0/rate, 1)[0]*1000.0

        spikes = np.concatenate((spikes,extra_spikes))

        if debug:
            # print "ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (len(spikes),len(extra_spikes))
            print(f"ISI buf overrun handled. len(spikes)={len(spikes)}, len(extra_spikes)={len(extra_spikes)}")


    else:
        spikes = np.resize(spikes,(i,))

    if debug:
        return spikes, extra_spikes
    else:
        return spikes


# Modified code from NeuroTools 0.2.0
def inh_adaptingmarkov_generator(a, bq, tau, t, t_stop):

    """
    Returns a SpikeList whose spikes are an inhomogeneous
    realization (dynamic rate) of the so-called adapting markov
    process (see references). The implementation uses the thinning
    method, as presented in the references.

    This is the 1d implementation, with no relative refractoriness.
    For the 2d implementation with relative refractoriness, 
    see the inh_2dadaptingmarkov_generator.

    Inputs:
    -------
        a,bq    - arrays of the parameters of the hazard function where a[i] and bq[i] 
                 will be active on interval [t[i],t[i+1]]
        tau    - the time constant of adaptation (in milliseconds).
        t      - an array specifying the time bins (in milliseconds) at which to 
                 specify the rate
        t_stop - length of time to simulate process (in ms)

    Note:
    -----
        - t_start=t[0]

        - a is in units of Hz.  Typical values are available 
          in Fig. 1 of Muller et al 2007, a~5-80Hz (low to high stimulus)

        - bq here is taken to be the quantity b*q_s in Muller et al 2007, is thus
          dimensionless, and has typical values bq~3.0-1.0 (low to high stimulus)

        - tau_s has typical values on the order of 100 ms


    References:
    -----------

    Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier 
    Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
    Neural Comput. 2007 19: 2958-3010.

    Devroye, L. (1986). Non-uniform random variate generation. New York: Springer-Verlag.

    Examples:
    ---------
        See source:trunk/examples/stgen/inh_2Dmarkov_psth.py


    See also:
    ---------
        inh_poisson_generator, inh_gamma_generator, inh_2dadaptingmarkov_generator

    """

    if np.shape(t)!=np.shape(a) or np.shape(a)!=np.shape(bq):
        raise ValueError('shape mismatch: t,a,b must be of the same shape')

    # get max rate and generate poisson process to be thinned
    rmax = np.max(a)
    ps = poisson_generator(rmax, t_start=t[0], t_stop=t_stop)

    # return empty if no spikes
    if len(ps) == 0:
        return np.array([])

    isi = np.zeros_like(ps)
    isi[1:] = ps[1:]-ps[:-1]
    isi[0] = ps[0] #-0.0 # assume spike at 0.0

    # gen uniform rand on 0,1 for each spike
    rn = np.array(np.random.uniform(0, 1, len(ps)))

    # instantaneous a,bq for each spike

    idx=np.searchsorted(t,ps)-1
    spike_a = a[idx]
    spike_bq = bq[idx]

    keep = np.zeros(np.shape(ps), bool)

    # thin spikes

    i = 0
    t_last = 0.0
    t_i = 0
    # initial adaptation state is unadapted, i.e. large t_s
    t_s = 1000*tau

    while(i<len(ps)):
        # find index in "t" time, without searching whole array each time
        t_i = np.searchsorted(t[t_i:],ps[i],'right')-1+t_i

        # evolve adaptation state
        t_s+=isi[i]

        if rn[i]<a[t_i]*np.exp(-bq[t_i]*np.exp(-t_s/tau))/rmax:
            # keep spike
            keep[i] = True
            # remap t_s state
            t_s = -tau*np.log(np.exp(-t_s/tau)+1)
        i+=1


    spike_train = ps[keep]

    return spike_train
