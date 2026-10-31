"""Candidate features that describe sorted units whose quality metrics look like those of degraded
units although they are 'good' (low firing rate, variable spike amplitudes, few spikes).
None of them uses the ground truth."""

import numpy as np
import pandas as pd
from scipy import stats
from spikeinterface.postprocessing import compute_spike_amplitudes

BIN_S = 5.0     # bin size of the firing-rate series (s)


def amplitude_features(a, t, duration):
    """spike amplitude distribution of a unit (a: |amplitudes|, t: spike times in s)"""
    med = np.median(a)
    out = dict(
        amp_cv=np.std(a) / np.mean(a),
        amp_skew=stats.skew(a),
        amp_kurt=stats.kurtosis(a),
        amp_lowfrac=np.mean(a < 0.5 * med),
        amp_p10_p90=np.percentile(a, 10) / np.percentile(a, 90),
        amp_drift=stats.spearmanr(t, a)[0] if len(a) > 3 else np.nan,
    )
    return out


def firing_features(t, duration):
    """temporal firing pattern of a unit (t: spike times in s)"""
    isi = np.diff(t)
    counts = np.histogram(t, bins=np.arange(0, duration + BIN_S, BIN_S))[0]
    lv = 3 / (len(isi) - 1) * np.sum(((isi[:-1] - isi[1:]) / (isi[:-1] + isi[1:])) ** 2) if len(isi) > 2 else np.nan
    return dict(
        isi_cv=np.std(isi) / np.mean(isi),
        isi_short_frac=np.mean(isi < 0.01),
        isi_lv=lv,
        bin_fano=np.var(counts) / np.mean(counts),
        bin_empty_frac=np.mean(counts == 0),
        bin_trend=stats.spearmanr(np.arange(len(counts)), counts)[0],
    )


def spatial_features(template, y):
    """footprint of a unit; template: samples x channels, y: channel depths (um)"""
    peak = -template.min(0)
    main = int(np.argmax(peak))
    order = np.argsort(peak)[::-1]
    w = np.where(peak > 0.2 * peak[main], peak, 0.0) ** 2
    ybar = np.sum(w * y) / np.sum(w)
    tm = template[:, main]
    trough = int(np.argmin(tm))
    after = tm[trough:]
    return dict(
        depth=y[main],
        main_to_neigh=peak[main] / np.mean(peak[order[1:4]]),
        n_ch_30=int((peak >= 0.3 * peak[main]).sum()),
        spread_um=float(np.sqrt(np.sum(w * (y - ybar) ** 2) / np.sum(w))),
        trough_to_peak=int(np.argmax(after)),
        peak_asym=float(after.max() / -tm[trough]),
    )


def cross_unit_features(templates, trains, fs, rates):
    """similarity of each unit with the other sorted units (split/merge candidates)"""
    flat = templates.reshape(templates.shape[0], -1)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    S = flat @ flat.T
    np.fill_diagonal(S, -1)
    rows = []
    for i in range(len(flat)):
        j = int(np.argmax(S[i]))
        ti, tj = trains[i], trains[j]
        # share of this unit's spikes within 1 ms of a spike of its most similar unit
        k = np.clip(np.searchsorted(tj, ti), 1, len(tj) - 1)
        near = np.minimum(np.abs(ti - tj[k - 1]), np.abs(ti - tj[k])) < 0.001 if len(tj) > 1 else np.zeros(len(ti), bool)
        union = np.sort(np.concatenate([ti, tj]))
        viol_union = np.mean(np.diff(union) < 0.0015)
        viol_i = np.mean(np.diff(ti) < 0.0015) if len(ti) > 1 else 0
        rows.append(dict(max_sim=S[i, j], n_sim_06=int((S[i] > 0.6).sum()), rate_ratio_twin=rates[i] / rates[j],
                         coinc_twin=float(np.mean(near)), union_viol_increase=viol_union - viol_i))
    return pd.DataFrame(rows)


def waveform_features(We):
    """spike-to-spike consistency of the waveforms on the unit's main channel"""
    rows = {}
    for u in We.unit_ids:
        inds = We.sparsity.unit_id_to_channel_indices[u]
        W = We.get_waveforms(u)
        pos = int(np.argmin(W.mean(0).min(0)))
        w = W[:, :, pos]
        tm = w.mean(0)
        trough = int(np.argmin(tm))
        ta = w[:, trough]
        corr = np.array([np.corrcoef(wi, tm)[0, 1] for wi in w])
        rows[int(u)] = dict(trough_cv=np.std(ta) / abs(np.mean(ta)), shape_corr_med=np.nanmedian(corr), shape_lowcorr=np.nanmean(corr < 0.7))
    return pd.DataFrame.from_dict(rows, orient="index")


def compute_extra_features(We, duration=600.0, job_kwargs=None):
    """DataFrame of candidate features indexed by (int) unit id, with a 'group' mapping"""
    fs = We.sampling_frequency
    ids = [int(u) for u in We.unit_ids]
    amps = compute_spike_amplitudes(We, peak_sign="neg", outputs="by_unit", load_if_exists=True, **(job_kwargs or {}))[0]
    trains = [We.sorting.get_unit_spike_train(u) / fs for u in We.unit_ids]
    templates = We.get_all_templates(mode="average")
    y = We.recording.get_channel_locations()[:, 1]
    rates = np.array([len(t) / duration for t in trains])
    rows = []
    for k, u in enumerate(We.unit_ids):
        a = np.abs(np.asarray(amps[u]))
        r = {}
        r.update(amplitude_features(a, trains[k], duration))
        r.update(firing_features(trains[k], duration))
        r.update(spatial_features(templates[k], y))
        rows.append(r)
    df = pd.DataFrame(rows, index=ids)
    df = df.join(cross_unit_features(templates, trains, fs, rates).set_index(pd.Index(ids)))
    df = df.join(waveform_features(We))
    df["ks_amplitude"] = We.sorting.get_property("Amplitude")
    df["ks_contam_pct"] = We.sorting.get_property("ContamPct")
    return df


GROUPS = {
    "amplitude distribution": ["amp_cv", "amp_skew", "amp_kurt", "amp_lowfrac", "amp_p10_p90", "amp_drift"],
    "firing pattern": ["isi_cv", "isi_short_frac", "isi_lv", "bin_fano", "bin_empty_frac", "bin_trend"],
    "spatial footprint and depth": ["depth", "main_to_neigh", "n_ch_30", "spread_um", "trough_to_peak", "peak_asym"],
    "waveform consistency": ["trough_cv", "shape_corr_med", "shape_lowcorr"],
    "split/merge candidates": ["max_sim", "n_sim_06", "rate_ratio_twin", "coinc_twin", "union_viol_increase"],
    "kilosort cluster properties": ["ks_amplitude", "ks_contam_pct"],
}
