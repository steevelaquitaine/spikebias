"""Waveform preprocessing variants to test whether they change the waveform-based quality
metrics (silhouette, snr) of sorted units. Each variant is self-contained (loads its own
waveforms) so that the variants can be run in parallel processes."""

import os
import numpy as np
import pandas as pd
import spikeinterface as si
from spikeinterface import extract_waveforms
from spikeinterface.core import ChannelSparsity
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import compute_quality_metrics

JOB1 = dict(n_jobs=1, progress_bar=False)


def load_memory_we(view):
    """fresh in-memory waveform extractor (waveforms loaded from disk)"""
    we0 = si.load_waveforms(view, with_recording=True)
    return we0, we0.select_units(we0.unit_ids)


def main_pos(W):
    """position of the main (largest trough) channel of a unit's waveforms (spikes x samples x channels)"""
    return int(np.argmin(W.mean(0).min(0)))


def t_svd(k):
    def f(W):
        out = np.empty_like(W)
        for c in range(W.shape[2]):
            M = W[:, :, c]
            mu = M.mean(0)
            U, S, Vt = np.linalg.svd(M - mu, full_matrices=False)
            out[:, :, c] = (U[:, :k] * S[:k]) @ Vt[:k] + mu
        return out
    return f


def t_amp_norm(W):
    tmpl = W.mean(0)
    p = main_pos(W)
    return (W / abs(tmpl[:, p].min())).astype(W.dtype)


def t_align(W, max_shift=6):
    p = main_pos(W)
    tmpl = W.mean(0)[:, p]
    t0 = int(np.argmin(tmpl))
    out = np.empty_like(W)
    for i in range(W.shape[0]):
        lo, hi = max(t0 - max_shift, 0), min(t0 + max_shift + 1, W.shape[1])
        s = t0 - (lo + int(np.argmin(W[i, lo:hi, p])))
        w = np.roll(W[i], s, axis=0)
        if s > 0:
            w[:s] = W[i, 0]
        elif s < 0:
            w[s:] = W[i, -1]
        out[i] = w
    return out


def top_channels_sparsity(we, n):
    """keep the n channels with the largest trough amplitude of each unit"""
    mask = np.zeros((len(we.unit_ids), we.get_num_channels()), dtype=bool)
    for i, u in enumerate(we.unit_ids):
        inds = we.sparsity.unit_id_to_channel_indices[u]
        amp = -we.get_template(u, mode="average").min(0)
        amp = amp if len(amp) == len(inds) else amp[inds]
        mask[i, inds[np.argsort(amp)[::-1][:n]]] = True
    return ChannelSparsity(mask, we.unit_ids, we.channel_ids)


def restrict_to_main_channels(we, n):
    """keep only the n channels with the largest trough amplitude of each unit (in memory: waveforms and sparsity)"""
    mask = np.zeros((len(we.unit_ids), we.get_num_channels()), dtype=bool)
    for i, u in enumerate(we.unit_ids):
        W = we._memory_objects["wfs_arrays"][u]
        inds = we.sparsity.unit_id_to_channel_indices[u]
        top = np.sort(np.argsort(W.mean(0).min(0))[:n])          # most negative troughs, in channel order
        we._memory_objects["wfs_arrays"][u] = W[:, :, top]
        mask[i, inds[top]] = True
    we.sparsity = ChannelSparsity(mask, we.unit_ids, we.channel_ids)
    we._template_cache = {}
    return we


def features(we, n_components=5, mode="by_channel_local", sparsity=None, with_snr=True):
    compute_principal_components(we, n_components=n_components, mode=mode, sparsity=sparsity, **JOB1)
    out = compute_quality_metrics(we, metric_names=["silhouette"], skip_pc_metrics=False, **JOB1)
    res = pd.DataFrame({"silhouette": out["silhouette"].values}, index=out.index.astype(int))
    if with_snr:
        res["snr"] = compute_quality_metrics(we, metric_names=["snr"], skip_pc_metrics=True, **JOB1)["snr"].values
    return res


def apply(we, t):
    for u in we.unit_ids:
        we._memory_objects["wfs_arrays"][u] = t(we._memory_objects["wfs_arrays"][u])
    return we


VARIANT_NAMES = (
    ["baseline settings (check)"]
    + [f"SVD denoising rank {k}" for k in (2, 3, 5)]
    + ["amplitude-normalized waveforms (silhouette only)", "trough-aligned spikes"]
    + [f"PCA on the {n} main channel(s)" for n in (1, 3, 6)]
    + [f"PCA {nc} components" for nc in (3, 10)]
    + ["PCA mode by_channel_global", "PCA mode concatenated (6 main channels)", "more spikes (cap 2000)"]
)


def compute_variant(name, view):
    """returns (name, DataFrame with silhouette and snr indexed by unit id)"""
    we0, we = load_memory_we(view)
    if name == "baseline settings (check)":
        return name, features(we)
    if name.startswith("SVD denoising rank"):
        return name, features(apply(we, t_svd(int(name.split()[-1]))))
    if name.startswith("amplitude-normalized"):
        return name, features(apply(we, t_amp_norm), with_snr=False)
    if name == "trough-aligned spikes":
        return name, features(apply(we, t_align))
    if name.startswith("PCA on the"):
        n = int(name.split()[3])
        return name, features(restrict_to_main_channels(we, n))
    if name.endswith("components"):
        return name, features(we, n_components=int(name.split()[1]))
    if name.startswith("PCA mode concatenated"):   # concatenated mode needs the same number of channels for every unit
        return name, features(restrict_to_main_channels(we, 6), mode="concatenated")
    if name.startswith("PCA mode"):
        return name, features(we, mode=name.split()[-1])
    if name == "more spikes (cap 2000)":
        we2 = extract_waveforms(we0.recording, we0.sorting, folder=None, mode="memory", sparse=True, sparsity=we0.sparsity,
                                ms_before=6, ms_after=6, max_spikes_per_unit=2000, seed=0, n_jobs=4, chunk_duration="1s", progress_bar=False)
        return name, features(we2)
    raise ValueError(name)
