import spikeinterface as si
import numpy as np
import pandas as pd
import shutil
from random import choices
import yaml 
import logging
import logging.config
from src.nodes import utils

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def label_firing_rates(Sorting, Recording, sorting_path:str, save:bool):
    """Save sorted units' firing rate property in Sorting Extractor

    Args:
        Sorting (Sorting Extractor):
        Recording (_type_): spikeinterface's Recording Extractor
        save (bool): write or not

    Returns:
        Sorting Extractor : Saves Sorting Extractor with "firing_rates" property as metadata:
    """
    rates = []
    for unit_id in Sorting.get_unit_ids():
        st = Sorting.get_unit_spike_train(unit_id=unit_id)
        rates += [len(st) / Recording.get_total_duration()]

    # add firing rates to properties
    Sorting.set_property("firing_rates", list(rates))

    # save
    if save:
        shutil.rmtree(sorting_path, ignore_errors=True)
        Sorting.save(folder=sorting_path)
    return Sorting
        

def get_sorted_unit_meta(sorted_path: str):
    """_summary_

    Returns:
        pd.DataFrame: _description_
    """
    # load Sorting extractor
    Sorting = si.load_extractor(sorted_path)

    # record
    unit_id_all = Sorting.unit_ids.tolist()
    # spike count / duration of the sorting (600 s): the stored "firing_rates" property of some sortings (e.g. the evoked regime
    # sampled at 40 kHz) were computed over 606 s
    counts = Sorting.count_num_spikes_per_unit()
    firing_rate_all = [counts[u] / 600 for u in Sorting.unit_ids]
    layer_all = Sorting.get_property("layer").tolist()
    layer_all = utils.standardize_layers(layer_all)
    KSLabel_all = utils.get_kslabel(Sorting)
    amplitude_all = utils.get_amplitude(Sorting)

    # store in dataframe
    return pd.DataFrame(
        np.array(
            [
                layer_all,
                firing_rate_all,
                KSLabel_all,
                amplitude_all,
            ]
        ).T,
        index=unit_id_all,
        columns=[
            "layer",
            "firing_rate",
            "kslabel",
            "amplitude",
        ],
    )


def get_synth_unit_meta(sorted_path: str):
    """_summary_

    Returns:
        pd.DataFrame: _description_
    """
    # load Sorting extractor
    Sorting = si.load_extractor(sorted_path)

    # record
    unit_id_all = Sorting.unit_ids.tolist()
    firing_rate_all = (
        Sorting.get_property("firing_rates").astype(np.float32).tolist()
    )
    KSLabel_all = utils.get_kslabel(Sorting)
    amplitude_all = utils.get_amplitude(Sorting)

    # store in dataframe
    return pd.DataFrame(
        np.array(
            [
                ["L5"]*len(unit_id_all),
                firing_rate_all,
                KSLabel_all,
                amplitude_all,
            ]
        ).T,
        index=unit_id_all,
        columns=[
            "layer",
            "firing_rate",
            "kslabel",
            "amplitude",
        ],
    )


def bootstrap_log_fr_std(firing_rate: list, N_BOOT: int):
    std_boot_all = []
    for ix in range(N_BOOT):
        fr_boot_i = choices(firing_rate, k=len(firing_rate))
        log_fr_boot_i = np.log10(fr_boot_i)
        std_boot_all.append(np.std(log_fr_boot_i))
    return std_boot_all


def plot_single_unit_ratio(ax, df_vivo, df_silico_sp, df_silico_ev, df_silico_nb, legend_cfg):
    """plot the proportions of single and multi-units
    sorted from the neuropixels probe recording
    """
    # setup figure
    shift = 0.3
    text_xpos_vivo = -0.3 + shift
    text_xpos_sili_sp = 0.7 + shift
    text_xpos_sili_ev = 1.7 + shift
    text_xpos_sili_nb = 2.7 + shift

    # multi-unit and single unit
    # colors
    color = np.array(
        [
            [1, 1, 1],
            [0.2, 0.2, 0.2],
        ]
    )

    # single-unit count
    n_single_units_vivo = sum(df_vivo["kslabel"] == "good")
    n_single_units_sili_sp = sum(df_silico_sp["kslabel"] == "good")
    n_single_units_sili_ev = sum(df_silico_ev["kslabel"] == "good")
    n_single_units_sili_nb = sum(df_silico_nb["kslabel"] == "good")
    
    # multi-unit count
    n_mu_vivo = df_vivo.shape[0] - n_single_units_vivo
    n_mu_sili_sp = df_silico_sp.shape[0] - n_single_units_sili_sp
    n_mu_sili_ev = df_silico_ev.shape[0] - n_single_units_sili_ev
    n_mu_sili_nb = df_silico_nb.shape[0] - n_single_units_sili_nb

    # build dataset
    df = pd.DataFrame()
    df["M"] = np.array([n_single_units_vivo, n_mu_vivo]) / df_vivo.shape[0]
    df["NS"] = (
        np.array([n_single_units_sili_sp, n_mu_sili_sp]) / df_silico_sp.shape[0]
    )
    df["E"] = (
        np.array([n_single_units_sili_ev, n_mu_sili_ev]) / df_silico_ev.shape[0]
    )
    df["S"] = (
        np.array([n_single_units_sili_nb, n_mu_sili_nb]) / df_silico_nb.shape[0]
    )

    # bar plot
    df.T.plot.bar(
        ax=ax, stacked=True, color=color, edgecolor=(0.5, 0.5, 0.5), rot=0, width=0.8, linewidth=0.5
    )
    
    # add unit counts
    ax.annotate(
        f"""{n_mu_vivo}""",
        (text_xpos_vivo, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_vivo}""",
        (text_xpos_vivo, 0.13),
        ha="center",
        color="k",
        rotation=0,
    )
    ax.annotate(
        f"""{n_mu_sili_sp}""",
        (text_xpos_sili_sp, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_sili_sp}""",
        (text_xpos_sili_sp, 0.13),
        ha="center",
        color="k",
        rotation=0,
    )
    # evoked 
    ax.annotate(
        f"""{n_mu_sili_ev}""",
        (text_xpos_sili_ev, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_sili_ev}""",
        (text_xpos_sili_ev, 0.03),
        ha="center",
        color="k",
        rotation=0,
    )
    # synthetic
    ax.annotate(
        f"""{n_mu_sili_nb}""",
        (text_xpos_sili_nb, 0.8),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_sili_nb}""",
        (text_xpos_sili_nb, 0.1),
        ha="center",
        color="k",
        rotation=0,
    )    
    ax.legend(
        ["single-unit", "multi-unit"],
        loc="upper left",
        bbox_to_anchor=(0, 1.25),
        **legend_cfg,
    )
    ax.set_ylabel("Proportion (ratio)")
    ax.set_xlabel("Experiment")
    return ax


def fix_firing_rates(df, Sorting, path, duration_sec=600, verbose=True):
    """Recompute the firing rates (spikes/s) of a sorting whose stored `firing_rates` were not divided by the duration of the sorting

    The stored firing rates of the DANDI sorting of the evoked regime sorted with Kilosort3 (10 minutes) are the spike counts divided
    by the duration of the whole 1-hour recording (3,618 s) instead of 10 minutes (about 6 times too low). A sorting is corrected when
    the duration implied by its stored rates (spike count / stored rate) is more than 1.5 times longer than `duration_sec`.

    Args:
        df (pd.DataFrame): unit metadata of the sorting (get_sorted_unit_meta), indexed by unit id
        Sorting: the SortingExtractor
        path (str): path of the sorting (for the message)
        duration_sec (float): duration of the sorting

    Returns:
        pd.DataFrame: df with corrected "firing_rate" if needed
    """
    counts = Sorting.count_num_spikes_per_unit()
    n_spikes = np.array([counts[u] for u in Sorting.unit_ids])
    implied_duration = n_spikes.sum() / np.nansum(Sorting.get_property("firing_rates").astype(float))
    if implied_duration > 1.5 * duration_sec:
        if verbose:
            print(f"  corrected {path}: stored rates imply a duration of {implied_duration:.0f} s, recomputed over {duration_sec} s")
        df = df.copy()
        df["firing_rate"] = (pd.Series(n_spikes, index=Sorting.unit_ids).loc[df.index.astype(Sorting.unit_ids.dtype)].values / duration_sec).astype(np.float32)
    return df
