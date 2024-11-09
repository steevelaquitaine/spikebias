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
    firing_rate_all = (
        Sorting.get_property("firing_rates").astype(np.float32).tolist()
    )
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
