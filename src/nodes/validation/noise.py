"""noise validation node

[TODO]: 

- move STATS to stats module

Returns:
    _type_: _description_
"""
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal # stats
import scikit_posthocs as sp
import torch
import copy

# SETUP PARAMETERS
RND_SEED = 0                # random seed
SFREQ_V = 30000          # sampling frequency
SFREQ_S = 40000        # sampling frequency
SFREQ_E = 20000        # sampling frequency
SFREQ_B = 32000        # sampling frequency
SF_HV = 20000          # horvath sampling frequency
SF_HS = 20000        # horvath model sampling frequency

def torch_mad(trace):
    return (trace - trace.mean(dtype=torch.float32)).abs().mean(dtype=torch.float32)


def measure_trace_noise(traces, sfreq, wind_end):
    """measure noise (mean absolute deviation)
    at consecutive segments of 1 second

    Args:
        traces: 2D array
    """
    traces = torch.from_numpy(traces)
    winds = np.arange(0, wind_end, 1)
    mads = []
    for wind_i in winds:
        segment = traces[wind_i * sfreq: (wind_i + 1) * sfreq]
        mads.append(torch_mad(segment).cpu().detach().numpy())
    return mads


def measure_noise_parallel_nv(traces, site):
    """Measure the minimum absolute deviation of a single trace
    over contiguous segments of one second
    "Site" is only used implicitly to pass each row of the
    traces_vivo matrix (the original timepoints x site matrix was transposed
    such that sites are in rows) to this function.

    Args:
        traces_vivo (np.array): a 1D trace array of site x timepoints
        site (int): the row used to implicitely extract that row from traces_vivo

    Returns:
        _type_: _description_
    """
    # get number of maximum one-sec periods
    wind_end = int(np.floor(len(traces)/SFREQ_V))
    
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces, SFREQ_V, wind_end))


def measure_noise_parallel_ns(traces, site):
    """Measure the minimum absolute deviation of a single trace
    over contiguous segments of one second
    "Site" is only used implicitly to pass each row of the
    traces_vivo matrix (the original timepoints x site matrix was transposed
    such that sites are in rows) to this function.

    Args:
        traces_silico (np.array): a 1D trace array of site x timepoints
        site (int): the row used to implicitely extract that row from traces_vivo

    Returns:
        _type_: _description_
    """
    # get number of maximum one-sec periods
    wind_end = int(np.floor(len(traces)/SFREQ_S))
    
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces, SFREQ_S, wind_end))


def measure_noise_parallel_ne(traces, site):
    """Measure the minimum absolute deviation of a single trace
    Args:
        traces_silico (np.array): a 1D trace array of site x timepoints
        site (int): the row used to implicitely extract that row from traces_vivo

    Returns:
        _type_: _description_
    """
    # get number of maximum one-sec periods
    wind_end = int(np.floor(len(traces)/SFREQ_E))
    
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces, SFREQ_E, wind_end))


def measure_noise_parallel_nb(traces, site):
    """Measure the minimum absolute deviation of a single trace
    over contiguous segments of one second
    "Site" is only used implicitly to pass each row of the
    traces_vivo matrix (the original timepoints x site matrix was transposed
    such that sites are in rows) to this function.

    Args:
        traces_silico (np.array): a 1D trace array of site x timepoints
        site (int): the row used to implicitely extract that row from traces_vivo

    Returns:
        _type_: _description_
    """
    # get number of maximum one-sec periods
    wind_end = int(np.floor(len(traces)/SFREQ_B))

    # each site row of the array is passed to a worker    
    return min(measure_trace_noise(traces, SFREQ_B, wind_end))


def measure_vivo_trace_noise_parallel_hv(traces, site):
    """Measure the minimum absolute deviation of a single trace
    over contiguous segments of one second
    "Site" is only used implicitly to pass each row of the
    traces matrix (the original timepoints x site matrix was transposed
    such that sites are in rows) to this function.

    Args:
        traces_vivo (np.array): a 1D trace array of site x timepoints
        site (int): the row used to implicitely extract that row from traces_vivo

    Returns:
        _type_: _description_
    """
    # get number of maximum one-sec periods
    wind_end = int(np.floor(len(traces)/SF_HV))
    
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces, SF_HV, wind_end))


def measure_silico_trace_noise_parallel_hs(traces, site):
    """Measure the minimum absolute deviation of a single trace
    over contiguous segments of one second
    "Site" is only used implicitly to pass each row of the
    traces matrix (the original timepoints x site matrix was transposed
    such that sites are in rows) to this function.

    Args:
        traces (np.array): a 1D trace array of site x timepoints
        site (int): the row used to implicitely extract that row from traces_vivo

    Returns:
        _type_: _description_
    """
    # get number of maximum one-sec periods
    wind_end = int(np.floor(len(traces)/SF_HV))
        
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces, SF_HS, wind_end))


def sem(data):
    return np.std(data) / np.sqrt(len(data))


def conf_interv95(data):
    return 1.96 * sem(data)


def get_layer_sites(silico_layers, layer: str = "L1"):
    if layer == "L2_3":
        return np.hstack(
            [np.where(silico_layers == "L2")[0], np.where(silico_layers == "L3")[0]]
        )
    else:
        return np.where(silico_layers == layer)[0]
    

# def compute_in_parallel(traces_v: np.array, traces_s: np.array, traces_e: np.array, traces_b: np.array):
#     """Compute noise for each layer (13 mins)"""

#     # VIVO
#     with ProcessPoolExecutor() as executor:
#         n_v = executor.map(
#             measure_noise_parallel_v,
#             traces_v.T,
#             np.arange(0, traces_v.shape[1], 1),
#         )
#     noise_v = list(n_v)

#     # SILICO
#     with ProcessPoolExecutor() as executor:
#         n_s = executor.map(
#             measure_noise_parallel_s,
#             traces_s.T,
#             np.arange(0, traces_s.shape[1], 1),
#         )
#     noise_s = list(n_s)
    
#     # EVOKED
#     with ProcessPoolExecutor() as executor:
#         n_e = executor.map(
#             measure_noise_parallel_e,
#             traces_e.T,
#             np.arange(0, traces_e.shape[1], 1),
#         )
#     noise_e = list(n_e)    

#     # buccino
#     with ProcessPoolExecutor() as executor:
#         n_b = executor.map(
#             measure_noise_parallel_b,
#             traces_b.T,
#             np.arange(0, traces_b.shape[1], 1),
#         )
#     noise_b = list(n_b)
#     return noise_v, noise_s, noise_e, noise_b


def get_in_parallel_single_nv(traces: np.array):
    """Compute noise for each layer for Marques-Smith"""

    with ProcessPoolExecutor() as executor:
        n_v = executor.map(
            measure_noise_parallel_nv,
            traces.T,
            np.arange(0, traces.shape[1], 1),
        )
    noise_v = list(n_v)
    return noise_v


def get_in_parallel_single_ns(traces: np.array):
    """Compute noise for each layer for Biophy. spont. model(13 mins)"""

    with ProcessPoolExecutor() as executor:
        n_s = executor.map(
            measure_noise_parallel_ns,
            traces.T,
            np.arange(0, traces.shape[1], 1),
        )
    noise_s = list(n_s)
    return noise_s


def get_in_parallel_single_ne(traces: np.array):
    """Compute noise for each layer for Bophy. evoked model (13 mins)"""

    with ProcessPoolExecutor() as executor:
        n_e = executor.map(
            measure_noise_parallel_ne,
            traces.T,
            np.arange(0, traces.shape[1], 1),
        )
    noise_e = list(n_e)
    return noise_e


def get_in_parallel_single_nb(traces: np.array):
    """Compute noise for each layer for Synthetic model(13 mins)"""

    with ProcessPoolExecutor() as executor:
        n_b = executor.map(
            measure_noise_parallel_nb,
            traces.T,
            np.arange(0, traces.shape[1], 1),
        )
    noise_b = list(n_b)
    return noise_b


def get_noise_data_hv(traces_v, layers_v):

    # 2. Compute layer-wise noise (13 mins) --------------

    # measure site noise of in vivo traces (parallelized, )
    with ProcessPoolExecutor() as executor:
        noise_by_trace = executor.map(
            measure_vivo_trace_noise_parallel_hv,
            traces_v.T,
            np.arange(0, traces_v.shape[1], 1),
        )
    vivo_noise_by_trace = list(noise_by_trace)

    # FORMAT PLOT DATA ----------
    layers = ["L1", "L2_3", "L4", "L5", "L6"]

    # calculate noise stats by layer
    vivo_noise = []
    for l_i in range(len(layers)):
        vivo_noise.append(np.array(vivo_noise_by_trace)[layers_v == layers[l_i]])

    # build dataset to plot
    # - vivo data
    vivo_data = pd.DataFrame(data=np.array(vivo_noise_by_trace), columns=["noise"])
    vivo_data["layer"] = layers_v
    vivo_data["experiment"] = "vivo"

    # drop sites outside layers
    mask = np.isin(vivo_data["layer"], layers)
    vivo_data = vivo_data[mask]
    vivo_data = vivo_data.sort_values(by=["layer"])
    return vivo_data


def get_noise_data_hs(traces_s, layers_s):

    # 2. Compute layer-wise noise (13 mins) --------------

    # measure site noise of fitted silico traces
    with ProcessPoolExecutor() as executor:
        silico_noise_by_trace = executor.map(
            measure_silico_trace_noise_parallel_hs,
            traces_s.T,
            np.arange(0, traces_s.shape[1], 1),
        )
    silico_noise_by_trace = list(silico_noise_by_trace)

    # FORMAT PLOT DATA ----------
    layers = ["L1", "L2_3", "L4", "L5", "L6"]

    # calculate noise stats by layer
    sili_noise = []
    for l_i in range(len(layers)):
        sites = get_layer_sites(layers_s, layer=layers[l_i])
        sili_noise.append(np.array(silico_noise_by_trace)[sites])

    # build dataset to plot
    # - silico data
    sili_data = pd.DataFrame(data=np.array(silico_noise_by_trace), columns=["noise"])

    # - group l2 and l3
    new_silico_layers = copy.copy(layers_s)
    new_silico_layers = np.select(
        [new_silico_layers == "L2", new_silico_layers == "L3"],
        ["L2_3", "L2_3"],
        new_silico_layers,
    )
    sili_data["layer"] = new_silico_layers
    sili_data["experiment"] = "silico"

    # drop sites outside layers
    mask = np.isin(sili_data["layer"], layers)
    sili_data = sili_data[mask]
    sili_data = sili_data.sort_values(by=["layer"])
    return sili_data


def count_sites(df, exp, layer):
    return len(df[(df["experiment"] == exp) & (df["layer"] == layer)])


def get_noise(df, exp, layer):
    return df[(df["experiment"] == exp) & (df["layer"] == layer)]["noise"].values


def get_mwu(df, exp1, exp2, layer):
    z, p = mannwhitneyu(
        get_noise(df, exp1, layer),
        get_noise(df, exp2, layer),
        method="exact",
    )
    print(
        f"""1 vs. 2, z={z}, p={p}, N_1={count_sites(df, exp1, layer)}, N_2={count_sites(df, exp2, layer)}"""
    )
    

def get_kk(df, exp):
    """kruskall wallis test
    """
    h, p = kruskal(
        get_noise(df, exp, "L1"),
        get_noise(df, exp, "L2/3"),
        get_noise(df, exp, "L4"),
        get_noise(df, exp, "L5"),
        get_noise(df, exp, "L6"),
    )
    print(f"H={h}, p={p}")
    print(f"""N_L1 = {count_sites(df, exp, "L1")} sites""")
    print(f"""N_L23 = {count_sites(df, exp, "L2/3")} sites""")
    print(f"""N_L4 = {count_sites(df, exp, "L4")} sites""")
    print(f"""N_L5 = {count_sites(df, exp, "L5")} sites""")
    print(f"""N_L6 = {count_sites(df, exp, "L6")} sites""")


def get_posthoc_dunn_holm_sidak(plot_data, exp):
    """posthoc test after kruskall wallis with Dunn and holm_sidak
    multiple comparison correction of p-values

    Args:
        plot_data (_type_): _description_
        exp (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = [
        get_noise(plot_data, exp, "L1"),
        get_noise(plot_data, exp, "L2/3"),
        get_noise(plot_data, exp, "L4"),
        get_noise(plot_data, exp, "L5"),
        get_noise(plot_data, exp, "L6"),
    ]
    # holm sidak method has more power than Bonferroni which is more conservative
    # Non-significance can indicate subtle differences, power issues, samll sample size,
    # or the balancing be due to how the Holm-Sidak correction controls Type I errors
    # while retaining power.
    # we can still look at the p-values to identify trends.
    df = sp.posthoc_dunn(data, p_adjust="holm-sidak")
    df.columns = ["L1", "L2/3", "L4", "L5", "L6"]
    df.index = ["L1", "L2/3", "L4", "L5", "L6"]
    return df