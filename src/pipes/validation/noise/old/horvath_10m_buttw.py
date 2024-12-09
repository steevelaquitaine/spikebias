# import libs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import spikeinterface as si
from concurrent.futures import ProcessPoolExecutor
import copy
import spikeinterface.extractors as se
import seaborn as sns

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.dataeng.silico import recording, probe_wiring

# SETUP PARAMETERS
RND_SEED = 0                # random seed
SFREQ_VIVO = 20000          # sampling frequency
SFREQ_SILICO = 20000        # sampling frequency
WIND_END = 3700             # last segment to calculate mad

# vivo
data_conf_hv, _ = get_config("vivo_horvath", "probe_1").values() 
RAW_PATH_hv = data_conf_hv["raw"]
PREP_PATH_hv = data_conf_hv["preprocessing"]["output"]["trace_file_path"]

# silico
# probe 1
data_conf_hs_p1, param_conf_hs_p1 = get_config("dense_spont", "probe_1").values()
RAW_PATH_hs_p1 = data_conf_hs_p1["dataeng"]["campaign"]["output"]["trace_file_path"]
PREP_PATH_hs_p1 = data_conf_hs_p1["preprocessing"]["output"]["trace_file_path"]

# probe 2
data_conf_hs_p2, param_conf_hs_p2 = get_config("dense_spont", "probe_2").values()
RAW_PATH_hs_p2 = data_conf_hs_p2["dataeng"]["campaign"]["output"]["trace_file_path"]
PREP_PATH_hs_p2 = data_conf_hs_p2["preprocessing"]["output"]["trace_file_path"]


# FIGURE SETTINGS
FIG_SIZE = (4, 4)
COLOR_VIVO = np.array([153, 153, 153]) / 255
COLOR_SILI = np.array([228, 26, 28]) / 255
COLOR_BUCCI = np.array([55, 126, 184]) / 255
# axes
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6  # 5-7 with Nature neuroscience as reference
plt.rcParams["lines.linewidth"] = 0.3
plt.rcParams["axes.linewidth"] = 0.3 #1
plt.rcParams["axes.spines.top"] = False
plt.rcParams["xtick.major.width"] = 0.3 #0.8 #* 1.3
plt.rcParams["xtick.minor.width"] = 0.3 #0.8 #* 1.3
plt.rcParams["ytick.major.width"] = 0.3 #0.8 #* 1.3
plt.rcParams["ytick.minor.width"] = 0.3 #0.8 #* 1.3
plt.rcParams["xtick.major.size"] = 3.5 * 1.1
plt.rcParams["xtick.minor.size"] = 2 * 1.1
plt.rcParams["ytick.major.size"] = 3.5 * 1.1
plt.rcParams["ytick.minor.size"] = 2 * 1.1
# legend
savefig_cfg = {"transparent":True, "dpi":300}
legend_cfg = {"frameon": False, "handletextpad": 0.1}
tight_layout_cfg = {"pad": 0.5}
LG_FRAMEON = False              # no legend frame


def measure_trace_noise(traces, sfreq, wind_end):
    """measure noise (mean absolute deviation)
    at consecutive segments of 1 second

    Args:
        traces: 2D array
    """
    winds = np.arange(0, wind_end, 1)
    mads = []
    for wind_i in winds:
        segment = traces[wind_i * sfreq : (wind_i + 1) * sfreq]
        mads.append(pd.DataFrame(segment).mad().values[0])
    return mads


def measure_vivo_trace_noise_parallel(traces_vivo, site):
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
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces_vivo, SFREQ_VIVO, WIND_END))


def measure_silico_trace_noise_parallel(traces_silico, site):
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
    # each site row of the array is passed to a worker
    return min(measure_trace_noise(traces_silico, SFREQ_SILICO, WIND_END))


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
    

def get_noise_data_v(traces_v, layers_v):

    # 2. Compute layer-wise noise (13 mins) --------------

    # measure site noise of in vivo traces (parallelized, )
    with ProcessPoolExecutor() as executor:
        noise_by_trace = executor.map(
            measure_vivo_trace_noise_parallel,
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


def get_noise_data_s(traces_s, layers_s):

    # 2. Compute layer-wise noise (13 mins) --------------

    # measure site noise of fitted silico traces
    with ProcessPoolExecutor() as executor:
        silico_noise_by_trace = executor.map(
            measure_silico_trace_noise_parallel,
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
    new_silico_layers[new_silico_layers == "L2"] = "L2_3"
    new_silico_layers[new_silico_layers == "L3"] = "L2_3"
    sili_data["layer"] = new_silico_layers
    sili_data["experiment"] = "silico"

    # drop sites outside layers
    mask = np.isin(sili_data["layer"], layers)
    sili_data = sili_data[mask]
    sili_data = sili_data.sort_values(by=["layer"])
    return sili_data

# load recordings
RecV = si.load_extractor(PREP_PATH_hv)
RecS1 = si.load_extractor(PREP_PATH_hs_p1)
RecS2 = si.load_extractor(PREP_PATH_hs_p2)

# load traces
traces_v = RecV.get_traces()
traces_s1 = RecS1.get_traces()
traces_s2 = RecS2.get_traces()

# get site layers
layers_v = RecV.get_property("layers")
layers_s1 = RecS1.get_property("layers")
layers_s2 = RecS2.get_property("layers")

# (11m)horvath
plot_data_v = get_noise_data_v(traces_v, layers_v)

# biophy. probe 1 (L1, L2/3)
plot_data_s1 = get_noise_data_s(traces_s1, layers_s1)

# biophy. probe 2 (L2/3, L4, L5, L6)
plot_data_s2 = get_noise_data_s(traces_s2, layers_s2)