"""pipeline to fit average in silico sites' noise to average in vivo
    noise for a specified layer "L1", "L2_3", "L4", "L5" or "L6"
    
    We apply gain and add the missing noise to the raw silico traces, then
    preprocess the traces and fit their noise to in vivo noise
    
    The gain is calculated from the ratios of the traces absolute amplitudes

author: steeve.laquitaine@epfl.ch
  date: 5.1.2023
speed: 20 min

usage:
    sbatch cluster/processing/fitting/marques/fit_marques_silico_l1.sbatch;
    sbatch cluster/processing/fitting/marques/fit_marques_silico_l2_3.sbatch;
    sbatch cluster/processing/fitting/marques/fit_marques_silico_l4.sbatch;
    sbatch cluster/processing/fitting/marques/fit_marques_silico_l5.sbatch;
    sbatch cluster/processing/fitting/marques/fit_marques_silico_l6.sbatch;
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import logging
import logging.config
import numpy as np
import os
import spikeinterface as si
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
import yaml
from contextlib import redirect_stdout
import spikeinterface.extractors as se
from csv import writer
import spikeinterface.full as si_full


# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.dataeng.silico.probe_wiring import wire_silico_marques_probe
from src.nodes.validation.layer import getAtlasInfo

# SETUP PARAMETERS
RND_SEED = 0                # random seed
SFREQ_VIVO = 30000          # sampling frequency
SFREQ_SILICO = 40000        # sampling frequency
WIND_END = 3700             # last segment to calculate mad
PERIOD_SECS = 2             # fit first secs of silico recording (for speed)
SCALE_X_FOR_SPEED = 10e4    # fit parameters
GAIN_ADJUSTM = 2            # manual gain adjustment to re-fit trace amplitude to in vivo after fitting noise, because it changes after preprocessing
X0 = 2.5 * SCALE_X_FOR_SPEED # fit initial parameter
MAX_ITER = 15               # fit parameters
TOL = 0.01                  # fit parameters


# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def measure_noise_at_consecutive_segs(traces, site, sfreq, wind_end):
    """measure noise (mean absolute deviation)
    at consecutive segments of 1 second
    """
    winds = np.arange(0, wind_end, 1)
    mads = []
    for wind_i in winds:
        segment = traces[wind_i * sfreq : (wind_i + 1) * sfreq, site]
        mads.append(pd.DataFrame(segment).mad().values[0])
    return mads


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


Nfeval = 1


def myfun_layer_old(missing_noise, *args):
    """_summary_
    DEPRECATED

    Args:
        missing_noise (np.float): _description_
        args:
        - args[0] (np.array): voltage traces (nsites x ntimepoints)

    Returns:
        _type_: _description_
    """
    # get args
    # - get in silico target site traces
    # - get in vivo layer noise
    traces = args[0]
    vivo_noise = args[1]
    fit_history_path = args[2]
    ntimepoints = traces.shape[1]
    nsites = traces.shape[0]

    # set seed for reproducibility
    np.random.seed(RND_SEED)

    # try with this amount of missing independent noise in silico
    missing_noise_traces = np.random.normal(0, missing_noise, [nsites, ntimepoints])
    with_noise = traces + missing_noise_traces

    # measure site noises in that in silico layer
    with ProcessPoolExecutor() as executor:
        sites_noise = executor.map(
            measure_silico_trace_noise_parallel,
            with_noise,
            np.arange(0, nsites, 1),
        )

    # minimize noise difference between vivo and silico
    silico_noise = np.mean(np.array(list(sites_noise)))
    objfun = abs(vivo_noise - silico_noise)
    
    # save minimization history
    with open(fit_history_path, "a") as f:
        with redirect_stdout(f):
            print("silico noise:", silico_noise)
            print("vivo noise:", vivo_noise)
            print("objfun:", objfun)
            print("----------------")

    # print history in terminal
    with open(fit_history_path, "r") as f:
        contents = f.read()
        print(contents)
    return objfun


def myfun_layer(missing_noise, *args):
    """objective function to minimize 
    to fit in silico noise to in vivo noise

    Args:
        missing_noise (np.float): _description_
        args:
        - args[0] (np.array): voltage traces for all probe sites (nsites=384 x ntimepoints)

    Returns:
        _type_: _description_
    """
    # get args
    # - get in silico target site traces
    # - get in vivo layer noise
    silico_traces = args[0]  # nsites x ntimepoints
    vivo_noise = args[1]
    gain = args[2]
    fit_history_path = args[3]
    data_conf = args[4]
    param_conf = args[5]
    silico_layers = args[6]
    layer = args[7]

    # get preprocessing parameters
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]

    ntimepoints = silico_traces.shape[1] # 384 for marques probe
    nsites = silico_traces.shape[0]
    
    # constrain missing noise to be positive
    if missing_noise < 0:
        return np.inf
    
    # test amount of missing independent noise in silico
    # - set seed for reproducibility
    np.random.seed(RND_SEED)
    missing_noise_traces = np.random.normal(
        0, missing_noise / SCALE_X_FOR_SPEED, [nsites, ntimepoints]
    )
    scaled_and_noised = silico_traces * gain + missing_noise_traces

    # cast, rewire and preprocess
    WithNoise = se.NumpyRecording(
        traces_list=[scaled_and_noised.T],
        sampling_frequency=SFREQ_SILICO,
    )
    WithNoise = wire_silico_marques_probe(data_conf, WithNoise)
    Bandpassed = si_full.bandpass_filter(
        WithNoise, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    Preprocessed = si_full.common_reference(
        Bandpassed, reference="global", operator="median"
    )
    traces_silico = Preprocessed.get_traces()

    # get in silico traces for this layer
    layer_sites = get_layer_sites(silico_layers, layer)
    silico_traces = traces_silico[:, layer_sites]

    # measure site noises in that layer
    nsites = silico_traces.shape[1]
    with ProcessPoolExecutor() as executor:
        sites_noise = executor.map(
            measure_silico_trace_noise_parallel,
            silico_traces.T,
            np.arange(0, nsites, 1),
        )

    # minimize noise difference between vivo and silico
    silico_noise = np.mean(np.array(list(sites_noise)))
    objfun = abs(vivo_noise - silico_noise)

    # save fitting history in text file
    with open(fit_history_path + ".txt", "a") as f:
        with redirect_stdout(f):
            print("silico noise:", silico_noise)
            print("vivo noise:", vivo_noise)
            print("missing noise:", missing_noise / SCALE_X_FOR_SPEED)
            print("gain:", gain)
            print("objfun:", objfun)
            print("----------------")

    # Format fit output in a .csv with headers
    List = [
        silico_noise,
        vivo_noise,
        missing_noise[0] / SCALE_X_FOR_SPEED,
        gain,
        objfun
        ]

    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(fit_history_path + ".csv", "a") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()

    # print history in terminal
    with open(fit_history_path + ".txt", "r") as f:
        contents = f.read()
        print(contents)
    return objfun


def callback(Xi):
    """callback function to print iterations_summary

    Args:
        Xi (_type_): missing noise (mad) parameter to solve
    """
    global Nfeval  # function evaluation
    print(Xi)
    print("{0:4d}   {1: 3.6f}".format(Nfeval, Xi[0]))
    Nfeval += 1


def label_layers(WiredRecording, blueconfig:str):
    """add site layers metadata to Recording Extractor

    Args:
        WiredRecording (_type_): SpikeInterface Recording Extractor
        blueconfig (str): path to blueconfig

    Returns:
        SpikeInterface Recording extractor: _description_
    """
    probe = WiredRecording.get_probe()
    _, site_layers = getAtlasInfo(blueconfig, probe.contact_positions)
    WiredRecording.set_property("layers", values=site_layers)
    return WiredRecording


def get_layer_sites(silico_layers, layer:str="L1"):
    if layer == "L2_3":
        return np.hstack(
            [np.where(silico_layers == "L2")[0], np.where(silico_layers == "L3")[0]]
        )
    else:
        return np.where(silico_layers == layer)[0]

    
def fit_noise_by_layer(
    traces_vivo, silico_traces, vivo_layers, silico_layers, layer: str, fit_history_path:str, gain:float,
    data_conf, param_conf
):
    """fit silico noise to in vivo noise for specified layer

    Args:
        traces_vivo (_type_): _description_
        silico_traces (_type_): _description_
        vivo_layers (_type_): _description_
        silico_layers (_type_): _description_
        layer (str): _description_
        fit_history_path (str): _description_
        gain (float): _description_
        data_conf (_type_): silico data conf
        param_conf (_type_): silico param conf

    Returns:
        float: missing noise in microVolts (root mean square error)
    """
    # measure layer's mean in vivo noise
    vivo_traces = traces_vivo[:, np.where(vivo_layers == layer)[0]].T
    nsites = vivo_traces.shape[0]
    with ProcessPoolExecutor() as executor:
        traces_noise = executor.map(
            measure_vivo_trace_noise_parallel,
            vivo_traces,
            np.arange(0, nsites, 1),
        )
    vivo_traces_noise = list(traces_noise)
    vivo_layer_noise = np.mean(np.array(vivo_traces_noise))

    logging.info("measured layer's average in vivo noise - done")

    # FIT

    # create fit output .csv file
    List = ["silico_noise", "vivo_noise", "missing_noise", "gain", "objfun"]

    with open(fit_history_path + ".csv", "w") as file:
        writer_object = writer(file)
        writer_object.writerow(List)
        file.close()

    # fit
    results = minimize(
        myfun_layer,
        X0,
        args=(
            silico_traces.T,
            vivo_layer_noise,
            gain,
            fit_history_path,
            data_conf,
            param_conf,
            silico_layers,
            layer,
        ),
        method="Nelder-Mead",
        tol=TOL,
        callback=callback,
        options={"maxiter": MAX_ITER, "disp": True},
    )
    print(results)
    missing_noise = results.x[0]
    
    logging.info("silico noise fit - done")
    return missing_noise / SCALE_X_FOR_SPEED


def run(layer: str="L1"):
    """fit average in silico sites' noise to average in vivo
    noise for the specified layer

    Args:
        layer (str, optional): _description_. Defaults to "L1".
        - "L1", "L2_3", "L4", "L5", "L6"
    """

    # track time
    t0 = time.time()
    logging.info(f"start fitting pipeline for layer {layer}.")

    # get config
    # vivo
    data_cfg_v, _ = get_config("vivo_marques", "c26").values()
    PREP_PATH_h_vivo = data_cfg_v["preprocessing"]["output"]["trace_file_path"]
    
    # silico
    data_cfg_s, param_conf_sili = get_config("silico_neuropixels", "2023_10_18").values()
    RAW_PATH_sili = data_cfg_s["dataeng"]["campaign"]["output"]["trace_file_path"]
    BLUECONFIG = data_cfg_s["dataeng"]["blueconfig"]
    MISSING_NOISE_PATH = data_cfg_s["preprocessing"]["fitting"]["missing_noise_path"]
    FIT_HISTORY_PATH = MISSING_NOISE_PATH + layer + "_" + "fit_history"
    
    logging.info("got config - done")

    # get silico recording period to fit
    period = np.arange(0, SFREQ_SILICO * PERIOD_SECS, 1)

    # load preprocessed in vivo and raw in silico Recordings
    # IN SILICO
    # - load raw silico traces
    RawRecording = pd.read_pickle(RAW_PATH_sili)
    raw_traces = RawRecording.iloc[period, :]

    # - cast as Recording extractor
    Recording = se.NumpyRecording(
        traces_list=[np.array(raw_traces)],
        sampling_frequency=SFREQ_SILICO,
    )

    # - wire and add metadata
    WiredSilicoRecording = wire_silico_marques_probe(data_cfg_s, Recording)
    WiredSilicoRecording = label_layers(WiredSilicoRecording, BLUECONFIG)

    # - load preprocessed in vivo
    # IN VIVO
    PreRecording_h_vivo = si.load_extractor(PREP_PATH_h_vivo)
    traces_vivo = PreRecording_h_vivo.get_traces()
    silico_traces = WiredSilicoRecording.get_traces()
    logging.info("loaded vivo and silico recordings - done")

    # calculate gain
    gain = (np.absolute(traces_vivo).max() / np.absolute(silico_traces).max()) * GAIN_ADJUSTM
    logging.info("calculated best fit gain - done")

    # get recording site layer metadata
    silico_layers = WiredSilicoRecording.get_property("layers")
    vivo_layers = PreRecording_h_vivo.get_property("layers")

    # takes 15 min per layer
    missing_noise = fit_noise_by_layer(
        traces_vivo, silico_traces, vivo_layers, silico_layers, layer,
        FIT_HISTORY_PATH, gain, data_cfg_s, param_conf_sili
    )
    logging.info("noise fitting - done")

    # check fit history
    fit_out = pd.read_csv(FIT_HISTORY_PATH + ".csv")
    
    # store layer sites
    layer_sites = get_layer_sites(silico_layers, layer)

    # save fitted results
    fit_results = {
        "gain": gain,
        "gain_adjustm": GAIN_ADJUSTM,
        "missing_noise_rms": missing_noise,
        "layer_sites_ix": layer_sites.tolist(),
        "layer": layer,
        "fit_history": fit_out,
        "seed": RND_SEED,
        }
    np.save(MISSING_NOISE_PATH + layer + ".npy", fit_results)

    logging.info(f"saving fit results in {MISSING_NOISE_PATH + layer} - done")
    logging.info(f"pipeline done in {np.round(time.time() - t0,2)} secs")
