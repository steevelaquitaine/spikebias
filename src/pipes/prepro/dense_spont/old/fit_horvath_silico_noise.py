"""pipeline to fit average in silico horvath probe 1's site noise to average in vivo
    noise for a specified layer "L1", "L2_3". Probe 1 does not contain sites in "L4", "L5" or "L6"
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

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config


# SETUP PARAMETERS
RND_SEED = 0                # random seed
SFREQ_VIVO = 20000          # sampling frequency
SFREQ_SILICO = 20000        # sampling frequency
WIND_END = 3700             # last segment to calculate mad


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


def myfun_layer(missing_noise, *args):
    # get args
    # - get in silico target site traces
    # - get in vivo layer noise
    traces = args[0]
    vivo_noise = args[1]
    fit_history_path = args[2]
    ntimepoints = traces.shape[1]
    nsites = traces.shape[0]
    
    np.random.seed(RND_SEED)
    
    # try with this amount of missing uncorrelated noise in silico
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


def callback(Xi):
    """callback function to print iterations_summary

    Args:
        Xi (_type_): missing noise (mad) parameter to solve
    """
    global Nfeval  # function evaluation
    print(Xi)
    print("{0:4d}   {1: 3.6f}".format(Nfeval, Xi[0]))
    Nfeval += 1


def get_layer_sites(silico_layers, layer:str="L1"):
    if layer == "L2_3":
        return np.hstack(
            [np.where(silico_layers == "L2")[0], np.where(silico_layers == "L3")[0]]
        )
    else:
        return np.where(silico_layers == layer)[0]

    
def fit_noise_by_layer(
    traces_vivo, fitted_traces_silico, vivo_layers, silico_layers, layer:str, fit_history_path:str
):
    """_summary_

    Args:
        traces_vivo (_type_): _description_
        fitted_traces_silico (_type_): _description_
        vivo_layers (_type_): _description_
        silico_layers (_type_): _description_
        layer (str): "L1", "L2_3", ...
        fit_history_path (str): _description_

    Returns:
        _type_: _description_
    """
    # set seed for reproducibility
    np.random.seed(RND_SEED)

    ## VIVO NOISE -------------

    # measure the average layer sites noise    
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
    print("measured noise (in vivo):", vivo_layer_noise)

    ## RAW SILICO NOISE -------------

    # measure the average layer sites noise
    # layer 2 and 3 are separated, merge into L2_3
    layer_sites = get_layer_sites(silico_layers, layer)
    silico_traces = fitted_traces_silico[:, layer_sites].T
    nsites = silico_traces.shape[0]
    with ProcessPoolExecutor() as executor:
        traces_noise = executor.map(
            measure_silico_trace_noise_parallel,
            silico_traces,
            np.arange(0, nsites, 1),
        )
    silico_traces_noise = list(traces_noise)
    silico_layer_noise = np.mean(np.array(silico_traces_noise))
    print("measured noise (in silico):", silico_layer_noise)

    ## FIT --------

    x0 = 5
    MAX_ITER = 15
    TOL = 0.01

    print(" iter     x ")
    results = minimize(
        myfun_layer,
        x0,
        args=(silico_traces, vivo_layer_noise, fit_history_path),
        method="Nelder-Mead",
        tol=TOL,
        callback=callback,
        options={"maxiter": MAX_ITER, "disp": True},
    )
    print(results)
    missing_noise = results.x[0]
    print("fit done")
    return missing_noise


def run(probe: str="probe_1", layer: str="L1"):
    """fit average in silico sites' noise to average in vivo
    noise for the specified layer

    Args:
        layer (str, optional): _description_. Defaults to "L1".
        - "L1", "L2_3", "L4", "L5", "L6"
    """

    # get config
    # HORVATH (vivo)
    data_conf_h_vivo, _ = get_config(
        "vivo_horvath", probe
    ).values()
    PREP_PATH_h_vivo = data_conf_h_vivo["preprocessing"]["output"]["trace_file_path"]

    # HORVATH (silico)
    data_conf_h_silico, _ = get_config(
        "silico_horvath",  "concatenated/"+probe
    ).values()
    PREP_PATH_h_silico = data_conf_h_silico["preprocessing"]["output"]["trace_file_path"]
    MISSING_NOISE_PATH = data_conf_h_silico["preprocessing"]["fitting"]["missing_noise_path"]
    FIT_HISTORY_PATH = MISSING_NOISE_PATH + layer + "_" + "fminsearch_history.txt"

    # track time
    t0 = time.time()
    logging.info("starting pipeline")

    # - get max in vivo trace
    PreRecording_h_vivo = si.load_extractor(PREP_PATH_h_vivo)
    traces_vivo = PreRecording_h_vivo.get_traces()

    # - get max in silico trace
    PreRecording_h_silico = si.load_extractor(PREP_PATH_h_silico)
    traces_silico = PreRecording_h_silico.get_traces()

    # get site layers
    silico_layers = PreRecording_h_silico.get_property("layers")
    vivo_layers = PreRecording_h_vivo.get_property("layers")

    logging.info("loaded recordings - done")

    # takes 1 min

    # get max amplitudes
    max_traces_vivo = traces_vivo.max()
    max_traces_silico = traces_silico.max()

    # fit in silico's max amplitude to in vivo's (calculate gain ratio)
    scale = max_traces_vivo / max_traces_silico
    scaled_silico = traces_silico * scale

    print("max amplitude (vivo, uV):", max_traces_vivo)
    print("max amplitude (silico, uV):", max_traces_silico)
    print("vivo/silico gain ratio:", scale)

    logging.info("traces scaling - done")

    # takes 15 min per layer
    missing_noise = fit_noise_by_layer(
        traces_vivo, scaled_silico, vivo_layers, silico_layers, layer, FIT_HISTORY_PATH
    )

    logging.info("noise fitting - done")
    
    # store layer sites
    layer_sites = get_layer_sites(silico_layers, layer)

    # read fit history
    with open(FIT_HISTORY_PATH, "r") as f:
        fit_history = f.read()

    # save fitted missing noise
    fit_results = {
        "probe": probe,
        "scale": scale,
        "missing_noise_rms": missing_noise,
        "layer_sites_ix": layer_sites.tolist(),
        "layer": layer,
        "fit_history": fit_history,
        "seed": RND_SEED
        }
    np.save(MISSING_NOISE_PATH + layer + ".npy", fit_results)

    logging.info(f"saving fit results in {MISSING_NOISE_PATH + layer} - done")
    logging.info(f"pipeline done in {np.round(time.time() - t0,2)} secs")