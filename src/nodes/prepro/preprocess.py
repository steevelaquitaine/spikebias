"""Pipeline that preprocesses a campaign traces

Usage:

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run
    python3.9 app.py simulation --pipeline preprocess --conf 2023_01_13

Returns:
    _type_: _description_
"""

import logging
import logging.config
import shutil
from sys import argv
from time import time
import numpy as np
import spikeinterface.full as si
import yaml
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import pandas as pd

# custom package
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro.filtering import wavelet_filter
from src.nodes.utils import get_config
from src.nodes.validation.layer import getAtlasInfo, loadAtlasInfo


# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# Exporting to Extractor  -------------------------------------------

def save_raw_rec_extractor(data_conf: dict, param_conf: dict, job_dict: dict):
    """Write raw simulated recording as a spikeinterface
    RecordingExtractor

    Args:
        data_conf (_type_): _description_

    Returns:
        _type_: _description_
    """
    # track time
    t0 = time()
    logger.info("Starting ...")
    
    # set traces read write paths
    READ_PATH = data_conf["recording"]["input"]
    WRITE_PATH = data_conf["recording"]["output"]

    # read and cast raw trace as array (1 min/h recording)
    trace = pd.read_pickle(READ_PATH)
    trace = np.array(trace)
    
    # cast trace as a SpikeInterface Recording object
    Recording = se.NumpyRecording(
        traces_list=[trace],
        sampling_frequency=param_conf["sampling_freq"],
    )
    
    # write (parallel processing works for 10 min recordings, else use 1 node for 1h recording otherwise
    # you get "out of memory error: "slurmstepd: error: Detected 50 oom-kill event(s). 
    # Some of your processes may have been killed by the cgroup out-of-memory handler."")
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Recording.save(folder=WRITE_PATH, format="binary", **job_dict)

    # log
    logger.info("Probe wiring done in  %s secs", round(time() - t0, 1))    
    
# add noise and gain  -------------------------------------------
    
def fit_and_cast_as_extractor(data_conf, offset, scale_and_add_noise):
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 

    takes 54 min
    note: RecordingExtractor is not dumpable and can't be processed in parallel
    """
    
    # track time
    t0 = time()
    logger.info("Starting ...")

    # cast (30 secs)
    Recording = recording.run(data_conf, offset=offset, scale_and_add_noise=scale_and_add_noise)
    logger.info(f"Done in {np.round(time()-t0,2)} secs")
    return Recording


def fit_and_cast_as_extractor_for_nwb(data_conf: dict, param_conf: dict,
                                      offset: bool, scale_and_add_noise):
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 

    takes 54 min
    note: RecordingExtractor is not dumpable and can't be processed in parallel
    """
    # track time
    t0 = time()
    logger.info("Starting ...")

    # cast (30 secs)
    Recording = recording.run_from_nwb(data_conf, param_conf, offset=offset,
                                       scale_and_add_noise=scale_and_add_noise)
    logger.info(f"Done in {np.round(time()-t0,2)} secs")
    return Recording


def fit_and_cast_as_extractor_dense_probe(data_conf: dict, offset: bool,
                                          scale_and_add_noise):
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 

    takes 54 min
    note: RecordingExtractor is not dumpable and can't be processed in parallel
    """
    # track time
    t0 = time()
    logger.info("Starting ...")

    # cast (30 secs)
    Recording = recording.run(data_conf, offset=offset, 
                              scale_and_add_noise=scale_and_add_noise)

    # remove 129th "test" channel (actually 128 because starts at 0)
    if len(Recording.channel_ids) == 129:
        Recording = Recording.remove_channels([128])
    logger.info(f"Done in {np.round(time()-t0,2)} secs")
    return Recording


def fit_and_cast_as_extractor_dense_probe_for_nwb(data_conf: dict, param_conf: dict, 
                                                  offset: bool, scale_and_add_noise):
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 

    takes 54 min
    note: RecordingExtractor is not dumpable and can't be processed in parallel
    """
    # track time
    t0 = time()
    logger.info("Starting ...")

    # cast (30 secs)
    Recording = recording.run_from_nwb(data_conf, param_conf, offset=offset, 
                                       scale_and_add_noise=scale_and_add_noise)

    # remove 129th "test" channel (actually 128 because starts at 0)
    if len(Recording.channel_ids) == 129:
        Recording = Recording.remove_channels([128])
    logger.info(f"Done in {np.round(time()-t0,2)} secs")
    return Recording

# wiring and metadata -------------------------------------------

def label_layers(data_conf, Recording, blueconfig, n_sites: int, load_atlas_metadata=True):
    """record electrode site layer property in RecordingExtractor
    
    Args:
        blueconfig (None): is always None
    """

    # load probe.wired trace
    probe = Recording.get_probe()

    # get site layers and curare
    if load_atlas_metadata:
        _, site_layers = loadAtlasInfo(data_conf)
    else:
        _, site_layers = getAtlasInfo(data_conf, blueconfig, probe.contact_positions)
    site_layers = np.array(site_layers)
    site_layers[site_layers == "L2"] = "L2_3"
    site_layers[site_layers == "L3"] = "L2_3"

    # sanity check
    assert len(site_layers) == n_sites, """site count does not match horvath's probe'"""

    # add metadata to RecordingExtractor
    Recording.set_property('layers', values=site_layers)
    return Recording


def set_property_layer(data_conf, Recording, blueconfig=None, n_sites=int, load_atlas_metadata=True):
    """Save layer metadata to the Recording Extractor
    """
    return label_layers(data_conf, Recording, blueconfig, n_sites=n_sites,
                        load_atlas_metadata=load_atlas_metadata)


def wire_probe(
        data_conf: dict,
        param_conf: dict, 
        Recording, 
        blueconfig, 
        save_metadata: bool,
        job_dict: dict, 
        n_sites: int, 
        load_atlas_metadata=True, 
        load_filtered_cells_metadata=True
        ):
    """wire the probe to the recording
    
    Args:
        data_conf (dict):
        param_conf (dict):
        Recording (RecordingExtractor):
        blueconfig (None): DEPRECATED should always be None
        save_metadata (bool)
        job_dict: dict, 
        load_atlas_metadata (Boolean): True: loads existing metadata, else requires the Atlas (download on Zenodo)
        load_filtered_cells_metadata: True: loads existing metadata; can only be true
        
    note: The wired Recording Extractor is written via 
    multiprocessing on 8 CPU cores, with 1G of memory per job 
    (n_jobs=8 and chunk_memory=1G)

    to check the number of physical cpu cores on your machine:
        cat /proc/cpuinfo | grep 'physical id' | sort -u | wc -l
    
    to check the number of logical cpu cores on your machine:
        nproc
    """    
    # track time
    t0 = time()
    logger.info("Starting ...")
    
    # get write path
    WRITE_PATH = data_conf["probe_wiring"]["full"]["output"]
    
    # run and write
    Recording = probe_wiring.run(Recording, data_conf, 
                                 param_conf, load_filtered_cells_metadata)

    # save metadata
    if save_metadata:
        Recording = set_property_layer(data_conf, Recording, blueconfig, 
                                       n_sites=n_sites,
                                       load_atlas_metadata=load_atlas_metadata)

    # write (parallel processing works for 10 min recordings, else use 1 node for 1h recording otherwise
    # you get "out of memory error: "slurmstepd: error: Detected 50 oom-kill event(s). 
    # Some of your processes may have been killed by the cgroup out-of-memory handler."")
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Recording.save(folder=WRITE_PATH, format="binary", **job_dict)
    logger.info(f"Done in {np.round(time()-t0,2)} secs")

# preprocessing -------------------------------------------

def run(dataset_conf: dict, param_conf: dict, filtering:str='butterworth'):
    """preprocess recording traces

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_
        filtering (str): 'butterworth' or 'wavelet'

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """
    if filtering == 'butterworth':
        # default filtering
        return run_butterworth_filtering(dataset_conf, param_conf)
    elif filtering == 'wavelet':
        # wavelet filtering
        return run_wavelet_filtering(dataset_conf, param_conf)
    

def run_noise_20_perc_lower(dataset_conf: dict, param_conf: dict, filtering:str='butterworth'):
    """preprocess recording traces

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_
        filtering (str): 'butterworth' or 'wavelet'

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """
    if filtering == 'butterworth':
        # default filtering
        return run_butterworth_filtering_noise_20_perc_lower(dataset_conf, param_conf)
    elif filtering == 'wavelet':
        # wavelet filtering
        raise NotImplementedError


def run_noise_0uV(dataset_conf: dict, param_conf: dict, filtering:str='butterworth'):
    """preprocess recording traces

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_
        filtering (str): 'butterworth' or 'wavelet'

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """
    if filtering == 'butterworth':
        # default filtering
        return run_butterworth_filtering_noise_0uV(dataset_conf, param_conf)
    elif filtering == 'wavelet':
        # wavelet filtering
        raise NotImplementedError    


def run_noise_0uV_gain_x(Recording, dataset_conf: dict, param_conf: dict, filtering:str='butterworth'):
    """preprocess recording traces

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_
        filtering (str): 'butterworth' or 'wavelet'

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """
    if filtering == 'butterworth':
        return run_butterworth(Recording, dataset_conf, param_conf)
    elif filtering == 'wavelet':
        # wavelet filtering
        raise NotImplementedError


def run_butterworth_filtering(dataset_conf: dict, param_conf: dict):
    """apply fourier filtering with butterworth filters to traces

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]

    # get trace
    trace = si.load_extractor(dataset_conf["probe_wiring"]["full"]["output"])
    logger.info(f"""Path of the preprocessed wired probe: {dataset_conf["probe_wiring"]["full"]["output"]}""")
    
    # bandpass
    bandpassed = si.bandpass_filter(
        trace, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )

    # set common reference
    referenced = si.common_reference(
        bandpassed, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd adjusted 10% less and noise ftd in {np.round(time()-t0,2)} secs")
    return referenced


def run_butterworth_filtering_noise_20_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["input_noise_20_perc_lower"]
    
    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 20% reduced noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_50_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_gain_fitd_noise_50_perc_lower"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 50% reduced noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_75_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_gain_fitd_noise_75_perc_lower"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 75% reduced noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_80_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_gain_fitd_noise_80_perc_lower"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 80% reduced noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_90_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_gain_fitd_noise_90_perc_lower"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 90% reduced noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_95_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_gain_fitd_noise_95_perc_lower"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 95% reduced noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_99_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_gain_fitd_noise_99_perc_lower"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 99% reduced noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_ftd_gain_ftd(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_noise_fitd_gain_fitd_int16"]

    # get trace
    Wired = si.load_extractor(PREP_PATH)
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces noise ftd gain ftd in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_none_gain_none(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_noise_none_gain_none_int16"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with noise none gain none in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_not_ftd(Wired, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]

    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on not fitted traces {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_ftd_gain_ftd_adj(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_noise_fitd_gain_fitd_adj_int16"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd adjusted and noise ftd in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_ftd_gain_ftd_adj10perc(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_noise_fitd_gain_fitd_adj10perc_int16"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd adjusted and noise ftd in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_ftd_gain_ftd_adj20perc(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_noise_fitd_gain_fitd_adj20perc_int16"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd adjusted and noise ftd in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_ftd_gain_ftd_adj05perc(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_noise_fitd_gain_fitd_adj05perc_int16"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd adjusted and noise ftd in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_ftd_gain_ftd_adj05perc_less(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_noise_fitd_gain_fitd_adj05perc_less_int16"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd adjusted 5% less and noise ftd in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_ftd_gain_ftd_adj10perc_less(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    WIRED_PATH = data_conf["probe_wiring"]["full"]["output"]

    # get trace
    Wired = si.load_extractor(
        WIRED_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {WIRED_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd adjusted 10% less and noise ftd in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_ftd_gain_ftd_adj20perc_less(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_noise_fitd_gain_fitd_adj20perc_less_int16"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd adjusted 20% less and noise ftd in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_40_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_gain_fitd_noise_40_perc_lower"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 40% reduced noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_60_perc_lower(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    PREP_PATH = data_conf["probe_wiring"]["40m"]["input_gain_fitd_noise_60_perc_lower"]

    # get trace
    Wired = si.load_extractor(
        PREP_PATH
    )
    logger.info(f"""Path of the preprocessed wired probe: {PREP_PATH}""")
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with 60% lower noise in {np.round(time()-t0,2)} secs")
    return Wired


def run_butterworth_filtering_noise_0uV(data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]

    # get trace
    Wired = si.load_extractor(
        data_conf["probe_wiring"]["input_noise_0uV"]
    )
    logger.info(f"""Path of the preprocessed wired probe: {data_conf["probe_wiring"]["input_noise_0uV"]}""")

    # bandpass
    bandpassed = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )

    # set common reference
    referenced = si.common_reference(
        bandpassed, reference="global", operator="median"
    )
    logger.info(f"Run Butterworth filtering on traces without external noise in {np.round(time()-t0,2)} secs")
    return referenced


def run_butterworth(Wired, data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]

    # bandpass
    bandpassed = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )

    # set common reference
    referenced = si.common_reference(
        bandpassed, reference="global", operator="median"
    )
    logger.info(f"Run Butterworth filtering on traces in {np.round(time()-t0,2)} secs")
    return referenced


def run_butterworth_filtering_buccino(Wired, data_conf, param_conf):
    """apply fourier filtering with butterworth filters to traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """

    t0 = time()
    
    # get write path    
    FREQ_MIN = param_conf["run"]["preprocessing"]["min_filter_freq"]
    FREQ_MAX = param_conf["run"]["preprocessing"]["max_filter_freq"]
    
    # compress to int16 (like the Kilosort sorters)
    Wired = spre.astype(Wired, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")    
    
    # band-pass filter
    Wired = si.bandpass_filter(
        Wired, freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")    

    # apply common reference
    Wired = si.common_reference(
        Wired, reference="global", operator="median"
    )
    logger.info(f"Applied common referencing in {np.round(time()-t0,2)} secs")    
    logger.info(f"Done running Butterworth filtering on traces with gain fitd in {np.round(time()-t0,2)} secs")
    return Wired


def run_wavelet_filtering(dataset_conf: dict, param_conf: dict):
    """apply wavelet filtering to traces

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        SpikeInterface RecordingExtractor: referenced and filtered traces
    """
    # get parameters
    prm = param_conf['wavelet_filtering']

    # get RecordingExtractor
    Wired = probe_wiring.load(dataset_conf)

    # apply wavelet filtering
    bandpassed = wavelet_filter(Wired,
                                duration_s=prm['duration_s'],
                                wavelet=prm['wavelet'],
                                method=prm['method'],
                                nlevel=prm['nlevel'])
    # set common reference
    referenced = si.common_reference(
        bandpassed, reference="global", operator="median"
    )
    return referenced


def preprocess_recording_dense_probe(data_conf, param_conf, job_dict):
    """preprocess recording

    takes 15 min (w/ multiprocessing, else 32 mins)
    """
    # track time
    t0 = time()
    logger.info("Starting 'preprocess_recording'")
    
    # write path
    WRITE_PATH = data_conf["preprocessing"]["full"]["output"]["trace_file_path"]
    
    # preprocess, write
    Preprocessed = run(data_conf, param_conf)
    
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time()-t0,2)} secs")
    

def preprocess_recording_npx_probe(data_conf, param_conf, job_dict: dict, filtering='butterworth'):
    """preprocess recording and write

    Args:   
        job_dict
        filtering: 'butterworth' or 'wavelet'

    takes 15 min (vs. 32 min w/o multiprocessing)
    """
    # takes 32 min
    t0 = time()
    logger.info("Starting 'preprocessing'")
    
    # write path
    WRITE_PATH = data_conf["preprocessing"]["full"]["output"]["trace_file_path"]
    
    # preprocess
    Preprocessed = run_butterworth_filtering_noise_ftd_gain_ftd_adj10perc_less(data_conf,
                                  param_conf)
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time()-t0,2)} secs")
    
# io ---------------------------------------
    
def write(preprocessed, data_conf:dict, job_dict:dict):
    """write preprocessed recording (with multiprocessing)
    
    Args:

    Returns:

    Note: 
        takes 15 min (vs. 32 min w/o multiprocessing)
    
        The max number of jobs is limited by the number of CPU cores
        Our nodes have 8 cores.
        n_jobs=8 and total_memory=2G sped up writing by a factor a 2X
        no improvement was observed for larger memory allocation
    """
    WRITE_PATH = data_conf["preprocessing"]["output"]["trace_file_path"]
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    

def load(data_conf: dict):
    """Load preprocessed recording from config

    Args:
        data_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    return si.load_extractor(
        data_conf["preprocessing"]["output"]["trace_file_path"]
    )

# entry point -----------------------------

if __name__ == "__main__":

    # start timer
    t0 = time()

    # parse run parameters
    conf_date = argv[1]

    # get config
    data_conf, param_conf = get_config(conf_date).values()

    # run preprocessing
    output = run(data_conf, param_conf)
    logger.info("Preprocessing done in  %s secs", round(time() - t0, 1))
