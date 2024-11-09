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
from src.nodes.dataeng.silico import probe_wiring
from src.nodes.prepro.filtering import wavelet_filter
from src.nodes.utils import get_config


# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


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
