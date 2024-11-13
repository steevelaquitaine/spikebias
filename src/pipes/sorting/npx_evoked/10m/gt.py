
"""postprocess ground truth for the biophysical simulation of neuropixels
in a evoked regime, extracting waveforms into 
a spikeinterface WaveformExtractor object saved in a
study folder.

author: steeve.laquitaine@epfl.ch



"""
import logging
import logging.config
import yaml
import spikeinterface as si
from time import time 
from spikeinterface import extract_waveforms
import resource

# custom package
from src.nodes.utils import get_config
from src.nodes.postpro import cell_type

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def postprocess_10m(cfg, duration_sec=600, create_full_study=False):
    """Record metadata to Ground truth's full SortingExtractor 
    and create WaveformExtractor for 10 minutes of ground truth
    saved in a study folder

    Args:
        cfg (_type_): _description_
        sorter (_type_): name of spike sorter from sorter_dict dictionary (e.g., kilosort2)
        sorter_params (dict): _description_
        duration_sec (int): default 600 sec
        - duration of choosen period of recording in seconds
        copy_binary_recording (bool): default False: copy recording as int16 binary or not
    """
    # track time
    t0 = time()
    
    # get paths
    GT_FULL_PATH = cfg["ground_truth"]["full"]["output"]
    GT_10m_PATH = cfg["ground_truth"]["10m"]["output"]
    STUDY_FULL_PATH = cfg["postprocessing"]["waveform"]["ground_truth"]["study"]
    STUDY_10M_PATH = cfg["postprocessing"]["waveform"]["ground_truth"]["10m"]["study"]
    PREP_PATH = cfg["preprocessing"]["full"]["output"]["trace_file_path"]
    BLUE_CFG = cfg["dataeng"]["blueconfig"]
    
    # load recording
    Prep = si.load_extractor(PREP_PATH)
    
    # select SortingExtractor period
    logger.info(f"Saving SortingExtractor for a duration of {duration_sec/60} minutes...")
    SortingTrue = si.load_extractor(GT_FULL_PATH)
    
    # saves true units' properties metadata
    SortingTrue = cell_type.label_true_cell_properties(SortingTrue, BLUE_CFG, GT_FULL_PATH, save=True)

    # save study for the full SortingExtractor
    if create_full_study:
        create_study(SortingTrue, Prep, STUDY_FULL_PATH)

    # save selected 10 minutes of ground trth
    SortingTrue = SortingTrue.frame_slice(
        start_frame=0, end_frame=duration_sec * SortingTrue.sampling_frequency
    )
    SortingTrue.save(folder=GT_10m_PATH, verbose=True, overwrite=True, n_jobs=-1, chunk_duration="1s", progress_bar=True)
    logger.info(f"Completed in %s", round(time() - t0, 1))
    
    # save study for 10 minute of GT SortingExtractor    
    create_study(SortingTrue, Prep, STUDY_10M_PATH)


def create_study(Sorting, Prep, study_path: str):
    """Saving GroundTruth WaveformExtractor in a
    study folder

    Args:
        Sorting (SortingExtractor): ground truth
        Prep (RecordingExtractor): preprocessed trace
        study_path (str): study path in which waveform extractor 
        - is saved
    """
    
    # report
    logger.info(f"Creating study folder with WaveformExtractor...")
    
    # setup parallel computing
    job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)    
    
    # track time
    t0 = time()
    
    # check system's opened file limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.info("Opened file limits:")
    logger.info(f"- soft: {soft}")
    logger.info(f"- hard (max): {hard}")
    
    # max out limit for extract_waveforms
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    logger.info(f"Max out limit")
    
    # extract waveforms
    # to load use: we = si.WaveformExtractor.load_from_folder(STUDY_ns)
    # dense waveforms are extracted first using a small number of spikes
    We = extract_waveforms(
        Prep,
        Sorting,
        study_path,
        sparse=True, # False
        ms_before=3.0,
        ms_after=3.0,
        max_spikes_per_unit=500,
        unit_batch_size=200,
        overwrite=True,
        seed=0,
        **job_kwargs
    )
    logger.info(f"Completed in %s", round(time() - t0, 1))


# SETUP CONFIG
exp = "silico_neuropixels"
run = "stimulus"
cfg, _ = get_config(exp, run).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# sort and postprocess
postprocess_10m(cfg, duration_sec=600, create_full_study=True)