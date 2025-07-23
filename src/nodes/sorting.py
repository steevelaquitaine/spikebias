"""Sorting and sorting postprocessing nodes

author: steeve.laquitaine@epfl.ch

Returns:
    _type_: _description_
"""

import os
import numpy as np
import logging
import logging.config
from time import time
import yaml
import spikeinterface.preprocessing as spre
import spikeinterface as si
import yaml
import shutil 
import spikeinterface.sorters as ss
import src.nodes.postpro.metadata as meta

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def remove_the_bad_channels(Wired, bad_channel_ids=None):
    """removes the bad channels
    (outside cortex) from the recording
    to speed up recording and remove
    memory pressure on CUDA

    Args:
        Wired (_type_): _description_
        bad_channel_ids (np.array): None

    Returns:
        _type_: _description_
    """
    # copy RecordingExtractor
    Wired = Wired.clone()
    
    # remove only manually selected bad channels 
    if not bad_channel_ids is None:
        logger.info('Removing manually selected channel_ids')
        pass
    else:
        # remove all the channels outside the cortex
        # label the bad channels with -1 (outside the cortex)
        layers = Wired.get_property("layers")
        bad_channels = np.zeros(len(layers))
        bad_channels[np.isin(layers, "Outside")] = -1
        bad_channel_ids = Wired.channel_ids[np.where(bad_channels == -1)[0]]
        logger.info('Removing all channel_ids outside cortex')
    
    # remove the bad channel ids
    Wired = Wired.remove_channels(bad_channel_ids)
    logger.info(f'New channel count: {Wired.get_num_channels()}')
    return Wired

    
def sort(sorter, wired_path, sorting_path, output_path, params: dict, duration_s: float, 
         copy_binary_recording=False, remove_bad_channels=False, bad_channel_ids=None):
    """Spike sort
    Setup recording (copy as int16 binary, (setup_recording=True),
    take a shorter duration) and sort ...
    or directly sort from an existing recording (setup_recording=True)
    
    Args:
        sorter (_type_): _description_
        wired_path (_type_): _description_
        sorting_path (_type_): _description_
        output_path (_type_): _description_
        params (dict): _description_
        duration_s (float): _description_
        copy_binary_recording (False): save once as binary to avoid 
        ... subsequent copying by Kilosort sorters

    Returns:    
        Sorting (SortingExtractor): contains metadata:
        - "sorting_duration"
    """
    
    # track time
    t0 = time()
    
    # set path of int16 binary recording copy
    REC_BIN_INT16_PATH = os.path.join(os.path.dirname(os.path.dirname(output_path)), sorter + "_rec_bin_int16_copy")
    job_kwargs = dict(n_jobs=-1, chunk_duration="10s", progress_bar=True)
    
    # if not exists, setup (copy) recording as int16 binary
    # save once as binary to avoid subsequent copying by Kilosort sorters
    if copy_binary_recording:

        # choose recording period
        Wired = si.load_extractor(wired_path)
        
        # remove bad channels
        # speeds up sorting, less memory intensive
        if remove_bad_channels:
            logger.info("Removing bad channels...")
            Wired = remove_the_bad_channels(Wired, bad_channel_ids)
            logger.info(f"Done removing bad channels in: %s", round(time() - t0, 1))
            logger.info(f"New number of channels for sorting: {Wired.get_num_channels()}")
        
        # select a shorter period of the recording
        end_frame =int(duration_s * Wired.get_sampling_frequency())
        Wired = Wired.frame_slice(start_frame=0, end_frame=end_frame)
        logger.info(f"Selected first {duration_s/60} minutes in: %s", round(time() - t0, 1))
        logger.info(f"New recording duration: {Wired.get_duration()} secs")
        
        # convert to int16
        Wired = spre.astype(Wired, "int16")
        logger.info("Done converting recording as int16 in: %s", round(time() - t0, 1))
                
        # save as int16 binary - our recording now points to the new binary folder 
        logger.info("Saving int16 binary recording...")       
        
        Wired = Wired.save(folder=REC_BIN_INT16_PATH, format='binary', overwrite=True, verbose=True, **job_kwargs)
        logger.info("Done copying int16 binary recording in: %s", round(time() - t0, 1))
        
        # unit-test
        assert Wired.binary_compatible_with(dtype="int16", time_axis=0, file_paths_lenght=1), "not int16 binary"
    
    # load existing copy of int16 binary recording
    Wired = si.load_extractor(REC_BIN_INT16_PATH)

    # remove bad channels
    # speeds up sorting, less memory intensive
    if (not copy_binary_recording) and remove_bad_channels:
        logger.info("Removing bad channels...")
        Wired = remove_the_bad_channels(Wired, bad_channel_ids)
        logger.info(f"Done removing bad channels in: %s", round(time() - t0, 1))

    # run sorting
    t0 = time()
    logger.info("Start sorting...")
    Sorting = ss.run_sorter(sorter_name = sorter,
                                recording = Wired,
                                remove_existing_folder = True,
                                output_folder = output_path,
                                verbose = True,
                                **params)

    # remove the empty units
    logger.info(f"Removing empty units...")
    Sorting = Sorting.remove_empty_units()
    logger.info(f"Done removing empty units.")    
    Sorting = Sorting.frame_slice(start_frame=0, end_frame=end_frame)
    logger.info(f"Done removing excess spikes.")
    sorting_duration = round(time() - t0, 1)
    logger.info(f"Done sorting: took %s", sorting_duration)

    # set metadata
    Sorting = meta.set_sorting_duration(Sorting, sorting_duration)
    Sorting = meta.set_hostname(Sorting)
    logger.info(f"Done running {sorter} in: %s", round(time() - t0, 1))
    logger.info("Saved sorting metadata.")
    
    # clear output path and save
    t0 = time()
    shutil.rmtree(sorting_path, ignore_errors=True)
    Sorting.save(folder=sorting_path)
    logger.info(f"Done saving {sorter} in: %s", round(time() - t0, 1))
    return Sorting, Wired


def sort_and_postprocess_10m(cfg:dict, sorter:str, sorter_params:dict, duration_sec:int=600, is_sort:bool=True,
                             is_postpro:bool=False, extract_wvf:bool=False, copy_binary_recording:bool=False,
                             remove_bad_channels:bool=False, bad_channel_ids: np.array=None):
    """Sort 10-minute recording

    Args:
        cfg (dict): data paths
        sorter (str): name of spike sorter from sorter_dict dictionary (e.g., kilosort2)
        sorter_params (dict): _description_
        duration_sec (int): default 600 sec, duration of chosen period of recording in seconds
        is_sort (bool): default True: sort or not
        is_postpro (bool): default False: postprocess or not
        extract_wvf (bool): default False: extract waveforms or not
        copy_binary_recording (bool): default False: copy recording as int16 binary or not
        remove_bad_channels (bool): default False: remove bad channels or not
    """
    # track
    t0 = time()
    logger.info("Started sorting 10 minutes recording.")
    
    # n_jobs =-1 uses ProcessPoolExecutor on all cores (typically 72 on our machines)
    # takes 5 mins (all cores: -1) instead of 30 mins (2 jobs)
    # tested 2 jobs for evoked (issue with many files (units) open - does not work)
    job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)    

    # READ PATHS
    WIRED_PATH = cfg["probe_wiring"]["full"]["output"]
    PREP_PATH = cfg["preprocessing"]["full"]["output"]["trace_file_path"]

    # WRITE PATHS
    SORTING_PATH = cfg["sorting"]["sorters"][sorter]["10m"]["output"]
    OUTPUT_PATH = cfg["sorting"]["sorters"][sorter]["10m"]["sort_output"]

    # postprocessing
    STUDY_PATH = cfg["postprocessing"]["waveform"]["sorted"]["study"][sorter]["10m"]

    # sort (10 mins)
    if is_sort:
        _, Wired = sort(
            sorter, WIRED_PATH, SORTING_PATH, OUTPUT_PATH, sorter_params, duration_sec,
            copy_binary_recording, remove_bad_channels=remove_bad_channels, bad_channel_ids=bad_channel_ids
            )
        logger.info(f"Done sorting with {sorter} in: %s", round(time() - t0, 1))
    else:
        Wired = si.load_extractor(WIRED_PATH)
        logger.info(f"Skipped sorting for {sorter} in %s", round(time() - t0, 1))

    # postprocess
    if is_postpro:
        meta.postprocess(Wired,
                         PREP_PATH,
                         SORTING_PATH,
                         STUDY_PATH,
                         extract_wvf=extract_wvf,
                         job_kwargs=job_kwargs)
        logger.info(f"Done postprocessing for {sorter}- metadata written in: %s", round(time() - t0, 1))
    else:
        logger.info(f"Skipped postprocessing for {sorter} in %s", round(time() - t0, 1))
    

def sort_and_postprocess_40m(cfg, 
                             sorter, 
                             sorter_params, 
                             duration_sec=2400, 
                             is_sort=True, 
                             is_postpro=False,
                             copy_binary_recording=False,
                             extract_wvf=False,
                             remove_bad_channels=False):
    """sort and postprocess 40 minutes recording over the entire 1 hour
    note: sorting 1 hour crashes due to an insufficient memory issue

    Args:
        cfg (_type_): _description_
        sorter (_type_): _description_
        sorter_params (_type_): _description_
        duration_sec (int, optional): _description_. Defaults to 2400.
        copy_binary_recording (bool, optional): _description_. Defaults to False.
    """
    # track
    t0 = time()
    
    # n_jobs =-1 uses ProcessPoolExecutor on all cores (typically 72 on our machines)
    # takes 5 mins (all cores: -1) instead of 30 mins (2 jobs)
    job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)    

    # SET READ PATHS
    WIRED_PATH = cfg["probe_wiring"]["full"]["output"]
    PREP_PATH = cfg["preprocessing"]["full"]["output"]["trace_file_path"]

    # KILOSORT WRITE PATHS    
    SORTING_PATH = cfg["sorting"]["sorters"][sorter]["40m"]["output"]
    OUTPUT_PATH = cfg["sorting"]["sorters"][sorter]["40m"]["sort_output"]

    # postprocessing
    STUDY_PATH = cfg["postprocessing"]["waveform"]["sorted"]["study"][sorter]["40m"]

    # sort
    if is_sort:          
        _, Wired = sort(sorter, WIRED_PATH, SORTING_PATH, OUTPUT_PATH, 
                        sorter_params, duration_sec, 
                        copy_binary_recording=copy_binary_recording, 
                        remove_bad_channels=remove_bad_channels)
        logger.info(f"Done sorting with {sorter} in: %s", round(time() - t0, 1))
    else:
        Wired = si.load_extractor(WIRED_PATH)
        logger.info(f"Skipped sorting for {sorter} in %s", round(time() - t0, 1))

    # postprocess
    if is_postpro:
        meta.postprocess(Wired, 
                         PREP_PATH, 
                         SORTING_PATH, 
                         STUDY_PATH, 
                         extract_wvf=extract_wvf,
                         job_kwargs=job_kwargs)
        logger.info(f"Done postprocessing for {sorter}- metadata written in: %s", round(time() - t0, 1))
    else:
        logger.info(f"Skipped postprocessing for {sorter} in %s", round(time() - t0, 1))


def sort_and_postprocess_full(cfg, 
                              sorter, 
                              sorter_params,
                              is_sort=True, 
                              is_postpro=False,
                              extract_wvf=False,
                              copy_binary_recording=False,
                              remove_bad_channels=False):
    """Spike sort recording and postprocess 
    (create WaveformExtractor and add metadata such as 
    site layers)

    Args:
        cfg (_type_): _description_
        sorter (_type_): _description_
        sorter_params (_type_): _description_
        is_sort (bool, optional): _description_. Defaults to True.
        is_postpro (bool, optional): _description_. Defaults to True.
        extract_wvf (bool, optional): _description_. Defaults to True.
        copy_binary_recording (bool, optional): _description_. Defaults to False.
        remove_bad_channels (bool, optional): _description_. Defaults to False.
    """
    
    # track
    t0 = time()
    
    # n_jobs =-1 uses ProcessPoolExecutor on all cores (typically 72 on our machines)
    # takes 5 mins (all cores: -1) instead of 30 mins (2 jobs)
    job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)    

    # SET READ PATHS
    WIRED_PATH = cfg["probe_wiring"]["full"]["output"]
    PREP_PATH = cfg["preprocessing"]["full"]["output"]["trace_file_path"]

    # KILOSORT WRITE PATHS    
    SORTING_PATH = cfg["sorting"]["sorters"][sorter]["full"]["output"]
    OUTPUT_PATH = cfg["sorting"]["sorters"][sorter]["full"]["sort_output"]

    # postprocessing
    STUDY_PATH = cfg["postprocessing"]["waveform"]["sorted"]["study"][sorter]["full"]

    # get full duration
    full_duration = si.load_extractor(WIRED_PATH).get_total_duration()
    
    # sort
    if is_sort:
        logger.info(f"Sorting with {sorter} ...")
        sort(sorter,
             WIRED_PATH,
             SORTING_PATH,
             OUTPUT_PATH,
             sorter_params,
             full_duration,
             copy_binary_recording=copy_binary_recording,
             remove_bad_channels=remove_bad_channels)
        logger.info(f"Done sorting with {sorter} in: %s", round(time() - t0, 1))
    else:
        logger.info(f"Skipped sorting for {sorter} in %s", round(time() - t0, 1))
    
    # postprocess
    if is_postpro:
        # load the wired probe with all the electrode sites
        # needed to get the sorted unit nearest sites and layers
        logger.info(f"Loading recording with all sites ...")
        Wired = si.load_extractor(WIRED_PATH)
        
        # postprocess
        logger.info(f"Postprocessing for {sorter} ...")
        meta.postprocess(Wired,
                         PREP_PATH,
                         SORTING_PATH,
                         STUDY_PATH,
                         extract_wvf=extract_wvf,
                         job_kwargs=job_kwargs)
        logger.info(f"Done postprocessing for {sorter} - metadata written in: %s", round(time() - t0, 1))
    else:
        logger.info(f"Skipped postprocessing for {sorter} in %s", round(time() - t0, 1))