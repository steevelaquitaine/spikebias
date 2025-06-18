"""Module for waveform extraction

usage:

    python src/nodes/waveform/run.py \
        --raw-recording-path '/home/steeve/steeve/epfl/code/spikebias/dataset/00_raw/recording_npx_spont'\
        --sorting-path '/home/steeve/steeve/epfl/code/spikebias/dataset/01_intermediate/sorting/npx_spont/SortingKS3' \
        --waveform-path '/home/steeve/steeve/epfl/code/spikebias/dataset/01_intermediate/waveforms/npx_spont/SortingKS3' \
        --sampling-freq 40000

tested on: 
    - Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K @3.2 GHz/5.8 GHz)

execution time: 3 min
"""
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import yaml
import logging
import logging.config
import logging.config
from time import time
import shutil 
import argparse
from spikeinterface import extract_waveforms

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

from src.nodes.prepro.run import run as prep


def run(Sorting, freq_min, waveform_path, 
        sparse, ms_before, ms_after, max_spikes_per_unit, 
        unit_batch_size, overwrite, seed, n_jobs, chunk_size
        progress_bar, sampling_freq):

    # load/process/save the recording
    Recording = prep(Recording, freq_min=freq_min)
    
    # take 10 min
    Recording = Recording.frame_slice(start_frame=0, end_frame=sampling_freq*600) # 10 min

    # load/process waveform
    job_kwargs = dict(n_jobs=32, chunk_size=800000, progress_bar=True)

    # extract waveforms
    WaveExtrator = extract_waveforms(Recording, Sorting, waveform_path, 
                        sparse=sparse, ms_before=ms_before,
                        ms_after=ms_after, max_spikes_per_unit=max_spikes_per_unit,
                        unit_batch_size=unit_batch_size, overwrite=overwrite,
                        seed=seed, **job_kwargs)
    return WaveExtrator



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Preprocess recording")
    
    # parameters
    parser.add_argument("--raw-recording-path", type=str, 
                        default=None, 
                        help="path of recording to preprocess")    
    parser.add_argument("--prep-recording-path", type=str, 
                        default=None, 
                        help="write path of recording to preprocess")        
    parser.add_argument("--sorting-path", type=str, 
                        default=None, 
                        help="path of sorting")    
    parser.add_argument("--freq-min", type=int, 
                        default=300, 
                        help="lower frequency of the high-pass filter")    
    parser.add_argument("--waveform-path", type=str, 
                        default=None, 
                        help="write path for waveforms")            
    parser.add_argument("--ms-before", type=float, 
                        default=3.0, 
                        help="ms before timestamp")        
    parser.add_argument("--ms-after", type=float, 
                        default=3.0, 
                        help="ms after timestamp")                
    parser.add_argument("--max-spikes-per-unit", type=int, 
                        default=500, 
                        help="max spikes per unit")     
    parser.add_argument("--unit-batch-size", type=int, 
                        default=200, 
                        help="max spikes per unit")                        
    parser.add_argument("--sampling-freq", type=int, 
                        default=40000, 
                        help="acquisition sampling frequency")        
    parser.add_argument("--n-jobs", type=int, 
                        default=30, 
                        help="number of jobs for multi-core processing")
    parser.add_argument("--dtype", type=str, 
                        default='float32', 
                        help="data type")
    parser.add_argument("--chunk-size", type=int, 
                        default=800000, 
                        help="number of samples in a memory chunk")    
    args = parser.parse_args()        

    # load recording and sorting
    Recording = si.load_extractor(args.raw_recording_path)
    Sorting = si.load_extractor(args.sorting_path)

    # extract waveforms
    WaveExtractor = run(Recording, Sorting, args.freq_min, 
                    args.waveform_path, args.ms_before, args.ms_after, args.max_spikes_per_unit, 
                    args.unit_batch_size, True, args.seed, args.n_jobs, 
                    args.chunk_size, args.progress_bar)