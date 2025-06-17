"""run standard preprocessing (compress, filter, apply common reference)

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

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def run(recording, freq_min:int=300):
    
    # intialize time tracking
    t0 = time()

    # compress to int16 (like the Kilosort sorters)
    recording = spre.astype(recording, "int16")
    logger.info(f"Compressed to int16 in {np.round(time()-t0,2)} secs")
    
    # band-pass filter
    recording = spre.highpass_filter(recording, freq_min=freq_min)
    logger.info(f"High-pass filtered in {np.round(time()-t0,2)} secs")

    # apply common reference
    recording = spre.common_reference(
        recording, reference="global", operator="median"
    )
    logger.info(f"Pipeline completed in {np.round(time()-t0,2)} secs")
    return recording



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Preprocess recording")
    
    # parameters
    parser.add_argument("--recording-path", type=str, 
                        default="dataset/00_raw/<file_name>", 
                        help="path of recording to preprocess")    
    parser.add_argument("--write-path", type=str, 
                        default="dataset/01_intermediate/preprocessing/<file_name>", 
                        help="path of recording to preprocess")        
    parser.add_argument("--freq-min", type=int, 
                        default=300, 
                        help="lower frequency of the high-pass filter")    
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

    # load recording 
    recording = si.load_extractor(args.recording_path)

    # do preprocessing
    recording = run(recording, args.freq_min)
    
    # save 
    recording.save(folder=args.write_path, n_jobs=args.n_jobs, 
                   verbose=True, progress_bar=True, overwrite=True, 
                   dtype=args.dtype, chunk_size=args.chunk_size)