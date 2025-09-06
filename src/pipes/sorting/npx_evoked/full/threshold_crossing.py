"""apply threshold crossing by electrode site (no spike sorting)

Usage:

    2. Activate `spikebias` environment.
        
        conda activate envs/spikebias
        
    3. Run the script with appropriate command-line arguments.

        nohup python -m src.pipes.sorting.npx_evoked.full.treshold_crossing \
            --recording-path dataset/00_raw/recording_npx_evoked \
                --save-path temp/no_spike_sorting/sorting_peak_npx_evoked \
                    --freq-min 300 \
                        > out_peak_detect.log

Execution time: 5 min
"""
import numpy as np
import logging
import logging.config
import yaml
import argparse
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import pandas as pd

# custom package
import spikeinterface as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

REFERENCE = "global"       # common reference preprocessing
OPERATOR = "median"        # common reference preprocessing

# set compute parameters
job_kwargs = dict(chunk_duration='1s', n_jobs=8, progress_bar=True)

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


if __name__ == "__main__":
    """Entry point
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run spike sorting threshold crossing pipeline.")
    parser.add_argument("--recording-path", default= './dataset/00_raw/recording_npx_evoked', help="wired probe recording path.")
    parser.add_argument("--save-path", default='./temp/no_spike_sorting/sorting_peak_npx_evoked', help="output extractor save path")
    parser.add_argument("--freq-min", type=int, default=300, help="high pass filter cutoff")
    args = parser.parse_args()
    
    # report parameters for visual check
    logger.info(f"recording_path: {args.recording_path}")
    logger.info(f"freq_min: {args.freq_min}")
    logger.info(f"save_path: {args.save_path}")

    # load and preprocess
    Recording = si.load_extractor(args.recording_path)
    Recording = spre.highpass_filter(Recording, freq_min=args.freq_min)
    Recording = spre.common_reference(Recording, reference=REFERENCE, operator=OPERATOR)
    Recording = spre.whiten(Recording, dtype='float32') # requires float

    # apply threshold crossing
    peaks = detect_peaks(
        recording=Recording,
        method='by_channel',
        peak_sign='neg',
        detect_threshold=5,
        **job_kwargs,
    )

    # Create threshold crossing extractor
    # - get sorting parameters
    duration = Recording.get_total_duration()
    sampling_frequency = Recording.get_sampling_frequency()
    peaks_df = pd.DataFrame(peaks)
    units = peaks_df.channel_index.values
    times = peaks_df.sample_index.values
    ThreshCrossing = se.NumpySorting.from_times_labels(times, units, sampling_frequency)
    
    # - add metadata
    ThreshCrossing.set_property('layer', Recording.get_property('layers'))
    print(ThreshCrossing)

    # save results
    ThreshCrossing.save(folder=args.save_path, overwrite=True)
    #np.save(SAVE_PATH, peaks)
    logger.info("Pipeline done")
