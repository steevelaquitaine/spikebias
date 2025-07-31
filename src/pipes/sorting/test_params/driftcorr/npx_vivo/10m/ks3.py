"""
This script runs a spike sorting pipeline using Kilosort3 with and without drift correction. 
It includes preprocessing, sorting, and postprocessing steps for neural probe recordings.

author: laquitainesteeve@gmail.com 

Modules:
- spikeinterface: For spike sorting and data extraction.
- logging: For detailed pipeline execution logs.
- argparse: For handling command-line arguments.

Key Features:
1. Drift correction: Sort data with or without drift correction.
2. Configurable parameters: Customize sorting parameters for Kilosort3.
3. Logging: Detailed logs for debugging and monitoring.
4. Command-line arguments: Dynamic configuration of input/output paths.

Command-line Arguments:
- `--recording-path`: Path to the wired probe recording.
- `--preprocess-path`: Path to the preprocessed recording.
- `--sorting-path-corrected`: Output path for sorting results with drift correction.
- `--sorting-output-path-corrected`: Output path for postprocessed results with drift correction.
- `--study-path-corrected`: Output path for study results with drift correction.
- `--sorting-path-not-corrected`: Output path for sorting results without drift correction.
- `--sorting-output-path-not-corrected`: Output path for postprocessed results without drift correction.
- `--study-path-not-corrected`: Output path for study results without drift correction.

Pipeline Steps:
1. Preprocessing: Prepares recording data for sorting.
2. Sorting: Executes Kilosort3 spike sorting with/without drift correction.
3. Postprocessing: Extracts waveforms and analyzes sorted data.
4. Comparison: Compares sorting results with/without drift correction.

Sorting Parameters:
- `detect_threshold`: Spike detection threshold.
- `projection_threshold`: Projection threshold.
- `preclust_threshold`: Pre-clustering threshold.
- `car`: Common average referencing flag.
- `minFR`: Minimum firing rate.
- `nblocks`: Number of processing blocks.
- `sig`: Signal standard deviation.
- `freq_min`: Minimum frequency for filtering.
- `sigmaMask`: Sigma mask for clustering.
- `lam`: Lambda parameter for clustering.
- `nPCs`: Number of principal components.
- `ntbuff`: Buffer size for processing.
- `nfilt_factor`: Filter factor.
- `do_correction`: Drift correction flag.
- `NT`: Buffer size for processing.
- `AUCsplit`: AUC split threshold.
- `wave_length`: Waveform length.
- `keep_good_only`: Flag to keep only good units.
- `skip_kilosort_preprocessing`: Flag to skip preprocessing.
- `scaleproc`: Scaling factor for processing.
- `save_rez_to_mat`: Flag to save results to .mat file.
- `delete_tmp_files`: Temporary files to delete.
- `delete_recording_dat`: Flag to delete recording data.

Outputs:

- Sorted data with/without drift correction.
- Postprocessed waveforms and study results.
- Comparison of total and single units between corrected and non-corrected sorting.

Usage:
1. Enable forward compatibility for CUDA libraries:
    
    ```bash
    sudo apt install gcc-11 g++-11
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
    cd /path/to/Kilosort3/CUDA/
    matlab -batch mexGPUall
    ```
2. Activate `spikesort_rtx5090` environment.

    conda activate envs/spikesort_rtx5090

3. Run the script with appropriate command-line arguments.

    nohup python -m src.pipes.sorting.test_params.driftcorr.npx_vivo.10m.ks3 \
        --recording-path dataset/00_raw/recording_marques_smith \
            --preprocess-path dataset/01_intermediate/preprocessing/recording_marques_smith \
                --sorting-path-corrected ./temp/npx_vivo/SortingKS3_10m_RTX5090_DriftCorr \
                    --sorting-output-path-corrected ./temp/npx_vivo/KS3_output_10m_RTX5090_DriftCorr/ \
                        --study-path-corrected ./temp/npx_vivo/study_ks3_10m_RTX5090_DriftCorr/ \
                            --sorting-path-not-corrected ./temp/npx_vivo/SortingKS3_10m_RTX5090_NoDriftCorr \
                                --sorting-output-path-not-corrected ./temp/npx_vivo/KS3_output_10m_RTX5090_NoDriftCorr/ \
                                    --study-path-not-corrected ./temp/npx_vivo/study_ks3_10m_RTX5090_NoDriftCorr/ \
                                            --extract-waveforms \
                                                --remove-bad-channels \
                                                    > out_ks3_m.log
"""

# import python packages
import os
import sys
import logging
import logging.config
import yaml
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface as si
import argparse
import torch
print("spikeinterface", si.__version__)

# project path
PROJ_PATH = "/home/steeve/steeve/epfl/code/spikebias/"
os.chdir(PROJ_PATH)
sys.path.append(os.path.join(PROJ_PATH, "src")) # enable custom package import

# import spikebias package
from src.nodes.sorting import sort_and_postprocess_10m

# recording parameters
REC_SECS = 600 

# sorting parameters
SORTER = "kilosort3"
SORTER_PATH = "/home/steeve/steeve/epfl/code/spikebias/dataset/01_intermediate/sorters/Kilosort3_buttw_forwcomp"
SORTER_PARAMS = {
    "detect_threshold": 6,
    "projection_threshold": [9, 9],
    "preclust_threshold": 8,
    "car": True,
    "minFR": 0, # modified
    "minfr_goodchannels": 0, # modified
    "nblocks": 5,
    "sig": 20,
    "freq_min": 300,
    "sigmaMask": 30,
    "lam": 20.0,
    "nPCs": 3,
    "ntbuff": 64,
    "nfilt_factor": 4,
    "do_correction": True,
    "NT": 65792, # modified
    "AUCsplit": 0.8,
    "wave_length": 61,
    "keep_good_only": False,
    "skip_kilosort_preprocessing": False,
    "scaleproc": None,
    "save_rez_to_mat": False,
    "delete_tmp_files": ("matlab_files",),
    "delete_recording_dat": False,
}

# manually selected channels to remove (most outside the cortex)
bad_channel_ids = None

# SET KS3 software environment variable
ss.Kilosort3Sorter.set_kilosort3_path(SORTER_PATH)

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


if __name__ == "__main__":
    """Entry point
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run spike sorting pipeline with or without drift correction.")
    parser.add_argument("--recording-path", default= './dataset/00_raw/recording_npx_spont', help="wired probe recording path.")
    parser.add_argument("--preprocess-path", default= './dataset/01_intermediate/preprocessing/recording_npx_spont', help="preprocess recording path")
    
    parser.add_argument("--remove-bad-channels", action='store_true', help="remove bad channels or not")
    parser.add_argument("--extract-waveforms", action='store_true', help="whether to extract waveforms or not")
    
    parser.add_argument("--sorting-path-corrected", default='./temp/SortingKS4_5m_RTX5090_DriftCorr', help="sorting output path")
    parser.add_argument("--sorting-output-path-corrected", default='./temp/KS4_output_5m_RTX5090_DriftCorr/', help="postprocess output path")
    parser.add_argument("--study-path-corrected", default='./temp/study_ks4_5m_RTX5090_DriftCorr/', help="study output path")

    parser.add_argument("--sorting-path-not-corrected", default='./temp/SortingKS4_5m_RTX5090_NoDriftCorr', help="sorting output path without correction")
    parser.add_argument("--sorting-output-path-not-corrected", default='./temp/KS4_output_5m_RTX5090_NoDriftCorr/', help="postprocess output path without correction")
    parser.add_argument("--study-path-not-corrected", default='./temp/study_ks4_5m_RTX5090_NoDriftCorr/', help="study output path without correction")
    
    args = parser.parse_args()
    
    # report parameters for visual check
    logger.info(f"recording_path: {args.recording_path}")
    logger.info(f"preprocess_path: {args.preprocess_path}")
    logger.info(f"remove_bad_channels: {args.remove_bad_channels}")
    logger.info(f"extract_waveforms: {args.extract_waveforms}")
    logger.info(f"sorting_path_corrected: {args.sorting_path_corrected}")
    logger.info(f"sorting_output_path_corrected: {args.sorting_output_path_corrected}")
    logger.info(f"study_path_corrected: {args.study_path_corrected}")
    logger.info(f"sorting_path_not_corrected: {args.sorting_path_not_corrected}")
    logger.info(f"sorting_output_path_not_corrected: {args.sorting_output_path_not_corrected}")
    logger.info(f"study_path_not_corrected: {args.study_path_not_corrected}")
    # configure read and write paths

    # with drift correction

    CFG_CORR = {
        'probe_wiring': {
            'full': {
                'output': args.recording_path
            }
        },
        'preprocessing': {
            'full': {
                'output': {
                    'trace_file_path': args.preprocess_path
                }
            }
        },
        'sorting': {
            'sorters': {
                f"{SORTER}": {
                    '10m': {
                        'output': args.sorting_path_corrected,
                        'sort_output':args.sorting_output_path_corrected
                    }
                }
            }
        },
        'postprocessing': {
            'waveform': {
                'sorted': {
                    'study': {
                        f"{SORTER}": {
                            '10m': args.study_path_corrected
                        }
                    }
                }
            }
        }
    }

    # without drift correction

    CFG_NO_CORR = {
        'probe_wiring': {
            'full': {
                'output': args.recording_path
            }
        },
        'preprocessing': {
            'full': {
                'output': {
                    'trace_file_path': args.preprocess_path
                }
            }
        },
        'sorting': {
            'sorters': {
                f"{SORTER}": {
                    '10m': {
                        'output': args.sorting_path_not_corrected,
                        'sort_output': args.sorting_output_path_not_corrected
                    }
                }
            }
        },
        'postprocessing': {
            'waveform': {
                'sorted': {
                    'study': {
                        f"{SORTER}": {
                            '10m': args.study_path_not_corrected
                        }
                    }
                }
            }
        }
    }

    # ensure the script is executed as the main program
    logger.info("Starting the sorting pipeline...")

    # sort with drift correction
    logger.info("Sorting with drift correction...")

    # spike sort    
    sort_and_postprocess_10m(CFG_CORR, SORTER, SORTER_PARAMS, duration_sec=REC_SECS, 
                            is_sort=True, is_postpro=False, extract_wvf=False, copy_binary_recording=True,
                            remove_bad_channels=args.remove_bad_channels, bad_channel_ids = bad_channel_ids)
    # post-process
    torch.cuda.empty_cache()
    sort_and_postprocess_10m(CFG_CORR, SORTER, SORTER_PARAMS, duration_sec=REC_SECS,
                            is_sort=False, is_postpro=True, extract_wvf=True, copy_binary_recording=True,
                            remove_bad_channels=False)

    logger.info("Sorting with drift correction...DONE")

    # sort without drift correction

    logger.info("Sorting without drift correction...")

    SORTER_PARAMS['do_correction'] = False

    # spike sort
    torch.cuda.empty_cache()
    sort_and_postprocess_10m(CFG_NO_CORR, SORTER, SORTER_PARAMS, duration_sec=REC_SECS,
                            is_sort=True, is_postpro=False, extract_wvf=False, copy_binary_recording=True,
                            remove_bad_channels=args.remove_bad_channels, bad_channel_ids = bad_channel_ids)
    # post-process
    torch.cuda.empty_cache()
    sort_and_postprocess_10m(CFG_NO_CORR, SORTER, SORTER_PARAMS, duration_sec=REC_SECS,
                            is_sort=False, is_postpro=True, extract_wvf=True, copy_binary_recording=True,
                            remove_bad_channels=False)

    logger.info("Sorting without drift correction...DONE")

    # compare sorting results
    SortingCorr = si.load_extractor(args.sorting_path_corrected)
    SortingNoCorr = si.load_extractor(args.sorting_path_not_corrected)
    
    # display total units
    print("Total units:")
    print(f"With correction: {len(SortingCorr.unit_ids)}")
    print(f"Without correction: {len(SortingNoCorr.unit_ids)}")

    # display single units
    print("\nSingle units:")
    print(f"With correction: {sum(SortingCorr.get_property('KSLabel') == 'good')}")
    print(f"Without correction: {sum(SortingNoCorr.get_property('KSLabel') == 'good')}")

    logger.info("Pipeline completed successfully.")