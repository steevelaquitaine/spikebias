"""sort and post-process 10 min of spontaneous recording with Kilosort 4.0 with NVIDIA RTX 5090 GPU

author: laquitainesteeve@gmail.com

Tested on Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K ＠3.2 GHz/5.8 GHz) with RTX 5090 GPU with 40GB VRAM

GPU resources <3GB VRAM and 100% util

Usage: 

    2. Activate `kilosort4_rtx5090` environment.
        
        conda activate envs/kilosort4_rtx5090
        
    3. Run the script with appropriate command-line arguments.

        nohup python -m src.pipes.sorting.npx_synth2.ks4 \
            --recording-path dataset/00_raw/recording_synth2 \
                --preprocess-path dataset/00_raw/recording_synth2 \
                    --sorting-path-corrected ./temp/pros_of_details/npx_synth2/SortingKS4_10m_RTX5090 \
                        --sorting-output-path-corrected ./temp/pros_of_details/npx_synth2/KS4_output_10m_RTX5090/ \
                            --study-path-corrected ./temp/pros_of_details/npx_synth2/study_ks4_10m_RTX5090/ \
                                --extract-waveforms \
                                    --remove-bad-channels \
                                        > out_ks4_synth2.log
"""

# import python packages
import os
import numpy as np
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
torch.cuda.empty_cache()

# setup project path
PROJ_PATH = "/home/steeve/steeve/epfl/code/spikebias/"
os.chdir(PROJ_PATH)
sys.path.append(os.path.join(PROJ_PATH, "src")) # enable custom package import

# import spikebias package
from src.nodes.sorting import sort_and_postprocess_10m

# setup recording parameters
REC_SECS = 600

# setup sorting parameters
SORTER = "kilosort4"

# these are the default parameters
# for spikeinterface 0.100.5
# note that there are no minFR and minFR_channels in ks4
# - we set batch_size to 10,000 instead of 60,0000 due to memory constrains
# - we set dminx to 25.6 um instead of None
SORTER_PARAMS = {
    "batch_size": 10000,
    "nblocks": 1,
    "Th_universal": 9,
    "Th_learned": 8,
    "do_CAR": True,
    "invert_sign": False,
    "nt": 61,
    "artifact_threshold": None,
    "nskip": 25,
    "whitening_range": 32,
    "binning_depth": 5,
    "sig_interp": 20,
    "nt0min": None,
    "dmin": None,
    "dminx": 25.6,
    "min_template_size": 10, # um
    "template_sizes": 5,     #  Number of sizes for universal spike templates (multiples of the min_template_size)
    "nearest_chans": 10,
    "nearest_templates": 100,
    "templates_from_data": True,
    "n_templates": 6,
    "n_pcs": 6,
    "Th_single_ch": 6,
    "acg_threshold": 0.2,
    "ccg_threshold": 0.25,
    "cluster_downsampling": 20,
    "cluster_pcs": 64,
    "duplicate_spike_bins": 15,
    "do_correction": True,
    "keep_good_only": False,
    "save_extra_kwargs": False,
    "skip_kilosort_preprocessing": False,
    "scaleproc": None,
}

# manually selected channels to remove (most outside the cortex)
# check with np.where(RecordingNS.get_property('layers')=='Outside')[0]
# bad_channel_ids = None
bad_channel_ids = np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  73,  74,  75,
        76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,
        89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
       102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
       115, 116, 117, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
       179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
       192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204,
       205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 265, 266, 267,
       268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
       281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293,
       294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306,
       307, 308, 309, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370,
       371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383])
# bad_channel_ids = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#             '86', '87', '88', '89', '90', '91', '92', '93', '94', '95',
#             '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
#             '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', 
#             '192', '193', '194', '195', '196', '197', '198', '199', '200', '201',
#             '278', '279', '280', '281', '282', '283', '284', '285', '286', '287',
#             '288', '289', '290', '291', '292', '293', '294', '295', '296', '297',
#             '374', '375', '376', '377', '378', '379', '380', '381', '382', '383'
#             ])

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

    args = parser.parse_args()
    
    # report parameters for visual check
    logger.info(f"recording_path: {args.recording_path}")
    logger.info(f"preprocess_path: {args.preprocess_path}")
    logger.info(f"remove_bad_channels: {args.remove_bad_channels}")
    logger.info(f"extract_waveforms: {args.extract_waveforms}")
    logger.info(f"sorting_path_corrected: {args.sorting_path_corrected}")
    logger.info(f"sorting_output_path_corrected: {args.sorting_output_path_corrected}")
    logger.info(f"study_path_corrected: {args.study_path_corrected}")
    
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

    # compare sorting results
    SortingCorr = si.load_extractor(args.sorting_path_corrected)
    
    # display total units
    print("Total units:")
    print(f"With correction: {len(SortingCorr.unit_ids)}")

    # display single units
    print("\nSingle units:")
    print(f"With correction: {sum(SortingCorr.get_property('KSLabel') == 'good')}")

    logger.info("Pipeline completed successfully.")