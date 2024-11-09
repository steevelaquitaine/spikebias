"""pipeline that replicates the comparison of kilosort2 (bonus Kilosort3) sorting with ground truth data
done in Buccino 2020's paper to validate our adjusted Kilosort2 and 3 pipelines.
    
TODO:

* has been replaced by compare_kilosort2_3_and_truth, to delete

Note:

    We needed to update Buccino 2020 paper notebook's code to run without docker container.
    We cannot use the "studyObject" framework which directly compaires all sorters
    because it relies on docker containers to handle the various sorter software (matlab, python)
    and dependency incompatibilities even between python softwares.
    
Results:

    Currently, we replicate Buccino's results but not perfectly.

Usage:

    # clone Buccino 2020's kilosort2 commit version in sorters_packages/
    git clone https://github.com/MouseLand/Kilosort2
    git checkout -q 48bf2b81d8ad

    # activate spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # TODO
    # - update env name
    # activate test_kilosort2_env 
    rm -rf ~/.cache/pip  # clear pip cache
    python3.9 -m venv test_kilosort2_env
    source test_kilosort2_env/bin/activate
    pip3.9 install -r requirements_test_kilosort2_env.txt

    # setup GPU for matlab
    cd sorters_packages/Kilosort2/CUDA/
    matlab -batch mexGPUall
    cd ../../../

    # run on cluster
    sbatch sbatch/test_kilosort2_and_3_vs_gt.sbatch

references:    
    (1) https://spikeinterface.github.io/blog/ground-truth-comparison-and-ensemble-sorting-of-a-synthetic-neuropixels-recording/
"""

import logging
import logging.config
import os
from pathlib import Path
from sys import argv
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.extractors as se
import spikeinterface.full as si_full
import spikeinterface.sorters as ss
import yaml

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# get config id
for arg_i, argument in enumerate(argv):
    if argument == "--conf":
        CONF_DATE = argv[arg_i + 1]

# read the pipeline's configuration
with open(
    f"conf/buccino_2020/{CONF_DATE}/dataset.yml", encoding="utf-8"
) as data_conf:
    data_conf = yaml.safe_load(data_conf)

# add Kilosort2 and 3 to path
ss.Kilosort2Sorter.set_kilosort2_path(
    data_conf["sorting"]["sorters"]["kilosort2"]["path"]
)
ss.Kilosort3Sorter.set_kilosort3_path(
    data_conf["sorting"]["sorters"]["kilosort3"]["path"]
)

# set data path
data_path = data_conf["sorting"]["ground_truth"]["input"]
data_path = Path(data_path)

# the raw data: this NWB file contains both the ground truth spikes and the raw data
data_filename = data_conf["raw_data"]["input"]

# set kilosort2 sorting write path
kilosort2_sorting_path = data_conf["sorting"]["sorters"]["kilosort2"]["output"]
kilosort3_sorting_path = data_conf["sorting"]["sorters"]["kilosort3"]["output"]

# recording_path = data_path / "recording/"
recording_path = data_conf["recording"]["input"]
gt_sorting_path = data_conf["sorting"]["ground_truth"]["output"]

# Figure path
FIG_PATH = data_conf["figures"]["sorters"][
    "kilosort2_and_3_replication_of_Buccino_2020"
]


# get Spikeinterface Recording object
print("getting recording")
if os.path.isdir(recording_path):
    t0 = time()
    logger.info("loading already processed recording")
    RX = si.load_extractor(recording_path)
    logger.info(
        "loading already processed recording done: %s", round(time() - t0, 1)
    )
else:
    t0 = time()
    logger.info("loading Nwb recording")
    RX = se.NwbRecordingExtractor(str(data_filename))
    logger.info("loading Nwb recording - done: %s", round(time() - t0, 1))
    # bandpass
    t0 = time()
    logger.info("filtering recording")
    RX = si_full.bandpass_filter(RX, freq_min=300, freq_max=6000)
    logger.info("filtering done: %s", round(time() - t0, 1))
    t0 = time()
    logger.info("saving")
    RX.save(folder=recording_path)
    logger.info("saving done: %s", round(time() - t0, 1))

# get Spikeinterface ground truth Sorting object
if os.path.isdir(gt_sorting_path):
    t0 = time()
    logger.info("loading already processed true sorting")
    SX_gt = si.load_extractor(gt_sorting_path)
    logger.info(
        "loading already processed true sorting - done: %s",
        round(time() - t0, 1),
    )
else:
    t0 = time()
    logger.info("loading NwB true sorting")
    SX_gt = se.NwbSortingExtractor(str(data_filename))
    logger.info(
        "loading NwB true sorting - done: %s",
        round(time() - t0, 1),
    )
    t0 = time()
    logger.info("saving")
    SX_gt.save(folder=gt_sorting_path)
    logger.info(
        "saving - done: %s",
        round(time() - t0, 1),
    )

# get Kilosort2 sorting
if os.path.isdir(kilosort2_sorting_path):
    t0 = time()
    logger.info("loading kilosort2 sorting")
    sorting_KS2 = si.load_extractor(kilosort2_sorting_path)
    logger.info(
        "loading kilosort2 sorting - done: %s",
        round(time() - t0, 1),
    )
else:
    # run sorting
    t0 = time()
    logger.info("running kilosort2 sorting")
    sorting_KS2 = ss.run_kilosort2(RX, verbose=True)
    logger.info(
        "running kilosort2 sorting - done: %s",
        round(time() - t0, 1),
    )
    t0 = time()
    logger.info("saving kilosort2 sorting")
    sorting_KS2.save(folder=kilosort2_sorting_path)
    logger.info(
        "saving kilosort2 sorting - done: %s",
        round(time() - t0, 1),
    )

# get Kilosort3 sorting
if os.path.isdir(kilosort3_sorting_path):
    t0 = time()
    logger.info("loading kilosort3 sorting")
    sorting_KS3 = si.load_extractor(kilosort3_sorting_path)
    logger.info(
        "loading kilosort3 sorting - done: %s",
        round(time() - t0, 1),
    )
else:
    # run sorting
    t0 = time()
    logger.info("running kilosort3 sorting")
    sorting_KS3 = ss.run_kilosort3(RX, verbose=True)
    logger.info(
        "running kilosort3 sorting - done: %s",
        round(time() - t0, 1),
    )
    t0 = time()
    logger.info("saving kilosort3 sorting")
    sorting_KS3.save(folder=kilosort3_sorting_path)
    logger.info(
        "saving kilosort3 sorting - done: %s",
        round(time() - t0, 1),
    )

# comppare Kilosort2 sorted units to ground truth
t0 = time()
logger.info("running comparison Kilosort2 summary")
cmp_gt_KS2 = sc.compare_sorter_to_ground_truth(
    SX_gt, sorting_KS2, exhaustive_gt=True, match_score=0.1
)
logger.info(
    "running comparison Kilosort2 summary - done: %s",
    round(time() - t0, 1),
)

# comppare Kilosort3 sorted units to ground truth
t0 = time()
logger.info("running comparison Kilosort3 summary")
cmp_gt_KS3 = sc.compare_sorter_to_ground_truth(
    SX_gt, sorting_KS3, exhaustive_gt=True, match_score=0.1
)
logger.info(
    "running comparison Kilosort3 summary - done: %s",
    round(time() - t0, 1),
)

# write our comparison summary
# between Kilosort2 and the ground truth used
# in Buccino 2020

# set the original values from Buccino's 2020 paper notebook (ref 1)
comparison_summary_df = pd.DataFrame(
    data=[250, 415, 245, 21, 2, 147, 168],
    index=[
        "num_gt",
        "num_sorter",
        "num_well_detected",
        "num_redundant",
        "num_overmerged",
        "num_false_positive",
        "num_bad",
    ],
    columns=["Kilosort2_Buccino_2020_values"],
)

# our replication
comparison_summary_df["Our_Kilosort2_replication"] = [
    SX_gt.get_num_units(),
    sorting_KS2.get_num_units(),
    cmp_gt_KS2.count_well_detected_units(well_detected_score=0.1),
    cmp_gt_KS2.count_redundant_units(),
    cmp_gt_KS2.count_overmerged_units(),
    cmp_gt_KS2.count_false_positive_units(),
    cmp_gt_KS2.count_bad_units(),
]

# get Kilosort3 data
comparison_summary_df["Kilosort3"] = [
    SX_gt.get_num_units(),
    sorting_KS3.get_num_units(),
    cmp_gt_KS3.count_well_detected_units(well_detected_score=0.1),
    cmp_gt_KS3.count_redundant_units(),
    cmp_gt_KS3.count_overmerged_units(),
    cmp_gt_KS3.count_false_positive_units(),
    cmp_gt_KS3.count_bad_units(),
]

comparison_summary_df = comparison_summary_df.T

# plot our replication of Buccino 2020
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
comparison_summary_df.T.plot.bar(
    ax=ax, rot=0, color=[(0.61, 0.79, 0.88), (0.19, 0.50, 0.74), (0, 0, 0)]
)
plt.xlabel("Study", fontsize=20)
plt.ylabel("Units (count)", fontsize=20)
plt.xticks(fontsize=16)
plt.title("Replication of Buccino 2020's Kilosort performance", fontsize=20)
plt.legend(fontsize=20)
fig.savefig(FIG_PATH)
fig.savefig(FIG_PATH)
