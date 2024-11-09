"""_summary_

Returns:
    _type_: _description_
"""

import logging
import logging.config

#!/usr/bin/env python
# coding: utf-8
import os
import shutil
import sys
from typing import Any, Dict

import spikeinterface.full as si
import yaml

from src.pipes.silico import recording

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def run(dataset_conf: dict, param_conf: dict):
    """run preprocessing
    # TODO: continue here !

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # simulated recordings
    (
        recording_write_path,
        simulated_traces,
    ) = recording__.run(dataset_conf, param_conf)

    # preprocess simulated recordings
    bandpassed_recording = si.bandpass_filter(
        simulated_traces, freq_min=300, freq_max=6000
    )

    # set common reference
    referenced_recording = si.common_reference(
        bandpassed_recording, reference="global", operator="median"
    )

    # write preprocessed recordings
    shutil.rmtree(recording_write_path, ignore_errors=True)
    rec_ready = referenced_recording.save(
        folder=recording_write_path, format="binary"
    )  # , n_jobs = 10, total_memory = '30G')
    return referenced_recording


if __name__ == "__main__":

    # parse pipeline parameters
    conf_date = sys.argv[1]

    # read the pipeline's configuration
    with open(
        f"conf/silico/{conf_date}/dataset.yml", encoding="utf-8"
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)

    with open(
        f"conf/silico/{conf_date}/parameters.yml", encoding="utf-8"
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)

    # run preprocessing
    output = run(dataset_conf, param_conf)
