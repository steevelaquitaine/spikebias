"""Export dataset to the NWB format

Usage:  

    sbatch cluster/nwb/dense_spont_p3.sh

Returns:
    _type_: _description_
"""
# SETUP PACKAGES 
import os
import spikeinterface as si
import numpy as np
import warnings
from pynwb.file import NWBFile, Subject
from pynwb import NWBHDF5IO
from datetime import datetime
from pathlib import Path
from pynwb import NWBHDF5IO
import uuid
from datetime import datetime
from dateutil.tz import tzlocal
import spikeinterface.extractors as se
from neuroconv.tools.spikeinterface import add_recording, add_sorting
import logging
import logging.config
import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


# SET PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023"
os.chdir(PROJ_PATH)
from src.nodes.utils import get_config
from src.nodes import utils
from src.nodes.dataeng.silico import probe_wiring

# SETUP CONFIG
data_conf, param_conf = get_config("dense_spont", "probe_3").values()
REC_PATH = data_conf["recording"]["output"]
GT_PATH = data_conf["ground_truth"]["output"]
NWB_PATH = data_conf["nwb"]

# SET PARAMETERS
params = {
    "subject_id": "005",
    "session_description": "Biophysical simulation of dense probe at depth 3 in the spontaneous regime",
    "identifier": str(uuid.uuid4()),
    "session_start_time": datetime.now(tzlocal()),
    "experimenter": "Laquitaine Steeve",
    "lab": "Blue Brain Project",
    "institution": "EPFL",
    "experiment_description": "Biophysical simulation of dense probe at depth 3",
    "session_id": "005",
    "related_publications": "doi:"
    }
    
# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def init_nwbfile(
    subject_id="001",
    session_description="Biophysical simulation in the spontaneous regime",
    identifier=str(uuid.uuid4()),
    session_start_time=datetime.now(tzlocal()),
    experimenter="Steeve Laquitaine",
    lab="Blue Brain Project",
    institution="EPFL",
    experiment_description="Biophysical simulation",
    session_id="001",
    related_publications="doi:",
):

    # parmaetrize session file
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        experimenter=experimenter,
        lab=lab,
        institution=institution,
        experiment_description=experiment_description,
        session_id=session_id,
        related_publications=related_publications,
        keywords=["Biophysical simulation"]
    )

    # subject metadata
    nwbfile.subject = Subject(
        subject_id=subject_id,
        species="Rattus norvegicus",
        age="P14D",
        sex="M",
        description="Wistar Rat",
    )
    return nwbfile


# get raw wired RecordingExtractor
Recording = si.load_extractor(REC_PATH)
Recording = probe_wiring.run(Recording, data_conf, param_conf)

# get SortingExtractor
# remove features that are not handeled because 3D
Sorting = si.load_extractor(GT_PATH)
#Sorting.delete_property("orientation")
logger.info("Ground truth metadata:")
logger.info(Sorting.get_property_keys())

# create NWB file and add extractors
nwbfile = init_nwbfile(**params)
add_recording(recording=Recording, nwbfile=nwbfile)
add_sorting(sorting=Sorting, nwbfile=nwbfile)

# save this to a NWB file
utils.create_if_not_exists(os.path.dirname(NWB_PATH))
with NWBHDF5IO(path=NWB_PATH, mode="w") as io:
    io.write(nwbfile)