"""write to NWB

Usage: 

   sbatch cluster/dandi/write_nwb/raw/biophy_isolated_reyes.sh
    
Takes 30 secs

Returns:
    _type_: _description_
"""
# SETUP PACKAGES 
import os
import spikeinterface as si
import warnings
from pynwb.file import NWBFile, Subject
from pynwb import NWBHDF5IO
from datetime import datetime
from pynwb import NWBHDF5IO
import uuid
from datetime import datetime
from dateutil.tz import tzlocal
from neuroconv.tools.spikeinterface import add_recording, add_sorting
import logging
import logging.config
import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)
from src.nodes.utils import get_config
from src.nodes import utils

# SETUP CONFIG
data_conf, param_conf = get_config("silico_reyes", "2023_01_11").values()
REC_PATH = data_conf["probe_wiring_isolated_cell"]["full"]["output"]
GT_PATH = data_conf["ground_truth_isolated_cell"]["full"]["output"]
NWB_PATH = data_conf["probe_wiring_isolated_cell_nwb"] # write path

# SET PARAMETERS
params = {
    "subject_id": "biophy_isolated_traces_reyes",
    "session_description": "Biophysical simulation of a recording with the Reyes probe for 40 secs in the spontaneous regime. The was isolated for cell 3754013.",
    "identifier": str(uuid.uuid4()),
    "session_start_time": datetime.now(tzlocal()),
    "experimenter": "Laquitaine Steeve",
    "lab": "Blue Brain Project",
    "institution": "EPFL",
    "experiment_description": "Biophysical simulation of a recording with the Reyes probe for 40 secs in the spontaneous regime. The was isolated for cell 3754013.",
    "session_id": "006",
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
        keywords=["Biophysical simulation", "Isolated extracellular traces", "Reyes probe"]
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

# get SortingExtractor
# remove features that are not handeled because 3D
Sorting = si.load_extractor(GT_PATH)
#Sorting.delete_property("orientation")

# create NWB file and add extractors
nwbfile = init_nwbfile(**params)
add_recording(recording=Recording, nwbfile=nwbfile)
add_sorting(sorting=Sorting, nwbfile=nwbfile)

# save this to a NWB file
utils.create_if_not_exists(os.path.dirname(NWB_PATH))
with NWBHDF5IO(path=NWB_PATH, mode="w") as io:
    io.write(nwbfile)