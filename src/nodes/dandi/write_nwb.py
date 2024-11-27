
import os
import spikeinterface as si
from datetime import datetime
from pynwb.file import NWBFile, Subject
from pynwb import NWBHDF5IO
from neuroconv.tools.spikeinterface import add_recording, add_sorting
import uuid
from dateutil.tz import tzlocal
import yaml
import logging
import logging.config

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

# spikebias package
from src.nodes import utils

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
    experimenter="Laquitaine Steeve",
    lab="Blue Brain Project",
    institution="EPFL",
    experiment_description="Biophysical simulation",
    session_id="001",
    related_publications="doi:",
):
    """Initialize NWB file with descriptive metadata

    Args:
        subject_id (str, optional): _description_. Defaults to "001".
        session_description (str, optional): _description_. Defaults to "Biophysical simulation in the spontaneous regime".
        identifier (_type_, optional): _description_. Defaults to str(uuid.uuid4()).
        session_start_time (_type_, optional): _description_. Defaults to datetime.now(tzlocal()).
        experimenter (str, optional): _description_. Defaults to "Laquitaine Steeve".
        lab (str, optional): _description_. Defaults to "Blue Brain Project".
        institution (str, optional): _description_. Defaults to "EPFL".
        experiment_description (str, optional): _description_. Defaults to "Biophysical simulation".
        session_id (str, optional): _description_. Defaults to "001".
        related_publications (str, optional): _description_. Defaults to "doi:".

    Returns:
        _type_: _description_
    """

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
        keywords=["Biophysical simulation", "dense extracellular recordings", "spike sorting"]
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



def run(rec_path: str, gt_path: str, nwb_path: str,  params: dict):
    """write NWB files

    Args:
        rec_path (_type_): RecordingExtractor path
        gt_path (_type_): Ground truth SortingExtractor path
        nwb_path (str): write path of nwb file
        params (_type_): nwb file metadata
    """

    # get gain and noise fitted and wired RecordingExtractor
    Recording = si.load_extractor(rec_path)

    # get SortingExtractor
    # remove features that are not handeled because 3D
    Sorting = si.load_extractor(gt_path)
    logger.info("Ground truth metadata:")
    logger.info(Sorting.get_property_keys())

    # create NWB file and add extractors
    nwbfile = init_nwbfile(**params)
    add_recording(recording=Recording, nwbfile=nwbfile)
    add_sorting(sorting=Sorting, nwbfile=nwbfile)

    # save this to a NWB file
    utils.create_if_not_exists(os.path.dirname(nwb_path))
    with NWBHDF5IO(path=nwb_path, mode="w") as io:
        io.write(nwbfile)
        
        
def run_for_vivo(rec_path: str, nwb_path: str,  params: dict):
    """write NWB files

    Args:
        rec_path (_type_): RecordingExtractor path
        gt_path (_type_): Ground truth SortingExtractor path
        nwb_path (str): write path of nwb file
        params (_type_): nwb file metadata
    """

    # get gain and noise fitted and wired RecordingExtractor
    Recording = si.load_extractor(rec_path)

    # create NWB file and add extractors
    nwbfile = init_nwbfile(**params)
    add_recording(recording=Recording, nwbfile=nwbfile)

    # save this to a NWB file
    utils.create_if_not_exists(os.path.dirname(nwb_path))
    with NWBHDF5IO(path=nwb_path, mode="w") as io:
        io.write(nwbfile)