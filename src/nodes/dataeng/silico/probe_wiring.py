"""Node to create in-silico probes and wire them to simulated recordings into a SpikeInterface Recording object

author: steeve.laquitaine@epfl.ch; laquitainesteeve@gmail.com

Returns:
    _type_: _description_
"""

import logging
import logging.config
import shutil
from time import time
import scipy
import MEAutility as MEA
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.full as si
import yaml
import os
from probeinterface import (
    Probe,
    generate_linear_probe,
    generate_multi_shank,
    generate_multi_columns_probe
    )
from sklearn.decomposition import PCA

# custom package
from src.nodes import utils
from src.nodes.dataeng.silico import recording
from src.nodes.dataeng.silico.filtering import (filter_microcircuit_cells,
                                                get_hex_01_cells)
from src.nodes.load import load_campaign_params

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def shift_shank(original_probe, i, d):
    """_summary_

    Args:
        original_probe (_type_): _description_
        i (_type_): _description_
        d (_type_): _description_

    Returns:
        _type_: _description_
    """
    pr = original_probe.copy()
    pr.move([i * d, 0])
    return pr


def run_reyes_8_x_16(dataset_conf: dict, param_conf: dict):
    """Reconstruct and wires probe to lfp trace into a recording
    Loads preprocessed traces from config path and wire probe

    Steps:
    1 - create a probe centered at the origin
    2 - then translate it to the target cell population's centroid,
    3 - align the probe vertical axis along the population's vertical axis.

    Args:
        dataset_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        NumpyRecording: probe-wired SpikeInterface "Recording" object
    """

    # set probe parameters
    Y_PITCH = param_conf["probe"]["reyes_16_x_8"]["y_pitch"]
    X_PITCH = param_conf["probe"]["reyes_16_x_8"]["x_pitch"]
    N_SHANK = param_conf["probe"]["reyes_16_x_8"]["n_shank"]
    N_ELEC = param_conf["probe"]["reyes_16_x_8"]["n_elec"]

    # set read path
    TRACE_FILE_PATH = dataset_conf["dataeng"]["campaign"]["output"][
        "trace_file_path"
    ]

    # get campaign parameters from one simulation
    simulation = load_campaign_params(dataset_conf)

    # get the soma positions of the microcircuit neurons
    # to position probe
    filtered_cells = filter_microcircuit_cells(simulation)

    # create probe
    multi_shank = generate_multi_shank(
        num_shank=N_SHANK, num_columns=1, num_contact_per_column=N_ELEC
    )

    # create probe's contact coordinates
    positions = np.array([0, 0]).reshape(1, 2)
    for i in np.arange(1, 128):
        posX = X_PITCH * int(i / N_ELEC)
        posY = Y_PITCH * (i % N_ELEC)
        newpos = np.array([posX, posY]).reshape(1, 2)
        positions = np.vstack((positions, newpos))
    multi_shank.set_contacts(positions)
    multi_shank = multi_shank.to_3d()

    # rotate the probe around the z axis
    multi_shank.rotate(90, axis=[0, 0, 1])

    # center the probe at the origin (double check)
    multi_shank.move(
        np.array(
            [
                -(X_PITCH * N_SHANK) / 2,
                0,
                -(Y_PITCH * N_ELEC) / 2,
            ]
        )
    )

    # find max axes
    center = np.mean(filtered_cells["soma_location"], axis=0)
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]

    # y-axis
    elevation = np.arctan2(
        np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2), main_axis[2]
    )

    # z axis
    azimuth = np.arctan2(main_axis[1], main_axis[0])

    # set the probe in place for recording
    multi_shank.rotate(elevation * 180 / np.pi, axis=[0, 1, 0])
    multi_shank.rotate(azimuth * 180 / np.pi, axis=[0, 0, 1])
    multi_shank.move(center)

    # set contact ids
    multi_shank.set_device_channel_indices(np.arange(simulation["n_sites"]))

    # read trace
    trace = pd.read_pickle(TRACE_FILE_PATH)

    # cast as a spikeinterface Recording object
    recording = se.NumpyRecording(
        traces_list=[np.array(trace)],
        sampling_frequency=simulation["lfp_sampling_freq"],
    )

    # map traces to probe channels
    return recording.set_probe(multi_shank)


def run_neuropixels_32(data_conf: dict, param_conf: dict):
    """Wires 32-contact 1-shank neuropixel probe to lfp trace into a recording
    Loads preprocessed traces from config path and wire probe

    Args:
        data_conf (dict): chosen data path config from path conf/../dataset.yml
        param_conf (dict): chosen parameter config from path conf/../parameters.yml

    Returns:
        NumpyRecording: probe-wired SpikeInterface "Recording" object
    """
    # get parameters
    CONTACT_IDS = eval(param_conf["probe"]["neuropixels_32"]["contact_ids"])
    n_contacts = len(CONTACT_IDS)

    t0 = time()

    # log
    logger.info("casting raw traces as SpikeInterface Recording object ...")

    # log
    logger.info(
        "casting as raw traces as SpikeInterface Recording object - done in %s",
        round(time() - t0, 1),
    )

    # log
    logger.info("reconstructing neuropixels-32 probe ...")

    t0 = time()

    # get Neuropixels-384 as a source template
    ProbeNeuropix384_2D = MEA.return_mea("Neuropixels-384")
    ProbeNeuropix384_2D_params = MEA.return_mea_info("Neuropixels-384")

    # create a 32-contact linear probe on the y-z plane
    ProbeNeuropix32_2D = generate_linear_probe(
        num_elec=n_contacts,
        ypitch=ProbeNeuropix384_2D_params["pitch"][0],
        contact_shapes=ProbeNeuropix384_2D_params["shape"],
        contact_shape_params={"width": ProbeNeuropix384_2D_params["size"]},
    )
    ProbeNeuropix32 = ProbeNeuropix32_2D.to_3d(axes="yz")

    # translate the Neuropixel-32 probe's 32 contacts at the coordinates of Neuropixels-384 probe's contact ids
    # 127 to 158. These contact ids cover the microcircuit layers relatively well.
    coord_shifts = (
        ProbeNeuropix384_2D.positions[CONTACT_IDS]
        - ProbeNeuropix32.contact_positions
    )
    y_shift = coord_shifts[0, 1]
    z_shift = coord_shifts[0, 2]
    ProbeNeuropix32.move([0, y_shift, z_shift])

    # reset the contact ids and channel ids to start from 0
    reset_contact_ids = np.arange(0, len(CONTACT_IDS), 1)
    ProbeNeuropix32.set_contact_ids(reset_contact_ids)
    ProbeNeuropix32.set_device_channel_indices(reset_contact_ids)

    # test that the target contacts are aligned
    assert (
        sum(
            sum(
                ProbeNeuropix384_2D.positions[CONTACT_IDS]
                - ProbeNeuropix32.contact_positions
            )
        )
        == 0
    ), "target contacts are not aligned"

    # get the campaign parameters from the first simulation
    simulation = load_campaign_params(data_conf)

    # get the seven-column circuit (hex_O1) cells
    hex_O1_cells = get_hex_01_cells(simulation)
    circuit_centroid = np.mean(hex_O1_cells["soma_location"], axis=0).values
    pca = PCA(n_components=3)
    pca.fit(hex_O1_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(
        np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2), main_axis[2]
    )
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    rotation_center = np.array([0, 0, 0])
    ProbeNeuropix32.rotate(
        theta=elevation * 180 / np.pi, center=rotation_center, axis=[0, 1, 0]
    )
    ProbeNeuropix32.rotate(
        theta=azimuth * 180 / np.pi, center=rotation_center, axis=[0, 0, 1]
    )
    ProbeNeuropix32.move(circuit_centroid)

    # log
    logger.info(
        "reconstructing neuropixels-32 probe - done in %s",
        round(time() - t0, 1),
    )

    # wire probe to traces
    logger.info(
        "wiring neuropixels-32 probe to recording - done",
    )
    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)
    trace_recording = recording.load(data_conf)    
    return trace_recording.set_probe(ProbeNeuropix32)


def _run_neuropixels_384_hex_O1(Recording, data_conf: dict, param_conf: dict):
    """Wires 384-contact 1-shank neuropixel 1 ProbeInterface probe to lfp trace
    into a SpikeInterface recording extractor object
    Loads preprocessed traces from config path and wires probe

    Args:
        Recording (SpikeInterface Recording Extractor):
        data_conf (dict): chosen data path config from path conf/../dataset.yml
        param_conf (dict): chosen parameter config from path conf/../parameters.yml

    Returns:
        NumpyRecording: probe-wired SpikeInterface "Recording" object
    """
    # get Neuropixels-384 as a model to replicate
    mea_ProbeNeuropix384_2D = MEA.return_mea("Neuropixels-384")
    mea_ProbeNeuropix384_2D_params = MEA.return_mea_info("Neuropixels-384")

    # set number of columns (n=4 for neuropixels 1)
    N_COLS = mea_ProbeNeuropix384_2D_params["dim"][1]

    # set number of contacts per column (n=96)
    N_CONTACT_PER_COL = mea_ProbeNeuropix384_2D_params["dim"][0]

    # set pitches (y:1 is not intuitive but see [1])
    Y_PITCH = (mea_ProbeNeuropix384_2D_params["pitch"][0],)  # inter-columns
    X_PITCH = (mea_ProbeNeuropix384_2D_params["pitch"][1],)  # inter-row

    # set stagger amount (20)
    STAGGER = mea_ProbeNeuropix384_2D_params["stagger"]

    # create 2D ProbeInterface probe object
    pi_ProbeNeuropix384_2D = generate_multi_columns_probe(
        num_columns=N_COLS,
        num_contact_per_column=N_CONTACT_PER_COL,
        xpitch=X_PITCH,
        ypitch=Y_PITCH,
        y_shift_per_column=[0, STAGGER, 0, STAGGER],
        contact_shapes=mea_ProbeNeuropix384_2D_params["shape"],
        contact_shape_params={"width": mea_ProbeNeuropix384_2D_params["size"]},
    )

    # make 3D
    pi_ProbeNeuropix384_3D = pi_ProbeNeuropix384_2D.to_3d(axes="yz")

    # ProbeInterface's probe first bottom contact is by default at coordinate (0,0,0) while
    # MEA's probe center is at (0,0,0). We need to align the two.
    # translate probeinterface's y and z to align both probes first bottom contact coordinates

    # get mea's first bottom contact y, z coordinates
    mea_contact_y = mea_ProbeNeuropix384_2D.positions[0, 1]
    mea_contact_z = mea_ProbeNeuropix384_2D.positions[0, 2]

    # get probeinterface's first bottom contact y, z coordinates
    pi_contact_y = pi_ProbeNeuropix384_3D.contact_positions[0, 1]
    pi_contact_z = pi_ProbeNeuropix384_3D.contact_positions[0, 2]

    # calculate translation
    y_shift = mea_contact_y - pi_contact_y
    z_shift = mea_contact_z - pi_contact_z

    # translate
    pi_ProbeNeuropix384_3D.move([0, y_shift, z_shift])

    # sanity check alignment
    assert (
        np.sum(
            pi_ProbeNeuropix384_3D.contact_positions
            - mea_ProbeNeuropix384_2D.positions
        )
        == 0
    ), "mea and pi do not match"

    # count contacts
    n_contacts = pi_ProbeNeuropix384_3D.get_contact_count()

    # set the contact ids and channel ids to start from 0
    pi_ProbeNeuropix384_3D.set_contact_ids(np.arange(n_contacts))
    pi_ProbeNeuropix384_3D.set_device_channel_indices(np.arange(n_contacts))

    # get the campaign parameters from the first simulation
    simulation = load_campaign_params(data_conf)

    # get the microcircuit (hex0) cells to position the probe at its centroid
    filtered_cells = get_hex_01_cells(simulation)
    circuit_centroid = np.mean(filtered_cells["soma_location"], axis=0).values
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(
        np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2), main_axis[2]
    )
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    rotation_center = np.array([0, 0, 0])

    # rotate
    pi_ProbeNeuropix384_3D.rotate(
        theta=elevation * 180 / np.pi, center=rotation_center, axis=[0, 1, 0]
    )

    # rotate
    pi_ProbeNeuropix384_3D.rotate(
        theta=azimuth * 180 / np.pi, center=rotation_center, axis=[0, 0, 1]
    )

    # translate
    pi_ProbeNeuropix384_3D.move(circuit_centroid)

    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)
    return Recording.set_probe(pi_ProbeNeuropix384_3D)


def wire_silico_marques_probe(data_conf: dict, recording):
    """Wires 384-contact 1-shank neuropixel 1 ProbeInterface probe to lfp trace
    into a SpikeInterface recording extractor object as our best reconstruction of 
    Marques probe
    Loads preprocessed traces from config path and wires probe

    Args:
        data_conf (dict): chosen data path config from path conf/../dataset.yml
        param_conf (dict): chosen parameter config from path conf/../parameters.yml

    Returns:
        NumpyRecording: probe-wired SpikeInterface "Recording" object
    """
    # get Neuropixels-384 as a model to replicate
    mea_ProbeNeuropix384_2D = MEA.return_mea("Neuropixels-384")
    mea_ProbeNeuropix384_2D_params = MEA.return_mea_info("Neuropixels-384")

    # set number of columns (n=4 for neuropixels 1)
    N_COLS = mea_ProbeNeuropix384_2D_params["dim"][1]

    # set number of contacts per column (n=96)
    N_CONTACT_PER_COL = mea_ProbeNeuropix384_2D_params["dim"][0]

    # set pitches (y:1 is not intuitive but see [1])
    Y_PITCH = (mea_ProbeNeuropix384_2D_params["pitch"][0],)  # inter-columns
    X_PITCH = (mea_ProbeNeuropix384_2D_params["pitch"][1],)  # inter-row

    # set stagger amount (20)
    STAGGER = mea_ProbeNeuropix384_2D_params["stagger"]

    # create 2D ProbeInterface probe object
    pi_ProbeNeuropix384_2D = generate_multi_columns_probe(
        num_columns=N_COLS,
        num_contact_per_column=N_CONTACT_PER_COL,
        xpitch=X_PITCH,
        ypitch=Y_PITCH,
        y_shift_per_column=[0, STAGGER, 0, STAGGER],
        contact_shapes=mea_ProbeNeuropix384_2D_params["shape"],
        contact_shape_params={"width": mea_ProbeNeuropix384_2D_params["size"]},
    )

    # make 3D
    pi_ProbeNeuropix384_3D = pi_ProbeNeuropix384_2D.to_3d(axes="yz")

    # ProbeInterface's probe first bottom contact is by default at coordinate (0,0,0) while
    # MEA's probe center is at (0,0,0). We need to align the two.
    # translate probeinterface's y and z to align both probes first bottom contact coordinates

    # get mea's first bottom contact y, z coordinates
    mea_contact_y = mea_ProbeNeuropix384_2D.positions[0, 1]
    mea_contact_z = mea_ProbeNeuropix384_2D.positions[0, 2]

    # get probeinterface's first bottom contact y, z coordinates
    pi_contact_y = pi_ProbeNeuropix384_3D.contact_positions[0, 1]
    pi_contact_z = pi_ProbeNeuropix384_3D.contact_positions[0, 2]

    # calculate translation
    y_shift = mea_contact_y - pi_contact_y
    z_shift = mea_contact_z - pi_contact_z

    # translate
    pi_ProbeNeuropix384_3D.move([0, y_shift, z_shift])

    # sanity check alignment
    assert (
        np.sum(
            pi_ProbeNeuropix384_3D.contact_positions
            - mea_ProbeNeuropix384_2D.positions
        )
        == 0
    ), "mea and pi do not match"

    # count contacts
    n_contacts = pi_ProbeNeuropix384_3D.get_contact_count()

    # set the contact ids and channel ids to start from 0
    pi_ProbeNeuropix384_3D.set_contact_ids(np.arange(n_contacts))
    pi_ProbeNeuropix384_3D.set_device_channel_indices(np.arange(n_contacts))

    # get the campaign parameters from the first simulation
    simulation = load_campaign_params(data_conf)

    # get the microcircuit (hex0) cells to position the probe at its centroid
    filtered_cells = get_hex_01_cells(simulation)
    circuit_centroid = np.mean(filtered_cells["soma_location"], axis=0).values
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(
        np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2), main_axis[2]
    )
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    rotation_center = np.array([0, 0, 0])

    # rotate
    pi_ProbeNeuropix384_3D.rotate(
        theta=elevation * 180 / np.pi, center=rotation_center, axis=[0, 1, 0]
    )

    # rotate
    pi_ProbeNeuropix384_3D.rotate(
        theta=azimuth * 180 / np.pi, center=rotation_center, axis=[0, 0, 1]
    )

    # translate
    pi_ProbeNeuropix384_3D.move(circuit_centroid)

    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)
    return recording.set_probe(pi_ProbeNeuropix384_3D)


def wire_silico_horvath_probe_1(data_conf: dict, param_conf:dict, recording):
    """create in silico horvath probe at depth 1

    Args:
        data_conf (dict): _description_
        params (dict): _description_
        recording (): _description_

    Returns:
        _type_: _description_
    """
    # get parameters
    CONTACT_IDS = eval(param_conf["probe"]["contact_ids"])

    # create 2d probeinterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=param_conf["probe"]["n_cols"],
        num_contact_per_column=param_conf["probe"]["n_contact_per_col"],
        xpitch=param_conf["probe"]["x_pitch"],
        ypitch=param_conf["probe"]["y_pitch"],
        y_shift_per_column=[0, param_conf["probe"]["stagger"], 0, param_conf["probe"]["stagger"]],
        contact_shapes=param_conf["probe"]["shape"],
        contact_shape_params={"width": param_conf["probe"]["width"]},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis too at 0
    Probe.rotate(90, axis=[0, 0, 1])
    Probe.move(
        np.array([-(param_conf["probe"]["x_pitch"]*(param_conf["probe"]["n_cols"]-1))/2,
        0,
        -(param_conf["probe"]["y_pitch"]*(param_conf["probe"]["n_contact_per_col"]-1))/2])
        )

    # set contact ids
    Probe.set_contact_ids(CONTACT_IDS)
    Probe.set_device_channel_indices(CONTACT_IDS)

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    # get the campaign parameters from the first simulation
    simulation = load_campaign_params(data_conf)
    filtered_cells = get_hex_01_cells(simulation)
    circuit_centroid = np.mean(filtered_cells["soma_location"], axis=0).values

    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2), main_axis[2])
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    Probe.rotate(elevation*180/np.pi, axis=[0, 1, 0])
    Probe.rotate(azimuth*180/np.pi, axis=[0, 0, 1])
    Probe.move(circuit_centroid)

    # position the probe at depth 1 that mimics 
    # the layer coverage of Horvath probe at depth 1
    # rotate around circuit/probe center then translate it
    SHIFT_FROM_CENTER_1 = main_axis*850
    Probe.rotate(0.62*180/np.pi, axis=[0, 1, 0])
    Probe.rotate(0.2*azimuth*180/np.pi, axis=[0, 0, 1])
    Probe.move(SHIFT_FROM_CENTER_1)
    
    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)
    return recording.set_probe(Probe)


def wire_silico_horvath_probe_2(data_conf: dict, param_conf: dict, recording):
    """create in silico horvath probe at depth 2

    Args:
        data_conf (dict): _description_
        params (dict): _description_
        recording (dict): _description_

    Returns:
        _type_: _description_
    """
    # get parameters
    CONTACT_IDS = eval(param_conf["probe"]["contact_ids"])

    # create 2d probeinterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=param_conf["probe"]["n_cols"],
        num_contact_per_column=param_conf["probe"]["n_contact_per_col"],
        xpitch=param_conf["probe"]["x_pitch"],
        ypitch=param_conf["probe"]["y_pitch"],
        y_shift_per_column=[0, param_conf["probe"]["stagger"], 0, param_conf["probe"]["stagger"]],
        contact_shapes=param_conf["probe"]["shape"],
        contact_shape_params={"width": param_conf["probe"]["width"]},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis too at 0
    Probe.rotate(90, axis=[0, 0, 1])
    Probe.move(
        np.array([-(param_conf["probe"]["x_pitch"]*(param_conf["probe"]["n_cols"]-1))/2,
        0,
        -(param_conf["probe"]["y_pitch"]*(param_conf["probe"]["n_contact_per_col"]-1))/2])
        )

    # set contact ids
    Probe.set_contact_ids(CONTACT_IDS)
    Probe.set_device_channel_indices(CONTACT_IDS)

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    # get the campaign parameters from the first simulation
    simulation = load_campaign_params(data_conf)
    filtered_cells = get_hex_01_cells(simulation)
    circuit_centroid = np.mean(filtered_cells["soma_location"], axis=0).values

    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2), main_axis[2])
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    Probe.rotate(elevation*180/np.pi, axis=[0, 1, 0])
    Probe.rotate(azimuth*180/np.pi, axis=[0, 0, 1])
    Probe.move(circuit_centroid)

    # position the probe at depth 1 that mimics 
    # the layer coverage of Horvath probe at depth 1
    # rotate around circuit/probe center then translate it
    SHIFT_FROM_CENTER_1 = main_axis*850
    SHIFT_FROM_DEPTH_1 = - main_axis*800
    Probe.move(SHIFT_FROM_CENTER_1 + SHIFT_FROM_DEPTH_1)
    
    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)
    return recording.set_probe(Probe)


def wire_silico_horvath_probe_3(data_conf: dict, param_conf: dict, recording):
    """create in silico horvath probe at depth 3

    Args:
        data_conf (dict): _description_
        params (dict): _description_
        recording (dict): _description_

    Returns:
        _type_: _description_
    """
    # get parameters
    CONTACT_IDS = eval(param_conf["probe"]["contact_ids"])

    # create 2d probeinterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=param_conf["probe"]["n_cols"],
        num_contact_per_column=param_conf["probe"]["n_contact_per_col"],
        xpitch=param_conf["probe"]["x_pitch"],
        ypitch=param_conf["probe"]["y_pitch"],
        y_shift_per_column=[0, param_conf["probe"]["stagger"], 0, param_conf["probe"]["stagger"]],
        contact_shapes=param_conf["probe"]["shape"],
        contact_shape_params={"width": param_conf["probe"]["width"]},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis too at 0
    Probe.rotate(90, axis=[0, 0, 1])
    Probe.move(
        np.array([-(param_conf["probe"]["x_pitch"]*(param_conf["probe"]["n_cols"]-1))/2,
        0,
        -(param_conf["probe"]["y_pitch"]*(param_conf["probe"]["n_contact_per_col"]-1))/2])
        )

    # set contact ids
    Probe.set_contact_ids(CONTACT_IDS)
    Probe.set_device_channel_indices(CONTACT_IDS)

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    # get the campaign parameters from the first simulation
    simulation = load_campaign_params(data_conf)
    filtered_cells = get_hex_01_cells(simulation)
    circuit_centroid = np.mean(filtered_cells["soma_location"], axis=0).values

    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2), main_axis[2])
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    Probe.rotate(elevation*180/np.pi, axis=[0, 1, 0])
    Probe.rotate(azimuth*180/np.pi, axis=[0, 0, 1])
    Probe.move(circuit_centroid)

    # position the probe at depth 3 that mimics 
    # the layer coverage of Horvath probe at depth 3
    # rotate around circuit/probe center then translate it
    SHIFT_FROM_CENTER_1 = main_axis*850
    SHIFT_FROM_DEPTH_1 = - main_axis*800
    SHIFT_FROM_DEPTH_2 = - main_axis*1000
    Probe.move(SHIFT_FROM_CENTER_1 + SHIFT_FROM_DEPTH_1 + SHIFT_FROM_DEPTH_2)
    
    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)
    return recording.set_probe(Probe)


def run_vivo_reyes_1_x_16(data_conf:dict, param_conf:dict):
    """wire the 1 shank 16 channels probe used in Reyes 2015

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """
    # get probe parameters
    MAPPING = eval(param_conf["S001E009F002_Raw"]["probe"]["contact_mapping"])
    N_CONTACTS = param_conf["S001E009F002_Raw"]["probe"]["n_contacts"]
    CONTACT_SHAPES = param_conf["S001E009F002_Raw"]["probe"]["contact_shapes"]
    CONTACT_SHAPE_PARAMS=param_conf["S001E009F002_Raw"]["probe"]["contact_shape_params"]
    PROBE_TYPE=param_conf["S001E009F002_Raw"]["probe"]["probe_type"]
    MARGIN=param_conf["S001E009F002_Raw"]["probe"]["margin"]
    YPITCH=param_conf["S001E009F002_Raw"]["probe"]["ypitch"]

    # map contacts
    mapping = MAPPING - min(MAPPING)

    # construct probe
    probe = generate_linear_probe(num_elec=N_CONTACTS, 
                                  ypitch=YPITCH, 
                                  contact_shapes=CONTACT_SHAPES, 
                                  contact_shape_params=CONTACT_SHAPE_PARAMS
                                  )
    probe.create_auto_shape(probe_type=PROBE_TYPE, margin=MARGIN)
    probe.set_device_channel_indices(mapping)

    # wire
    Recording = recording.load(data_conf)
    return Recording.set_probe(probe, in_place=True)


def run_silico_horvath_1(Recording, data_conf: dict, param_conf: dict, load_filtered_cells_metadata=True):
    """create in silico horvath probe at depth 1

    Args:
        data_conf (dict): _description_
        params (dict): _description_

    Returns:
        _type_: _description_
    """
    # get parameters
    CONTACT_IDS = eval(param_conf["probe"]["contact_ids"])

    # create 2d probeinterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=param_conf["probe"]["n_cols"],
        num_contact_per_column=param_conf["probe"]["n_contact_per_col"],
        xpitch=param_conf["probe"]["x_pitch"],
        ypitch=param_conf["probe"]["y_pitch"],
        y_shift_per_column=[0, param_conf["probe"]["stagger"], 0, param_conf["probe"]["stagger"]],
        contact_shapes=param_conf["probe"]["shape"],
        contact_shape_params={"width": param_conf["probe"]["width"]},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis too at 0
    Probe.rotate(90, axis=[0, 0, 1])
    Probe.move(
        np.array([-(param_conf["probe"]["x_pitch"]*(param_conf["probe"]["n_cols"]-1))/2,
        0,
        -(param_conf["probe"]["y_pitch"]*(param_conf["probe"]["n_contact_per_col"]-1))/2])
        )

    # set contact ids
    Probe.set_contact_ids(CONTACT_IDS)
    Probe.set_device_channel_indices(CONTACT_IDS)

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    # get the campaign parameters from the first simulation
    # load/save filtered cells metadata
    if load_filtered_cells_metadata:
        filtered_cells = np.load(data_conf["metadata"]["filtered_cells"], allow_pickle=True).item()
    else:
        simulation = load_campaign_params(data_conf)
        filtered_cells = get_hex_01_cells(simulation)
        np.save(data_conf["metadata"]["filtered_cells"], filtered_cells)
    circuit_centroid = np.mean(filtered_cells["soma_location"], axis=0).values

    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2), main_axis[2])
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    Probe.rotate(elevation*180/np.pi, axis=[0, 1, 0])
    Probe.rotate(azimuth*180/np.pi, axis=[0, 0, 1])
    Probe.move(circuit_centroid)

    # position the probe at depth 1 that mimics 
    # the layer coverage of Horvath probe at depth 1
    # rotate around circuit/probe center then translate it
    SHIFT_FROM_CENTER_1 = main_axis*850
    Probe.rotate(0.62*180/np.pi, axis=[0, 1, 0])
    Probe.rotate(0.2*azimuth*180/np.pi, axis=[0, 0, 1])
    Probe.move(SHIFT_FROM_CENTER_1)
    
    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)
    return Recording.set_probe(Probe)


def run_silico_horvath_2(Recording, data_conf: dict, param_conf: dict, load_filtered_cells_metadata=True):
    """create in silico horvath probe at depth 2

    Args:
        data_conf (dict): _description_
        params (dict): _description_

    Returns:
        _type_: _description_
    """
    # get parameters
    CONTACT_IDS = eval(param_conf["probe"]["contact_ids"])

    # create 2d probeinterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=param_conf["probe"]["n_cols"],
        num_contact_per_column=param_conf["probe"]["n_contact_per_col"],
        xpitch=param_conf["probe"]["x_pitch"],
        ypitch=param_conf["probe"]["y_pitch"],
        y_shift_per_column=[0, param_conf["probe"]["stagger"], 0, param_conf["probe"]["stagger"]],
        contact_shapes=param_conf["probe"]["shape"],
        contact_shape_params={"width": param_conf["probe"]["width"]},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis too at 0
    Probe.rotate(90, axis=[0, 0, 1])
    Probe.move(
        np.array([-(param_conf["probe"]["x_pitch"]*(param_conf["probe"]["n_cols"]-1))/2,
        0,
        -(param_conf["probe"]["y_pitch"]*(param_conf["probe"]["n_contact_per_col"]-1))/2])
        )

    # set contact ids
    Probe.set_contact_ids(CONTACT_IDS)
    Probe.set_device_channel_indices(CONTACT_IDS)

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    # get the campaign parameters from the first simulation
    if load_filtered_cells_metadata:
        filtered_cells = np.load(data_conf["metadata"]["filtered_cells"], filtered_cells, allow_pickle=True).item()
    else:
        simulation = load_campaign_params(data_conf)
        filtered_cells = get_hex_01_cells(simulation)
        # save
        utils.create_if_not_exists(os.path.dirname(data_conf["metadata"]["filtered_cells"]))
        np.save(data_conf["metadata"]["filtered_cells"], filtered_cells)
        
    circuit_centroid = np.mean(filtered_cells["soma_location"], axis=0).values

    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2), main_axis[2])
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    Probe.rotate(elevation*180/np.pi, axis=[0, 1, 0])
    Probe.rotate(azimuth*180/np.pi, axis=[0, 0, 1])
    Probe.move(circuit_centroid)

    # position the probe at depth 1 that mimics 
    # the layer coverage of Horvath probe at depth 1
    # rotate around circuit/probe center then translate it
    SHIFT_FROM_CENTER_1 = main_axis*850
    SHIFT_FROM_DEPTH_1 = - main_axis*800
    Probe.move(SHIFT_FROM_CENTER_1 + SHIFT_FROM_DEPTH_1)
    
    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)    
    return Recording.set_probe(Probe)


def run_silico_horvath_3(Recording, data_conf: dict, param_conf: dict, load_filtered_cells_metadata=True):
    """create in silico horvath probe at depth 3

    Args:
        data_conf (dict): _description_
        params (dict): _description_

    Returns:
        _type_: _description_
    """
    # get parameters
    CONTACT_IDS = eval(param_conf["probe"]["contact_ids"])

    # create 2d probeinterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=param_conf["probe"]["n_cols"],
        num_contact_per_column=param_conf["probe"]["n_contact_per_col"],
        xpitch=param_conf["probe"]["x_pitch"],
        ypitch=param_conf["probe"]["y_pitch"],
        y_shift_per_column=[0, param_conf["probe"]["stagger"], 0, param_conf["probe"]["stagger"]],
        contact_shapes=param_conf["probe"]["shape"],
        contact_shape_params={"width": param_conf["probe"]["width"]},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis too at 0
    Probe.rotate(90, axis=[0, 0, 1])
    Probe.move(
        np.array([-(param_conf["probe"]["x_pitch"]*(param_conf["probe"]["n_cols"]-1))/2,
        0,
        -(param_conf["probe"]["y_pitch"]*(param_conf["probe"]["n_contact_per_col"]-1))/2])
        )

    # set contact ids
    Probe.set_contact_ids(CONTACT_IDS)
    Probe.set_device_channel_indices(CONTACT_IDS)

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    # get the campaign parameters from the first simulation
    if load_filtered_cells_metadata:
        filtered_cells = np.load(data_conf["metadata"]["filtered_cells"], filtered_cells, allow_pickle=True).item()
    else:
        simulation = load_campaign_params(data_conf)
        filtered_cells = get_hex_01_cells(simulation)
        # save
        utils.create_if_not_exists(os.path.dirname(data_conf["metadata"]["filtered_cells"]))
        np.save(data_conf["metadata"]["filtered_cells"], filtered_cells)
    circuit_centroid = np.mean(filtered_cells["soma_location"], axis=0).values
    
    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(filtered_cells["soma_location"])
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2), main_axis[2])
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    Probe.rotate(elevation*180/np.pi, axis=[0, 1, 0])
    Probe.rotate(azimuth*180/np.pi, axis=[0, 0, 1])
    Probe.move(circuit_centroid)

    # position the probe at depth 3 that mimics 
    # the layer coverage of Horvath probe at depth 3
    # rotate around circuit/probe center then translate it
    SHIFT_FROM_CENTER_1 = main_axis*850
    SHIFT_FROM_DEPTH_1 = - main_axis*800
    SHIFT_FROM_DEPTH_2 = - main_axis*1000
    Probe.move(SHIFT_FROM_CENTER_1 + SHIFT_FROM_DEPTH_1 + SHIFT_FROM_DEPTH_2)
    
    # cast as SpikeInterface Recording object
    # should have been casted before with recording.run(..)    
    return Recording.set_probe(Probe)


def run_vivo_marques(Recording, data_conf: dict):
    """reconstruct and wire Marques probe to recording traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """
    # get dataset paths
    CHANNEL_MAP_FILE = data_conf["channel_map"]

    # get Neuropixels-384 parameters
    npx_prms = MEA.return_mea_info("Neuropixels-384")

    # get the channel locations from the Recording provided
    # by the authors' channel map file and channel locations
    channel_map = scipy.io.loadmat(CHANNEL_MAP_FILE)
    locations = np.zeros((384, 2))
    locations[:, 0] = channel_map["xcoords"][:, 0]
    locations[:, 1] = channel_map["ycoords"][:, 0]

    # create 2D probe with the locations
    ProbeVivo = Probe(ndim=2, si_units="um")
    ProbeVivo.set_contacts(
        positions=locations,
        shapes=npx_prms["shape"],
        shape_params={
            "width": npx_prms["size"],
        },
    )
    ProbeVivo.create_auto_shape(probe_type="tip")

    # set contact and channel ids (mapped to the rows of the locations array)
    ProbeVivo.set_contact_ids(np.arange(0, 384, 1))  # default order in Kilosort
    ProbeVivo.set_device_channel_indices(Recording.get_channel_ids())

    # wire probe
    return Recording.set_probe(ProbeVivo)


def run_vivo_marques_old(data_conf: dict, param_conf: dict):
    """reconstruct and wire Marques probe to recording traces

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        _type_: _description_
    """
    # get Neuropixels-384 as a model to replicate
    mea_ProbeNeuropix384_2D_params = MEA.return_mea_info("Neuropixels-384")

    # set pitches (y:1 is not intuitive but see [1])
    Y_PITCH = (mea_ProbeNeuropix384_2D_params["pitch"][0],)  # inter-columns
    X_PITCH = (mea_ProbeNeuropix384_2D_params["pitch"][1],)  # inter-row
    STAGGER = mea_ProbeNeuropix384_2D_params["stagger"]

    # create 2D ProbeInterface probe object
    probe = generate_multi_columns_probe(
        num_columns=param_conf["probe"]["n_columns"],
        num_contact_per_column=param_conf["probe"]["n_contacts_per_column"],
        xpitch=X_PITCH,
        ypitch=Y_PITCH,
        y_shift_per_column=[0, STAGGER, 0, STAGGER],
        contact_shapes=mea_ProbeNeuropix384_2D_params["shape"],
        contact_shape_params={"width": mea_ProbeNeuropix384_2D_params["size"]},
    )

    # make 3D
    probe = probe.to_3d(axes="yz")

    # set the contact ids and channel ids to start from 0
    # create Marques site - mapping
    contact_ids = np.vstack([
        np.arange(0,384,4),
        np.arange(2,384,4),
        np.arange(1,384,4),
        np.arange(3,384,4)
        ]).reshape(1,384).squeeze()

    probe.set_contact_ids(contact_ids)
    probe.set_device_channel_indices(contact_ids)

    # cast as SpikeInterface Recording object
    # should have been casted beforehand
    trace_recording = recording.load(data_conf)
    return trace_recording.set_probe(probe)


def run(Recording, data_conf: dict, param_conf: dict, load_filtered_cells_metadata=True):
    """Wire the recording with the probe configured in
    conf/../parameters.yml

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        NumpyRecording: probe-wired SpikeInterface "Recording" object
    """
    if param_conf["run"]["probe"] == "neuropixels_32":
        logger.info("probe: neuropixels_32")
        return run_neuropixels_32(data_conf, param_conf)
    if param_conf["run"]["probe"] == "neuropixels_384_hex_O1":
        logger.info("probe: neuropixels_384_hex_O1")
        return _run_neuropixels_384_hex_O1(Recording, data_conf, param_conf)    
    if param_conf["run"]["probe"] == "reyes_128":
        logger.info("probe: reyes_128")
    if param_conf["run"]["probe"] == "vivo_reyes":
        logger.info("probe: vivo_reyes")
        return run_vivo_reyes_1_x_16(data_conf, param_conf)
    if param_conf["run"]["probe"] == "silico_horvath_probe_1":
        logger.info("probe: silico_horvath_probe_1")
        return run_silico_horvath_1(Recording, data_conf, param_conf, load_filtered_cells_metadata=load_filtered_cells_metadata)
    if param_conf["run"]["probe"] == "silico_horvath_probe_2":
        logger.info("probe: silico_horvath_probe_2")
        return run_silico_horvath_2(Recording, data_conf, param_conf, load_filtered_cells_metadata=load_filtered_cells_metadata)
    if param_conf["run"]["probe"] == "silico_horvath_probe_3":
        logger.info("probe: silico_horvath_probe_3")
        return run_silico_horvath_3(Recording, data_conf, param_conf, load_filtered_cells_metadata=load_filtered_cells_metadata)
    if param_conf["run"]["probe"] == "vivo_marques":
        logger.info("probe: vivo_marques")
        return run_vivo_marques(Recording, data_conf)
    else:
        raise NotImplementedError("""This probe was not implemented. 
                                  Implement it in src/nodes/dataeng/silico/probe_wiring.py""")
    

def write(Recording, data_conf: dict, job_dict: dict):
    """Write probe-wired SpikeInterface "Recording" Extractor

    Args:
        Recording (_type_): _description_
        data_conf (dict): _description_

    Note: 
        The max number of jobs is limited by the number of CPU cores
        Our nodes have 8 cores.
        n_jobs=8 and total_memory=2G sped up writing by a factor a 2X
        no improvement was observed for larger memory allocation
    """
    # track time
    t0 = time()

    # get write path
    WRITE_PATH = data_conf["probe_wiring"]["output"]

    # write (parallel processing works for 10 min recordings, else use 1 node for 1h recording otherwise
    # you get "out of memory error: "slurmstepd: error: Detected 50 oom-kill event(s). 
    # Some of your processes may have been killed by the cgroup out-of-memory handler."")
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Recording.save(folder=WRITE_PATH, format="binary", **job_dict)

    # log
    logger.info("Probe wiring done in  %s secs", round(time() - t0, 1))
