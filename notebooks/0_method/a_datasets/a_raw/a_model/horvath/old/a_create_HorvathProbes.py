"""Create Horvath probes at depth 1, 2,3 

Returns:
    _type_: _description_
"""
import bluepy as bp
from probeinterface import generate_multi_columns_probe
from sklearn.decomposition import PCA
import numpy as np

def create_horvath_probe_depth_1(
        path_to_blueconfig: str,
        n_cols: int = 4,
        n_contact_per_col: int = 32,
        x_pitch: float = 22.5,
        y_pitch: float = 22.5,
        stagger: float = 0,
        shape: str = "square",
        width: float = 20
        ):

    # create 2d probeinterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=n_cols,
        num_contact_per_column=n_contact_per_col,
        xpitch=x_pitch,
        ypitch=y_pitch,
        y_shift_per_column=[0, stagger, 0, stagger],
        contact_shapes=shape,
        contact_shape_params={"width": width},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis too at 0
    Probe.rotate(90, axis=[0,0,1])
    Probe.move(np.array([-(x_pitch*(n_cols-1))/2,0,-(y_pitch*(n_contact_per_col-1))/2]))

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    simulation = bp.Simulation(path_to_blueconfig)
    circuit = simulation.circuit
    soma_location = circuit.cells.get(
        {"$target": "hex_O1"}, properties=[bp.Cell.X, bp.Cell.Y, bp.Cell.Z]
    )
    circuit_centroid = np.mean(soma_location, axis=0).values

    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(soma_location)
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2),main_axis[2])
    azimuth = np.arctan2(main_axis[1],main_axis[0])
    Probe.rotate(elevation*180/np.pi,axis=[0,1,0])
    Probe.rotate(azimuth*180/np.pi,axis=[0,0,1])
    Probe.move(circuit_centroid)

    # position the probe at depth 2 that mimics 
    # the layer coverage of Horvath probe at depth 2
    # rotate around circuit/probe center then translate it
    SHIFT_FROM_CENTER_1 = main_axis*850
    Probe.rotate(0.62*180/np.pi,axis=[0,1,0])
    Probe.rotate(0.2*azimuth*180/np.pi,axis=[0,0,1])
    Probe.move(SHIFT_FROM_CENTER_1)
    return Probe

def create_horvath_probe_depth_2(
        path_to_blueconfig: str,
        n_cols: int = 4,
        n_contact_per_col: int = 32,
        x_pitch: float = 22.5,
        y_pitch: float = 22.5,
        stagger: float = 0,
        shape: str = "square",
        width: float = 20
        ):

    # create 2d probeinterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=n_cols,
        num_contact_per_column=n_contact_per_col,
        xpitch=x_pitch,
        ypitch=y_pitch,
        y_shift_per_column=[0, stagger, 0, stagger],
        contact_shapes=shape,
        contact_shape_params={"width": width},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis too at 0
    Probe.rotate(90, axis=[0, 0, 1])
    Probe.move(np.array([-(x_pitch*(n_cols-1))/2, 0,-(y_pitch*(n_contact_per_col-1))/2]))

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    simulation = bp.Simulation(path_to_blueconfig)
    circuit = simulation.circuit
    soma_location = circuit.cells.get(
        {"$target": "hex_O1"}, properties=[bp.Cell.X, bp.Cell.Y, bp.Cell.Z]
    )
    circuit_centroid = np.mean(soma_location, axis=0).values

    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(soma_location)
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2),main_axis[2])
    azimuth = np.arctan2(main_axis[1],main_axis[0])
    Probe.rotate(elevation*180/np.pi,axis=[0,1,0])
    Probe.rotate(azimuth*180/np.pi,axis=[0,0,1])
    Probe.move(circuit_centroid)

    # rotate at center then translate
    SHIFT_FROM_CENTER_1 = main_axis*850
    SHIFT_FROM_DEPTH_1 = - main_axis*800
    Probe.move(SHIFT_FROM_CENTER_1 + SHIFT_FROM_DEPTH_1)
    return Probe

def create_horvath_probe_depth_3(
        path_to_blueconfig: str,
        n_cols: int = 4,
        n_contact_per_col: int = 32,
        x_pitch: float = 22.5,
        y_pitch: float = 22.5,
        stagger: float = 0,
        shape: str = "square",
        width: float = 20
        ):

    # create 2D ProbeInterface probe object
    CenteredProbe = generate_multi_columns_probe(
        num_columns=n_cols,
        num_contact_per_column=n_contact_per_col,
        xpitch=x_pitch,
        ypitch=y_pitch,
        y_shift_per_column=[0, stagger, 0, stagger],
        contact_shapes=shape,
        contact_shape_params={"width": width},
    )

    # make 3D
    Probe = CenteredProbe.to_3d()

    # center probe at (0,0,0)
    # - align probe with yz plane
    # - center probe's z axis at 0
    Probe.rotate(90, axis=[0,0,1])
    Probe.move(np.array([-(x_pitch*(n_cols-1))/2,0,-(y_pitch*(n_contact_per_col-1))/2]))

    # get the campaign parameters from one of the simulation
    # get the microcircuit (hex_01) cells to position the probe at its centroid
    # calculate circuit's centroid by averaging of cell soma coordinates
    simulation = bp.Simulation(path_to_blueconfig)
    circuit = simulation.circuit
    soma_location = circuit.cells.get(
        {"$target": "hex_O1"}, properties=[bp.Cell.X, bp.Cell.Y, bp.Cell.Z]
    )
    circuit_centroid = np.mean(soma_location, axis=0).values

    # rotate and center probe to circuit center
    pca = PCA(n_components=3)
    pca.fit(soma_location)
    main_axis = pca.components_[0]
    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2),main_axis[2])
    azimuth = np.arctan2(main_axis[1],main_axis[0])
    Probe.rotate(elevation*180/np.pi,axis=[0,1,0])
    Probe.rotate(azimuth*180/np.pi,axis=[0,0,1])
    Probe.move(circuit_centroid)

    # rotate at center then translate along probe axis
    SHIFT_FROM_CENTER_1 = main_axis*850
    SHIFT_FROM_DEPTH_1 = - main_axis*800
    SHIFT_FROM_DEPTH_2 = - main_axis*1000
    Probe.move(SHIFT_FROM_CENTER_1 + SHIFT_FROM_DEPTH_1 + SHIFT_FROM_DEPTH_2)
    return Probe

# Create Horvath probe at depth 1
HorvathProbe1 = create_horvath_probe_depth_1(
    path_to_blueconfig='/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw/neuropixels_lfp_10m_384ch_hex0_rou03_pfr09_40Khz_2023_09_19/37cf900a-35ed-42fb-beb6-5da4a37d3aa3/0/BlueConfig',
)

# Create Horvath probe at depth 2
HorvathProbe2 = create_horvath_probe_depth_2(
    path_to_blueconfig='/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw/neuropixels_lfp_10m_384ch_hex0_rou03_pfr09_40Khz_2023_09_19/37cf900a-35ed-42fb-beb6-5da4a37d3aa3/0/BlueConfig',
)

# Create Horvath probe at depth 3
HorvathProbe3 = create_horvath_probe_depth_3(
    path_to_blueconfig='/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw/neuropixels_lfp_10m_384ch_hex0_rou03_pfr09_40Khz_2023_09_19/37cf900a-35ed-42fb-beb6-5da4a37d3aa3/0/BlueConfig',
)
