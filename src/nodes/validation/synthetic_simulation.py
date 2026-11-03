"""Simulation of synthetic recordings with MEArec (Supplementary Fig. 6c-e)

Functions copied from notebooks/1_results/supp_fig6_CD_pros_of_added_details_yield_revised.ipynb.
MEArec is imported where needed (use the environment envs/mearec_si_0_103_0).
"""

import json
import os
import numpy as np


def set_metadata_to_mearec_simulations(recording_generator, Recording, Sorting):
    
    # set cell features
    features = ['cell_type', 'snr', 'mtype', 'soma_position', 'bursting', 'exp_decay', 'max_spikes_per_burst', 'max_burst_duration']
    n_cells = len(recording_generator.spiketrains)

    # set cell properties in sorting extractor
    for feature in features:
        feature_data = []
        for cell_ix in range(n_cells):
            feature_data.append(recording_generator.spiketrains[cell_ix].annotations[feature])
        Sorting.set_property(feature, np.array(feature_data))

    # set electrode properties in recording extractor
    # by default, channel_locations are attached to Recording extractor
    Recording.set_property('layer', np.array(['L5']*384)) # layer
    return Recording, Sorting


def standardize_metadata(GroundTruth):
    """standardize ground truth extractor metadata
    from mearec simulatio

    Args:
        GroundTruth (_type_): _description_

    Returns:
        _type_: _description_
    """
    if 'soma_position' in GroundTruth.get_property_keys():
        GroundTruth.set_property('x', GroundTruth.get_property('soma_position')[:,0])
        GroundTruth.set_property('y', GroundTruth.get_property('soma_position')[:,1])
        GroundTruth.set_property('z', GroundTruth.get_property('soma_position')[:,2])
        GroundTruth.delete_property('soma_position')

    if 'soma_location' in GroundTruth.get_property_keys():
        GroundTruth.set_property('x', GroundTruth.get_property('soma_location')[:,0])
        GroundTruth.set_property('y', GroundTruth.get_property('soma_location')[:,1])
        GroundTruth.set_property('z', GroundTruth.get_property('soma_location')[:,2])
        GroundTruth.delete_property('soma_location')

    if 'bursting' in GroundTruth.get_property_keys():
        GroundTruth.set_property('bursting', GroundTruth.get_property('bursting').astype(int))
    
    if 'exp_decay' in GroundTruth.get_property_keys():
        GroundTruth.set_property('exp_decay', GroundTruth.get_property('exp_decay').astype(str))
    
    if 'max_burst_duration' in GroundTruth.get_property_keys():
        GroundTruth.set_property('max_burst_duration', GroundTruth.get_property('max_burst_duration').astype(str))
    
    if 'max_spikes_per_burst' in GroundTruth.get_property_keys():
        GroundTruth.set_property('max_spikes_per_burst', GroundTruth.get_property('max_spikes_per_burst').astype(str))
    return GroundTruth


def standardize_site_location_property(Recording, zero_coord=0):
    """
    zero_coord: coordinate with zero values
    """
    site_pos = Recording.get_channel_locations()
    site_pos = np.insert(site_pos, zero_coord, 0, axis=1)
    Recording.set_property('location3D', site_pos)
    return Recording


def standardize_biophy_site_location_property(Recording):
    """
    zero_coord: coordinate with zero values
    """
    site_pos = Recording.get_property('location')
    Recording.set_property('location3D', site_pos)
    return Recording


def report_simulation_parameters(rec_gen):
    """
    Reports simulation parameters, checks for bursting, and displays example cell metadata.

    Args:
        rec_gen (RecordingGenerator): The recording generator object.
    """
    # report parameters
    display('Simulation parameters:', rec_gen.info)

    # check bursting
    bursting = []
    for ix in range(250):
        bursting.append(rec_gen.spiketrains[1].annotations['bursting'])
    print('Bursting:', any(bursting))

    # example of a cell metadata
    CELL_IDX = 0
    print('\nExample cell type:', rec_gen.template_celltypes[CELL_IDX])
    display('Its spiking metadata:', rec_gen.spiketrains[CELL_IDX].annotations)
    print('Its spike times:', rec_gen.spiketrains[CELL_IDX].times)


def save_simulation_parameters_to_assets(rec_prms:dict, rec_gen, input_params_path:str, output_params_path:str):
        
    folder_path = os.path.dirname(input_params_path)
    os.makedirs(folder_path, exist_ok=True)

    # save input simulation parameters to MEAREC
    with open(input_params_path, 'w') as f:
        json.dump(rec_prms, f, indent=4)

    folder_path = os.path.dirname(output_params_path)
    os.makedirs(folder_path, exist_ok=True)

    # save parameters recorded in recording generator
    with open(output_params_path, 'w') as f:
        json.dump(rec_gen.params, f, indent=4)


def get_cell_type_property_from_mearec(simulation):
    cell_types = []
    for ix in range(len(simulation.spiketrains)):
        cell_types.append(simulation.spiketrains[ix].annotations['cell_type'])
    return np.array(cell_types)


# DESCRIPTIVE FEATURES
