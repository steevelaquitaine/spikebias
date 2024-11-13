"""functions to analyse unit types

notes:

    `mtypes` can be mapped with (see Markram 2015, p460 and 462): 
    - Excitatory pyramidal, 
    - Interneurons: 
        - Parvalbumin (LBC and NBC), 
        - Somatostatin (Martinotti cells (MC))
        - VIPs (mostly small basket cells SBCs)
    
Returns:
    _type_: _description_
"""
import shutil
import pandas as pd
from src.nodes.truth.silico import ground_truth 


def label_true_cell_properties(data_cfg: dict, Sorting, blueconfig_path: str, true_sorting_path: str, save: bool):
    """set true cell properties in spikeinterface's sorting extractor 

    Args:
        data_cfg: datasets path config 
        Sorting: Ground truth Sorting Extractor
        blueconfig_path (None): path of BBP's blueconfig. Is not available so should always be None.
        anymore after publication.
        true_sorting_path (str): Ground Truth Sorting Extractor
    
    Returns:
        Sorting Extractor
    """
    # get circuit's cell properties    
    if blueconfig_path:
        import bluepy
        simulation = bluepy.Simulation(blueconfig_path)
        cell_properties = list(simulation.circuit.cells.available_properties)
        df = simulation.circuit.cells.get(
            Sorting.unit_ids, properties=list(cell_properties)
        )
        cell_properties = [prop.replace("@dynamics:", "dynamics_") for prop in cell_properties]
        df.columns = cell_properties
    
        # cast as string for saving as h5
        df.orientation = df.orientation.astype(str)

        # save cell properties to h5 file in the local project path
        df.to_hdf(data_cfg["metadata"]["cell_properties"], key="cell_properties", format='t', mode="w")
    else:
        # cell properties to h5 file in the local project path
        df = pd.read_hdf(data_cfg["metadata"]["cell_properties"], key="cell_properties")
    
    # add as property to Sorting Extractor
    for prop in df.columns:
        Sorting.set_property(prop, df[prop].values.tolist())
    
    # make a "layers" copy of "layer"
    # for convenience
    Sorting.set_property("layers", df["layer"].values.tolist())

    # save Sorting Extractor
    if save:
        shutil.rmtree(true_sorting_path, ignore_errors=True)
        Sorting.save(folder=true_sorting_path)
    return Sorting


def get_interneurons(data_conf:dict, cell_type:str):
    """Get unit ids of "cell type" interneurons 

    Args:
        data_conf (dict): _description_
        cell_type (str): 
        - parvalbumin: 'LBC|NBC'
        - somatostatin: 'MC'
        - VIP: 'SBC'

    Returns:
        _type_: _description_
    """
    # filter all near-contact pyramidal cells
    from src.nodes.load import load_campaign_params 
    simulation = load_campaign_params(data_conf)
    SortingTrue = ground_truth.load(data_conf)

    # get cell types
    cell_types = simulation["circuit"].cells.get(SortingTrue.unit_ids, properties=['mtype'])
    interneuron = cell_types[cell_types["mtype"].str.contains(cell_type, case=False)].index.values
    return interneuron


def get_pyramidal(data_conf:dict):
    
    # filter all near-contact pyramidal cells
    from src.nodes.load import load_campaign_params 
    simulation = load_campaign_params(data_conf)
    SortingTrue = ground_truth.load(data_conf)

    # get cell types
    cell_types = simulation["circuit"].cells.get(SortingTrue.unit_ids, properties=['morph_class'])
    return cell_types[cell_types["morph_class"] == "PYR"].index