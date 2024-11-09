import shutil
import pandas as pd
from src.nodes.utils import get_config
import spikeinterface as si



def label_sorted_unit_layer_vivo_horvath(experiment:str="vivo_horvath", run:str="probe_1"):
    """label sorted unit layers from in vivo horvath dataset
    Sorted units nearest contact and its layer are added as properties
    to SpikeInterface's SortingExtractor.
    """
    data_conf_1, _ = get_config(experiment, run).values()
    SORTED_PATH_1 = data_conf_1["sorting"]["sorters"]["kilosort3"]["output"]
    METADATA_1 = data_conf_1["postprocessing"]["sorted_neuron_metadata"]

    # get triangulated layer location
    metadata_1 = pd.read_csv(METADATA_1)

    # standardize values
    metadata_1["layer"] = metadata_1["layer"].apply(lambda x: x.replace(" ", ""))
    metadata_1["layer"] = metadata_1["layer"].replace("L2", "L2/3")
    metadata_1["layer"] = metadata_1["layer"].replace("L3", "L2/3")
    metadata_1["layer"] = metadata_1["layer"].replace("Outsideofthecortex", "Outside")

    # add layer as property
    sorting_1 = si.load_extractor(SORTED_PATH_1)

    sorting_1.set_property("layer", metadata_1["layer"].tolist())

    # add nearest contact as property
    sorting_1.set_property("contact", metadata_1["contact"].tolist())

    # save
    shutil.rmtree(SORTED_PATH_1, ignore_errors=True)
    sorting_1.save(folder=SORTED_PATH_1)
    print("All done")

    
def label_sorted_unit_layer_simulations(Sorting, experiment:str, run:str, save:bool):
    """label sorted units' layer and contact from in silico simulations
    for spontaneous and evoked conditions
    Sorted units nearest contact and its layer are added as properties
    to SpikeInterface's SortingExtractor.

    Args:
        Sorting: Sorting extractor

    """
    # set experiment paths for the three depths
    data_cfg, _ = get_config(experiment, run).values()
    SORTING_PATH = data_cfg["sorting"]["sorters"]["kilosort3"]["output"]
    METADATA_FILE = data_cfg["postprocessing"]["sorted_neuron_metadata"]

    # get triangulated layer location
    metadata = pd.read_csv(METADATA_FILE)

    # standardize layer names
    metadata["layer"] = metadata["layer"].apply(lambda x: x.replace(" ", ""))
    metadata["layer"] = metadata["layer"].replace("L2", "L2/3")
    metadata["layer"] = metadata["layer"].replace("L3", "L2/3")
    metadata["layer"] = metadata["layer"].replace("Outsideofthecortex", "Outside")

    # Save layer as a property
    Sorting.set_property("layer", metadata["layer"].tolist())
    
    # Save nearest site as a property
    Sorting.set_property("contact", metadata["contact"].tolist())

    # save Sorting Extractor
    if save:
        shutil.rmtree(SORTING_PATH, ignore_errors=True)
        Sorting.save(folder=SORTING_PATH)
    print("All done")
    return Sorting