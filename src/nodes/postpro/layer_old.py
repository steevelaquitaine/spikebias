import shutil
import pandas as pd
from src.nodes.utils import get_config
import spikeinterface as si

def label_sorted_unit_layer_horvath_silico():
    """label sorted units' layer and contact from in silico simulations
    Sorted units nearest contact and its layer are added as properties
    to SpikeInterface's SortingExtractor.
    """
    # set experiment paths for the three depths
    EXPERIMENT = "silico_horvath"

    data_conf_1, _ = get_config(EXPERIMENT, "concatenated/probe_1").values()
    SORTED_PATH_1 = data_conf_1["sorting"]["sorters"]["kilosort3"]["output"]
    metadata_FILE_1 = data_conf_1["postprocessing"]["sorted_neuron_metadata"]

    data_conf_2, _ = get_config(EXPERIMENT, "concatenated/probe_2").values()
    SORTED_PATH_2 = data_conf_2["sorting"]["sorters"]["kilosort3"]["output"]
    metadata_FILE_2 = data_conf_2["postprocessing"]["sorted_neuron_metadata"]

    data_conf_3, _ = get_config(EXPERIMENT, "concatenated/probe_3").values()
    SORTED_PATH_3 = data_conf_3["sorting"]["sorters"]["kilosort3"]["output"]
    metadata_FILE_3 = data_conf_3["postprocessing"]["sorted_neuron_metadata"]

    # get triangulated layer location
    metadata_1 = pd.read_csv(metadata_FILE_1)
    metadata_2 = pd.read_csv(metadata_FILE_2)
    metadata_3 = pd.read_csv(metadata_FILE_3)

    # standardize values
    metadata_1["layer"] = metadata_1["layer"].apply(lambda x: x.replace(" ", ""))
    metadata_1["layer"] = metadata_1["layer"].replace("L2", "L2/3")
    metadata_1["layer"] = metadata_1["layer"].replace("L3", "L2/3")
    metadata_1["layer"] = metadata_1["layer"].replace("Outsideofthecortex", "Outside")

    metadata_2["layer"] = metadata_2["layer"].apply(lambda x: x.replace(" ", ""))
    metadata_2["layer"] = metadata_2["layer"].replace("L2", "L2/3")
    metadata_2["layer"] = metadata_2["layer"].replace("L3", "L2/3")
    metadata_2["layer"] = metadata_2["layer"].replace("Outsideofthecortex", "Outside")

    metadata_3["layer"] = metadata_3["layer"].apply(lambda x: x.replace(" ", ""))
    metadata_3["layer"] = metadata_3["layer"].replace("L2", "L2/3")
    metadata_3["layer"] = metadata_3["layer"].replace("L3", "L2/3")
    metadata_3["layer"] = metadata_3["layer"].replace("Outsideofthecortex", "Outside")

    # add layer as property
    sorting_1 = si.load_extractor(SORTED_PATH_1)
    sorting_2 = si.load_extractor(SORTED_PATH_2)
    sorting_3 = si.load_extractor(SORTED_PATH_3)

    sorting_1.set_property("layer", metadata_1["layer"].tolist())
    sorting_2.set_property("layer", metadata_2["layer"].tolist())
    sorting_3.set_property("layer", metadata_3["layer"].tolist())
    
    # add nearest contact as property
    sorting_1.set_property("contact", metadata_1["contact"].tolist())
    sorting_2.set_property("contact", metadata_2["contact"].tolist())
    sorting_3.set_property("contact", metadata_3["contact"].tolist())

    # save
    shutil.rmtree(SORTED_PATH_1, ignore_errors=True)
    sorting_1.save(folder=SORTED_PATH_1)
    shutil.rmtree(SORTED_PATH_2, ignore_errors=True)
    sorting_2.save(folder=SORTED_PATH_2)
    shutil.rmtree(SORTED_PATH_3, ignore_errors=True)
    sorting_3.save(folder=SORTED_PATH_3)
    print("All done")


def label_sorted_unit_layer_vivo_horvath():
    """label sorted unit layers from in vivo horvath dataset
    Sorted units nearest contact and its layer are added as properties
    to SpikeInterface's SortingExtractor.
    """
    # set experiment paths for the three depths
    EXPERIMENT = "vivo_horvath"

    data_conf_1, _ = get_config(EXPERIMENT, "probe_1").values()
    SORTED_PATH_1 = data_conf_1["sorting"]["sorters"]["kilosort3"]["output"]
    METADATA_1 = data_conf_1["postprocessing"]["sorted_neuron_metadata"]

    data_conf_2, _ = get_config(EXPERIMENT, "probe_2").values()
    SORTED_PATH_2 = data_conf_2["sorting"]["sorters"]["kilosort3"]["output"]
    METADATA_2 = data_conf_2["postprocessing"]["sorted_neuron_metadata"]

    data_conf_3, _ = get_config(EXPERIMENT, "probe_3").values()
    SORTED_PATH_3 = data_conf_3["sorting"]["sorters"]["kilosort3"]["output"]
    METADATA_3 = data_conf_3["postprocessing"]["sorted_neuron_metadata"]

    # get triangulated layer location
    metadata_1 = pd.read_csv(METADATA_1)
    metadata_2 = pd.read_csv(METADATA_2)
    metadata_3 = pd.read_csv(METADATA_3)

    # standardize values
    metadata_1["layer"] = metadata_1["layer"].apply(lambda x: x.replace(" ", ""))
    metadata_1["layer"] = metadata_1["layer"].replace("L2", "L2/3")
    metadata_1["layer"] = metadata_1["layer"].replace("L3", "L2/3")
    metadata_1["layer"] = metadata_1["layer"].replace("Outsideofthecortex", "Outside")

    metadata_2["layer"] = metadata_2["layer"].apply(lambda x: x.replace(" ", ""))
    metadata_2["layer"] = metadata_2["layer"].replace("L2", "L2/3")
    metadata_2["layer"] = metadata_2["layer"].replace("L3", "L2/3")
    metadata_2["layer"] = metadata_2["layer"].replace("Outsideofthecortex", "Outside")

    metadata_3["layer"] = metadata_3["layer"].apply(lambda x: x.replace(" ", ""))
    metadata_3["layer"] = metadata_3["layer"].replace("L2", "L2/3")
    metadata_3["layer"] = metadata_3["layer"].replace("L3", "L2/3")
    metadata_3["layer"] = metadata_3["layer"].replace("Outsideofthecortex", "Outside")

    # add layer as property
    sorting_1 = si.load_extractor(SORTED_PATH_1)
    sorting_2 = si.load_extractor(SORTED_PATH_2)
    sorting_3 = si.load_extractor(SORTED_PATH_3)

    sorting_1.set_property("layer", metadata_1["layer"].tolist())
    sorting_2.set_property("layer", metadata_2["layer"].tolist())
    sorting_3.set_property("layer", metadata_3["layer"].tolist())

    # add nearest contact as property
    sorting_1.set_property("contact", metadata_1["contact"].tolist())
    sorting_2.set_property("contact", metadata_2["contact"].tolist())
    sorting_3.set_property("contact", metadata_3["contact"].tolist())

    # save
    shutil.rmtree(SORTED_PATH_1, ignore_errors=True)
    sorting_1.save(folder=SORTED_PATH_1)
    shutil.rmtree(SORTED_PATH_2, ignore_errors=True)
    sorting_2.save(folder=SORTED_PATH_2)
    shutil.rmtree(SORTED_PATH_3, ignore_errors=True)
    sorting_3.save(folder=SORTED_PATH_3)
    print("All done")