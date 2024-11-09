"""Pipeline that describes and create a new channel weight file of 32 channels centered along the 30K microcircuit (`hexo` group) 
from a 384 contact neuropixel probe, 2 Millions neurons channel weight file.

Usage:

    # Create python virtual environment
    python3.9 -m venv env_silico
    source env_silico/bin/activate

    # Then pip3.9 install
    pip3.9 install -r requirements_silico.txt

    # run channel weight editing pipeline
    python -m src.pipes.silico.channel_weights

Returns:
    (.h5): new channel weighht file at path specified in conf/silico_neuropixels/data_conf

Note: the code can be substantially optimized for speed
"""

import logging
import logging.config
from time import time

import h5py
import numpy as np
import yaml
from matplotlib import pyplot as plt

from src.nodes.utils import get_config

# NEUROPIXELS
EXP = "silico_neuropixels"
SIMULATION_DATE = "2023_02_08"
N_CONTACTS_TO_KEEP = 32
CONTACTS_TO_KEEP = np.arange(127, 127 + N_CONTACTS_TO_KEEP)

data_conf, param_conf = get_config(EXP, SIMULATION_DATE).values()
SRC_WEIGHT_PATH = data_conf["campaign"]["source_weights"]
DEST_WEIGHT_PATH = data_conf["campaign"]["edited_weights"]

# REYES
REYES_EXP = "supp/silico_reyes"
REYES_SIMULATION_DATE = "2023_01_13"
reyes_data_conf, _ = get_config(REYES_EXP, REYES_SIMULATION_DATE).values()
target_cell_path = reyes_data_conf["campaign"]["target_cells"]

# SET FIG PATHS
FIG_PATH = data_conf["figures"]["silico"]["channel_weights"]


# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def describe_weights(weight_path: str):
    """_summary_

    Args:
        weight_path (str): _description_
    """

    # read weight file
    with h5py.File(weight_path, "r") as weights:

        # describe keys
        logger.info("\nWeight keys:", weights.keys())

        # describe neuron_ids key type
        logger.info("neuron_ids (type):", type(weights["neuron_ids"]))

        # shape
        logger.info("neuron_ids (count):", weights["neuron_ids"].shape)

        # describe "offsets" type
        logger.info("offsets (type):", type(weights["offsets"]))

        # count "offsets" component keys --> returns 211,712 neurons (stats: takes 40)
        logger.info("offsets (count):", len(weights["offsets"].keys()))

        # get type of offset components
        logger.info("'1000036' (type):", type(weights["offsets"]["1000036"]))

        # get shape
        logger.info("offsets (shape):", weights["offsets"]["1000036"].shape)

        # get type of "sec_ids"
        logger.info("sec_ids (type):", type(weights["sec_ids"]))

        # count components --> returns 211,712 neurons (stats: takes 1 min)
        logger.info("sec_ids (cell count):", len(weights["sec_ids"].keys()))

        # count a neuron's component
        logger.info("'1000036' (shape):", weights["sec_ids"]["1000036"].shape)

        # describe "electrodes" structure type (see 1 for "group" type)
        logger.info("\nElectrodes (type):", type(weights["electrodes"]))

        # describe electrodes group keys
        logger.info(
            "\nElectrodes:\n-", list(weights["electrodes"].keys())[:10], "..."
        )
        logger.info("-", list(weights["electrodes"].keys())[-1])

        # describe "electrode_grid"
        logger.info(
            "\nelectrode_grid:\n- type:",
            weights["electrodes"].get("electrode_grid", getclass=True),
        )
        logger.info(
            "- shape:",
            weights["electrodes"]["electrode_grid"]["1000036"].shape,
        )

        ## count "electrode_grid" component keys (stats: takes 10 min) -> returns 211,712 neurons
        # logger.info("\n",len(weights["electrodes"]["electrode_grid"].keys()))
        # weights["electrodes"]["electrode_grid"].visititems(print)

        # analye electrodes components
        logger.info(
            "\nNeuropixels-384_0:\n-type:", type(weights["electrodes"])
        )
        logger.info(
            "\nNeuropixels-384_0 components:\n-type:",
            weights["electrodes"]["Neuropixels-384_0"].keys(),
        )
        logger.info(
            "\nlayer:\n-type:",
            type(weights["electrodes"]["Neuropixels-384_0"]["layer"]),
        )
        logger.info(
            "-shape:",
            weights["electrodes"]["Neuropixels-384_0"]["layer"].shape,
        )

        logger.info(
            "\nlocation:\n-type:",
            type(weights["electrodes"]["Neuropixels-384_0"]["location"]),
        )
        logger.info(
            "-type:",
            weights["electrodes"]["Neuropixels-384_0"]["location"].shape,
        )
        logger.info(
            "-content:",
            weights["electrodes"]["Neuropixels-384_0"]["location"][:],
        )

        logger.info(
            "\noffset:\n-type:",
            type(weights["electrodes"]["Neuropixels-384_0"]["offset"]),
        )
        logger.info(
            "-shape:",
            weights["electrodes"]["Neuropixels-384_0"]["offset"].shape,
        )

        logger.info(
            "\nregion:\n-type:",
            type(weights["electrodes"]["Neuropixels-384_0"]["region"]),
        )
        logger.info(
            "-shape:",
            weights["electrodes"]["Neuropixels-384_0"]["region"].shape,
        )

        logger.info(
            "\ntype:\n-type:",
            type(weights["electrodes"]["Neuropixels-384_0"]["type"]),
        )
        logger.info(
            "-shape:", weights["electrodes"]["Neuropixels-384_0"]["type"].shape
        )

        #  plot channel contact channel weights
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))
        axes[0].imshow(
            weights["electrodes"]["electrode_grid"]["1000036"][:, :]
        )
        axes[0].set_title("neuron 1000036")
        axes[1].imshow(
            weights["electrodes"]["electrode_grid"]["1000089"][:, :]
        )
        axes[1].set_title("neuron 1000089")
        axes[1].set_xlabel("384 contacts (id)")
        axes[2].imshow(
            weights["electrodes"]["electrode_grid"]["1000103"][:, :]
        )
        axes[2].set_title("neuron 1000103")
        plt.tight_layout()

        # write figure
        fig.savefig(FIG_PATH)


def get_reyes_neurons(target_cells_path: str):
    with h5py.File(target_cells_path, "r") as target_cell_file:
        target_cells = target_cell_file["hex0"][:]
    return target_cells


def create_new_weight_file(src_weight_path: str, dest_weight_path: str):
    """Create a new weight file by duplicating a weight file stored in "src_weight_path"
    Stats: takes 22 min for 384 channels, 2 Millions neurons weights

    Args:
        src_weight_path (str): _description_
        dest_weight_path (str): _description_
    """

    t_0 = time()
    logger.info("Creating new weight file ...")

    # duplicate source weight file
    src_weights = h5py.File(src_weight_path, "r")
    with h5py.File(dest_weight_path, "w") as dest_file:
        for obj in src_weights.keys():
            src_weights.copy(obj, dest_file)
    src_weights.close()

    logger.info(
        "Creating new weight file - done in %s secs",
        np.round(time() - t_0, 2),
    )


def update_metadata(
    contacts_to_keep, src_weight_path: str, dest_weight_path: str
):
    """update the new weight file's channel metadata

    Args:
        contacts_to_keep (_type_): _description_
        src_weight_path (str): _description_
        dest_weight_path (str): _description_
    """

    # reconstruct name of metadata to keep
    metadata_to_keep = []
    for c_i in contacts_to_keep:
        metadata_to_keep.append(f"Neuropixels-384_{c_i}")

    # get metadata
    with h5py.File(src_weight_path) as raw_weights:
        metadata = list(raw_weights["electrodes"].keys())

    # filter contact metadata
    contact_metadata = [m_i for m_i in metadata if "Neuropixels" in m_i]

    # get metadata to delete
    metadata_to_drop = list(set(contact_metadata) - set(metadata_to_keep))

    # remove unwanted contacts from the new weight file
    t_0 = time()

    logger.info("Removing unwanted contacts from the new weight file...")
    with h5py.File(dest_weight_path, "a") as dest_file:
        try:
            for c_i in metadata_to_drop:
                del dest_file["electrodes"][c_i]
                try:
                    dest_file["electrodes"][c_i].value
                except KeyError as err:
                    logger.info(err)
                    pass
        except KeyError as err:
            logger.info(err)
            pass

    logger.info(
        "Removing unwanted contacts from the new weight file - done in %s secs",
        np.round(time() - t_0, 2),
    )

    # test the number of contacts in the new file
    with h5py.File(dest_weight_path, "r") as dest_weights:
        logger.info(
            "new number of contacts: %s",
            len(dest_weights["electrodes"].keys()) - 1,
        )
        logger.info("new metadata: %s", dest_weights["electrodes"].keys())
        assert (
            len(dest_weights["electrodes"].keys()) == len(contacts_to_keep) + 1
        ), "The number contacts produced is wrong"


def update_channel_weights(
    neurons_to_keep, contacts_to_keep, src_weight_path, dest_weight_path: str
):
    """update the channel weight file's neurons and weight matrices

    Args:
        neurons_to_keep (_type_): _description_
        contacts_to_keep (_type_): _description_
        src_weight_path (_type_): _description_
        dest_weight_path (str): _description_
    """

    t_0 = time()

    # get electrode grid from source file
    src_weights = h5py.File(src_weight_path, "r")
    electrode_grid = src_weights["electrodes/electrode_grid"]

    # update destination file
    with h5py.File(dest_weight_path, "a") as dest_weights:

        # reset electrode grid
        try:
            del dest_weights["electrodes/electrode_grid"]
            dest_weights.create_group(
                "electrodes/electrode_grid",
            )
        except KeyError as err:
            logger.info(err)
            pass

        logger.info("Updating each neuron's channel weights ...")

        # create a new grid with the selected neurons and contacts
        count = 0
        for name in neurons_to_keep:
            name_str = str(name)

            try:
                # add its edite weights
                dest_weights.create_dataset(
                    f"electrodes/electrode_grid/{name_str}",
                    data=electrode_grid[name_str][:, CONTACTS_TO_KEEP],
                )
            except KeyError as err:
                logger.info(err)
                pass

            # count neurons
            count += 1

            # print progress
            logger.info(
                f"""neuron {count}/{len(neurons_to_keep)} - {dest_weights[f"electrodes/electrode_grid/{name_str}"]}"""
            )

            # test contacts
            assert dest_weights[f"electrodes/electrode_grid/{name_str}"].shape[
                1
            ] == len(contacts_to_keep), "The new number of contacts is wrong"

        # test neurons
        assert len(dest_weights[f"electrodes/electrode_grid/"].keys()) == len(
            neurons_to_keep
        ), "The new number of neurons is wrong"

    # close source weight file
    src_weights.close()

    logger.info(
        "Updating each neuron's channel weights - done in %s secs",
        np.round(time() - t_0, 2),
    )


def update_neuron_ids(
    neuron_ids: np.array,
    source_weight_path: str,
    dest_weight_path: str,
):
    """_summary_

    Args:
        neuron_ids (np.array): _description_
        weight_path (str): _description_
    """

    t_0 = time()

    logger.info(
        "Updating neuron ids and circuit attribute ...",
    )

    # update destination file
    with h5py.File(dest_weight_path, "a") as weights:

        # reset neuron ids
        try:
            del weights["neuron_ids"]
            weights.create_dataset(
                "neuron_ids",
                data=neuron_ids,
            )
        except KeyError as err:
            logger.info(err)
            pass

    # update attribute
    source_weights = h5py.File(source_weight_path, "r")
    src_circuit = source_weights["neuron_ids"].attrs["circuit"]
    with h5py.File(dest_weight_path, "a") as dest_weights:
        dest_weights["neuron_ids"].attrs["circuit"] = src_circuit
    source_weights.close()

    logger.info(
        "Updating neuron ids and circuit attribute - done in %s secs",
        np.round(time() - t_0, 2),
    )


# describe file content
# describe_weights(weight_path=SRC_WEIGHT_PATH)

# create new weight file
create_new_weight_file(SRC_WEIGHT_PATH, DEST_WEIGHT_PATH)

# update contact metadata
update_metadata(CONTACTS_TO_KEEP, SRC_WEIGHT_PATH, DEST_WEIGHT_PATH)

# get the neurons to keep
neurons_to_keep = get_reyes_neurons(target_cell_path)

# update the weights channels
update_channel_weights(
    neurons_to_keep, CONTACTS_TO_KEEP, SRC_WEIGHT_PATH, DEST_WEIGHT_PATH
)

# update the neurons
update_neuron_ids(neurons_to_keep, SRC_WEIGHT_PATH, DEST_WEIGHT_PATH)
