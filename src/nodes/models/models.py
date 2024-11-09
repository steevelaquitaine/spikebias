"""Useful functions for the model node
author: steeve.laquitaine@epfl.ch
"""

import os
import spikeinterface as si
import spikeinterface.core.template_tools as ttools
from spikeinterface import comparison
from spikeinterface.qualitymetrics import compute_quality_metrics as qm
from spikeinterface import qualitymetrics
import pandas as pd
from cebra import CEBRA
import cebra
import torch
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns
from sklearn import metrics
import cebra.models
import shutil
from spikeinterface.postprocessing import compute_principal_components
import multiprocessing
from src.nodes import utils
from src.nodes.models.utils import plot as mplot
from src.nodes.models.Flc import utils as mutils


class CebraSpike1():
    
    def __init__():
        """model instantiation
        """
        pass
    
    def get_layerwise_model(model_cfg: dict, train: bool, model_path: str, dataset, layers: list):
        """train discrete label-supervised CEBRA
        
        Args:
            model_cfg
        
        """
        
        # pre-allocate the models
        CebraL23 = None
        CebraL4 = None
        CebraL5 = None
        CebraL6 = None
                
        # training 
        if train:
            
            # instantiate the models
            if "l23" in layers:
                CebraL23 = CEBRA(**model_cfg)
            if "l4" in layers:
                CebraL4 = CEBRA(**model_cfg)
            if "l5" in layers:
                CebraL5 = CEBRA(**model_cfg)
            if "l6" in layers:
                CebraL6 = CEBRA(**model_cfg)

            # train the models
            if "l23" in layers:
                CebraL23.fit(dataset["data_l23"], dataset["label_l23"])
            if "l4" in layers:
                CebraL4.fit(dataset["data_l4"],dataset["label_l4"])
            if "l5" in layers:
                CebraL5.fit(dataset["data_l5"], dataset["label_l5"])
            if "l6" in layers:
                CebraL6.fit(dataset["data_l6"], dataset["label_l6"])

            # save the models
            utils.create_if_not_exists(model_path)
            if "l23" in layers:
                CebraL23.save(model_path + "cebra_l23.pt")
            if "l4" in layers:
                CebraL4.save(model_path + "cebra_l4.pt")
            if "l5" in layers:
                CebraL5.save(model_path + "cebra_l5.pt")
            if "l6" in layers:
                CebraL6.save(model_path + "cebra_l6.pt")
        else:
            # load the models
            if "l23" in layers:
                CebraL23 = cebra.CEBRA.load(model_path + "cebra_l23.pt")
            if "l4" in layers:
                CebraL4 = cebra.CEBRA.load(model_path + "cebra_l4.pt")
            if "l5" in layers:
                CebraL5 = cebra.CEBRA.load(model_path + "cebra_l5.pt")
            if "l6" in layers:
                CebraL6 = cebra.CEBRA.load(model_path + "cebra_l6.pt")

        # get the embeddings
        CebraL23_em = None
        CebraL4_em = None
        CebraL5_em = None
        CebraL6_em = None
        
        if "l23" in layers:
            CebraL23_em = CebraL23.transform(dataset["data_l23"])
        if "l4" in layers:
            CebraL4_em = CebraL4.transform(dataset["data_l4"])
        if "l5" in layers:
            CebraL5_em = CebraL5.transform(dataset["data_l5"])
        if "l6" in layers:
            CebraL6_em = CebraL6.transform(dataset["data_l6"])
            
        return {
            "model_l23": CebraL23,
            "model_l4": CebraL4,
            "model_l5": CebraL5,
            "model_l6": CebraL6,
            "l23": CebraL23_em,
            "l4": CebraL4_em,
            "l5": CebraL5_em,
            "l6": CebraL6_em,
        }


    def get_layerwise_mixed_model(model_cfg: dict, train: bool, model_path: str, dataset, layers: list=["l23", "l4", "l5", "l6"]):
        """train mixed continuous and discrete label-supervised CEBRA"""

        # check that continuous labels are arrays
        if isinstance(dataset["cont_label_l23"], pd.DataFrame):
            
            cont_l23 = None
            cont_l4 = None
            cont_l5 = None
            cont_l6 = None
            
            if "l23" in layers:
                cont_l23 = dataset["cont_label_l23"].values
            if "l4" in layers:
                cont_l4 = dataset["cont_label_l4"].values
            if "l5" in layers:
                cont_l5 = dataset["cont_label_l5"].values
            if "l6" in layers:
                cont_l6 = dataset["cont_label_l6"].values

        # train or load
        if train:
            
            CebraL23 = None
            CebraL4 = None
            CebraL5 = None
            CebraL6 = None
            
            # instantiate model
            if "l23" in layers:
                CebraL23 = CEBRA(**model_cfg)
            if "l4" in layers:
                CebraL4 = CEBRA(**model_cfg)
            if "l5" in layers:
                CebraL5 = CEBRA(**model_cfg)
            if "l6" in layers:
                CebraL6 = CEBRA(**model_cfg)

            # train model
            if "l23" in layers:
                CebraL23.fit(dataset["data_l23"], cont_l23, dataset["label_l23"])
            if "l4" in layers:
                CebraL4.fit(dataset["data_l4"], cont_l4, dataset["label_l4"])
            if "l5" in layers:
                CebraL5.fit(dataset["data_l5"], cont_l5, dataset["label_l5"])
            if "l6" in layers:
                CebraL6.fit(dataset["data_l6"], cont_l6, dataset["label_l6"])

            # save model
            utils.create_if_not_exists(model_path)
            if "l23" in layers:
                CebraL23.save(model_path + "cebra_mixed_labels_l23.pt")
            if "l4" in layers:
                CebraL4.save(model_path + "cebra_mixed_labels_l4.pt")
            if "l5" in layers:
                CebraL5.save(model_path + "cebra_mixed_labels_l5.pt")
            if "l6" in layers:
                CebraL6.save(model_path + "cebra_mixed_labels_l6.pt")
        else:
            # load
            if "l23" in layers:
                CebraL23 = cebra.CEBRA.load(model_path + "cebra_labels_l23.pt")
            if "l4" in layers:
                CebraL4 = cebra.CEBRA.load(model_path + "cebra_labels_l4.pt")
            if "l5" in layers:
                CebraL5 = cebra.CEBRA.load(model_path + "cebra_labels_l5.pt")
            if "l6" in layers:
                CebraL6 = cebra.CEBRA.load(model_path + "cebra_labels_l6.pt")

        # get embedding
        CebraL23_em = None
        CebraL4_em = None
        CebraL5_em = None
        CebraL6_em = None
        
        if "l23" in layers:
            CebraL23_em = CebraL23.transform(dataset["data_l23"])
        if "l4" in layers:
            CebraL4_em = CebraL4.transform(dataset["data_l4"])
        if "l5" in layers:
            CebraL5_em = CebraL5.transform(dataset["data_l5"])
        if "l6" in layers:
            CebraL6_em = CebraL6.transform(dataset["data_l6"])
        return {
            "model_l23": CebraL23,
            "model_l4": CebraL4,
            "model_l5": CebraL5,
            "model_l6": CebraL6,
            "l23": CebraL23_em,
            "l4": CebraL4_em,
            "l5": CebraL5_em,
            "l6": CebraL6_em,
        }


    def get_pooled_model(model_cfg: dict, train: bool, model_path, dataset):
        """train discrete label-supervised CEBRA
        Args
            train (bool)
            model_path (str): save path
            dataset (dict):
            - "data": spike data
            - "label": supervised labels
            max_iter (int): number of training iterations

        Returns:

        """
        if train:
            # instantiate model
            CebraPooled = CEBRA(**model_cfg)
            # train model
            CebraPooled.fit(dataset["data"], dataset["label"])
            # save model
            utils.create_if_not_exists(model_path)
            CebraPooled.save(model_path + "cebra_pooled.pt")
        else:
            # load
            CebraPooled = cebra.CEBRA.load(model_path + "cebra_pooled.pt")

        # get embedding
        CebraPooled_em = CebraPooled.transform(dataset["data"])
        return {"model": CebraPooled, "embedding": CebraPooled_em}


    def get_pooled_mixed_model(model_cfg: dict, train: bool, model_path, dataset):
        """train discrete label-supervised CEBRA
        Args
            train (bool)
            model_path (str): save path
            dataset (dict):
            - "data": spike data
            - "label": supervised labels
            max_iter (int): number of training iterations

        Returns:

        """
        # check that continuous label object is an array
        if isinstance(dataset["cont_label"], pd.DataFrame):
            cont_label = dataset["cont_label"].values

        # train or load
        if train:
            # instantiate model
            CebraPooled = CEBRA(**model_cfg)
            # train model
            CebraPooled.fit(dataset["data"], cont_label, dataset["label"])
            # save model
            utils.create_if_not_exists(model_path)
            CebraPooled.save(model_path + "cebra_mixed_labels_pooled.pt")
        else:
            # load
            CebraPooled = cebra.CEBRA.load(model_path + "cebra_mixed_labels_pooled.pt")

        # get embedding
        CebraPooled_em = CebraPooled.transform(dataset["data"])
        return {"model": CebraPooled, "embedding": CebraPooled_em}


    def select_model(
        mixed_model: bool,
        model_cfg: dict,
        train: bool,
        model_path: str,
        train_layerwise,
        train_pooled,
        layers: list
    ):
        """select the CEBRA model to train

        Args:
            mixed_model (bool):
            True: CEBRA model with a mix of discrete
            - (unit quality labels) and continuous quality metrics
            - auxiliary variables
            False: CEBRA model with only discrete unit quality labels
            model_cfg (dict): CEBRA model parameters
            train (bool): True: train; False: load
            model_path (str): path where the model is saved
            train_layerwise (_type_): _description_
            train_pooled (_type_): _description_

        Returns:
            _type_: _description_
        """

        # if mix of discrete unit quality and continuous waveform
        # quality metrics
        if mixed_model:
            model_lyr_ev_20Khz = CebraSpike1.get_layerwise_mixed_model(
                model_cfg, train, model_path, train_layerwise, layers
            )
            model_pool_ev_20Khz = CebraSpike1.get_pooled_mixed_model(
                model_cfg, train, model_path, train_pooled
            )
        else:
            # if discrete quality labels
            # train supervised-CEBRA with discrete quality label
            model_lyr_ev_20Khz = CebraSpike1.get_layerwise_model(model_cfg, train, model_path, train_layerwise, layers)
            model_pool_ev_20Khz = CebraSpike1.get_pooled_model(model_cfg, train, model_path, train_pooled)
        return model_lyr_ev_20Khz, model_pool_ev_20Khz


    def decode(embed_train, embed_test, label_train, label_test):
        """decoding using a k-Nearest Neighbor clustering technique
        We use the fixed number of neighbors 2
        """
        # predict
        decoder = cebra.KNNDecoder(n_neighbors=2, metric="cosine")
        decoder.fit(embed_train, label_train)
        prediction = decoder.predict(embed_test)

        # calculate performance metrics
        # precision and recall are for label 1 ("good" units)
        accuracy = sklearn.metrics.accuracy_score(label_test, prediction)
        bal_accuracy = sklearn.metrics.balanced_accuracy_score(label_test, prediction)
        precision = sklearn.metrics.precision_score(label_test, prediction, pos_label=1)
        recall = sklearn.metrics.recall_score(label_test, prediction, pos_label=1)
        f1_score = sklearn.metrics.f1_score(label_test, prediction, pos_label=1)
        mae = np.median(abs(prediction - label_test))
        r2 = sklearn.metrics.r2_score(label_test, prediction)
        return {
            "metrics": {
                "mae": mae,
                "r2": r2,
                "accuracy": accuracy,
                "bal_accuracy": bal_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            },
            "prediction": prediction,
        }


    def train_test_eval(
        model_cfg,
        train: bool,
        model_path: str,
        train_layerwise,
        test_pooled,
        train_pooled,
        show_dims=[0, 1, 2],
        tight_layout_cfg={"pad": 0.001},
        mixed_model: bool = False,
        layers: list = ["l23", "l4", "l5", "l6"]
    ):

        # train on evoked and plot
        model_lyr_ev_20Khz, model_pool_ev_20Khz = CebraSpike1.select_model(
            mixed_model, model_cfg, train, model_path, train_layerwise, train_pooled, layers
        )

        # plot trained embeddings
        # layer-wise
        #fig = plt.figure(figsize=(7, 1.5))
        # fig = mplot.plot_model1_em_by_layer(
        #     fig, train_layerwise, model_lyr_ev_20Khz, show_dims, [-1, 1]
        # )
        #fig.tight_layout(**tight_layout_cfg)

        # pooled
        #fig = plt.figure(figsize=(1.5, 1.5))
        # ax = mplot.plot_model2_em_by_layer(
        #     fig, train_pooled, model_pool_ev_20Khz["embedding"], show_dims, [-1, 1]
        # )
        #ax.set_title("Trained pooled")

        # check training convergence of the pooled model
        #fig = plt.figure(figsize=(1.5, 1.5))
        #ax = plt.subplot(111)
        # ax = cebra.plot_loss(
        #     model_pool_ev_20Khz["model"],
        #     color=[0.5, 0.5, 0.5],
        #     label="pooled model",
        #     ax=ax,
        # )

        # disconnect axes (R style)
        #ax.set_title("Pooled model's loss function")
        #ax.spines[["right", "top"]].set_visible(False)
        #ax.spines["bottom"].set_position(("axes", -0.05))
        #ax.yaxis.set_ticks_position("left")
        #ax.spines["left"].set_position(("axes", -0.05))
        #ax.spines["right"].set_visible(False)

        # check training convergence of the layer models
        #fig, ax = plt.subplots(figsize=(1.5, 1.5))
        # if "l23" in layers:
        #     ax = cebra.plot_loss(
        #         model_lyr_ev_20Khz["model_l23"], color=[1, 0.8, 0.8], label="l23", ax=ax
        #     )
        # if "l4" in layers:            
        #     ax = cebra.plot_loss(
        #         model_lyr_ev_20Khz["model_l4"], color=[0.6, 1, 0.6], label="l4", ax=ax
        #     )
        # if "l5" in layers:
        #     ax = cebra.plot_loss(
        #         model_lyr_ev_20Khz["model_l5"], color=[0.4, 0.4, 1], label="l5", ax=ax
        #     )
        # if "l6" in layers:
        #     ax = cebra.plot_loss(
        #         model_lyr_ev_20Khz["model_l6"], color=[0.2, 1, 0.2], label="l6", ax=ax
        #     )

        # disconnect axes (R style)
        # ax.set_title("Layer-wise loss functions")
        # ax.spines[["right", "top"]].set_visible(False)
        # ax.spines["bottom"].set_position(("axes", -0.05))
        # ax.yaxis.set_ticks_position("left")
        # ax.spines["left"].set_position(("axes", -0.05))
        # ax.spines["right"].set_visible(False)

        # DECODE SPONT **************************

        # pooled ************
        # get the spontaneous embedding from the pooled evoked model
        emb_pool_sp_20Khz = model_pool_ev_20Khz["model"].transform(test_pooled["data"])
        # plot embeddings
        # pooled
        #fig = plt.figure(figsize=(1.5, 1.5))
        # ax = mplot.plot_model2_em_by_layer(
        #     fig, test_pooled, emb_pool_sp_20Khz, show_dims, [-1, 1]
        # )
        #ax.set_title("Test pooled")
        # assign color to each quality label
        # (good "r", poor "k")
        #colr = np.array(["None"] * len(test_pooled["label"]))
        #colr[test_pooled["label"] == 1] = "r"
        #colr[test_pooled["label"] == 0] = "k"
        #ax.view_init(20, 45, 0)  # elevation, azimuth, roll
        # scat = ax.scatter(
        #     emb_pool_sp_20Khz[:, 0],
        #     emb_pool_sp_20Khz[:, 1],
        #     emb_pool_sp_20Khz[:, 2],
        #     c=colr,
        #     edgecolors="w",
        #     linewidths=0.2,
        #     s=20,
        # )

        # evaluate decoding
        eval_rez = CebraSpike1.decode(
            model_pool_ev_20Khz["embedding"],  # pooled evoked embedding
            emb_pool_sp_20Khz,  # pooled spont. embedding
            train_pooled["label"],
            test_pooled["label"],
        )

        print("Model performance metrics:\n")
        print("pooled:", eval_rez["metrics"])

        # diagnostic (error analysis)
        #fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
        #ax.plot(eval_rez["prediction"], "r:")
        #ax.plot(test_pooled["label"])
        return {
            "model_lyr_ev": model_lyr_ev_20Khz,
            "model_pool_ev": model_pool_ev_20Khz,
            "emb_pool_sp": emb_pool_sp_20Khz,
            "eval_rez": emb_pool_sp_20Khz,
        }
        

    def train_test_eval_layerwise(
        model_lyr_ev_20Khz,
        data_lyr_sp_20Khz,
        data_lyr_ev_20Khz,
        show_dims,
        tight_layout_cfg: dict,
    ):
        # layer-wise ************

        # get the spontaneous embedding from the pooled evoked model
        em_lyr_sp_20Khz = {}
        l23 = model_lyr_ev_20Khz["model_l23"].transform(data_lyr_sp_20Khz["data_l23"])
        l4 = model_lyr_ev_20Khz["model_l4"].transform(data_lyr_sp_20Khz["data_l4"])
        l5 = model_lyr_ev_20Khz["model_l5"].transform(data_lyr_sp_20Khz["data_l5"])
        l6 = model_lyr_ev_20Khz["model_l6"].transform(data_lyr_sp_20Khz["data_l6"])
        em_lyr_sp_20Khz["l23"] = l23
        em_lyr_sp_20Khz["l4"] = l4
        em_lyr_sp_20Khz["l5"] = l5
        em_lyr_sp_20Khz["l6"] = l6

        # plot test embeddings
        # fig = plt.figure(figsize=(7, 1.5))
        # ax = mplot.plot_model1_em_by_layer(
        #     fig, data_lyr_sp_20Khz, em_lyr_sp_20Khz, show_dims, [-1, 1]
        # )

        # decode
        eval_rez_l23 = CebraSpike1.decode(
            model_lyr_ev_20Khz["l23"],  # evoked embedding
            em_lyr_sp_20Khz["l23"],  # spont embedding
            data_lyr_ev_20Khz["label_l23"],
            data_lyr_sp_20Khz["label_l23"],
        )
        eval_rez_l4 = CebraSpike1.decode(
            model_lyr_ev_20Khz["l4"],  # evoked embedding
            em_lyr_sp_20Khz["l4"],  # spont embedding
            data_lyr_ev_20Khz["label_l4"],
            data_lyr_sp_20Khz["label_l4"],
        )
        eval_rez_l5 = CebraSpike1.decode(
            model_lyr_ev_20Khz["l5"],  # evoked embedding
            em_lyr_sp_20Khz["l5"],  # spont embedding
            data_lyr_ev_20Khz["label_l5"],
            data_lyr_sp_20Khz["label_l5"],
        )
        eval_rez_l6 = CebraSpike1.decode(
            model_lyr_ev_20Khz["l6"],  # evoked embedding
            em_lyr_sp_20Khz["l6"],  # spont embedding
            data_lyr_ev_20Khz["label_l6"],
            data_lyr_sp_20Khz["label_l6"],
        )
        print("accuracy metrics:\n")
        print("l23:", eval_rez_l23["metrics"])
        print("l4:", eval_rez_l4["metrics"])
        print("l5:", eval_rez_l5["metrics"])
        print("l6:", eval_rez_l6["metrics"])

        # # diagnostic (error analysis)
        # fig, ax = plt.subplots(1, 4, figsize=(5.5, 1.5))
        # # l2/3
        # ax[0].plot(eval_rez_l23["prediction"], "r:")
        # ax[0].plot(data_lyr_sp_20Khz["label_l23"])
        # # l4
        # ax[1].plot(eval_rez_l4["prediction"], "r:")
        # ax[1].plot(data_lyr_sp_20Khz["label_l4"])
        # # l5
        # ax[2].plot(eval_rez_l5["prediction"], "r:")
        # ax[2].plot(data_lyr_sp_20Khz["label_l5"])
        # fig.tight_layout(**tight_layout_cfg)
        # # l6
        # ax[3].plot(eval_rez_l6["prediction"], "r:")
        # ax[3].plot(data_lyr_sp_20Khz["label_l6"])
        # fig.tight_layout(**tight_layout_cfg)
        
        return {
            "eval_rez_l23": eval_rez_l23,
            "eval_rez_l4": eval_rez_l4,
            "eval_rez_l5": eval_rez_l5,
            "eval_rez_l6": eval_rez_l6,
            "em_lyr_sp": em_lyr_sp_20Khz,
        }
        

    def plot_em(
        ax, CebraL4_em, quality_label, dims: list = [0, 1, 2], xlim: list = [-1, 1]
    ):
        """plot the embedding, on which dots are
        sorted units colored by sorting quality
        (good in "red", poor in "black")

        Args:
            ax (_type_): axis
            CebraL4_em (np.array): embedding
            quality_label (np.array): quality labels
            - (1: good, 0: poor)

        Returns:
            scat: plot handle
        """
        # set color for good units in red, poor in black
        colr = np.array(["None"] * len(quality_label))
        colr[quality_label == 1] = "r"
        colr[quality_label == 0] = "k"

        # plot
        ax.view_init(20, 45, 0)  # elevation, azimuth, roll
        scat = ax.scatter(
            CebraL4_em[:, dims[0]],
            CebraL4_em[:, dims[1]],
            CebraL4_em[:, dims[2]],
            c=colr,
            edgecolors="w",
            linewidths=0.2,
            s=20,
        )
        # aesthetics
        # disconnect axes (R style)
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines["bottom"].set_position(("axes", -0.05))
        ax.spines["left"].set_position(("axes", -0.05))
        ax.spines["right"].set_visible(False)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_zlim(xlim)
        return scat

        
    def plot_model1_em_by_layer(fig, dataset, em, show_dims=[0, 1, 2], xlim=[-1, 1]):

        # L2/3
        ax = fig.add_subplot(1, 4, 1, projection="3d")
        scat = CebraSpike1.plot_em(ax, em["l23"], dataset["label_l23"], show_dims, xlim)
        ax.set_title("L2/3")
        # L4
        ax = fig.add_subplot(1, 4, 2, projection="3d")
        scat = CebraSpike1.plot_em(ax, em["l4"], dataset["label_l4"], show_dims, xlim)
        ax.set_title("L4")
        # L5
        ax = fig.add_subplot(1, 4, 3, projection="3d")
        scat = CebraSpike1.plot_em(ax, em["l5"], dataset["label_l5"], show_dims, xlim)
        ax.set_title("L5")
        # L6
        ax = fig.add_subplot(1, 4, 4, projection="3d")
        scat = CebraSpike1.plot_em(ax, em["l6"], dataset["label_l6"], show_dims, xlim)
        ax.set_title("L6")
        return fig
