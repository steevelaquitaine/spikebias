"""
"""

import os
import spikeinterface as si
import spikeinterface.core.template_tools as ttools
from spikeinterface import comparison
from spikeinterface.qualitymetrics import compute_quality_metrics as qm
from spikeinterface import qualitymetrics
import pandas as pd
import numpy as np
import sklearn
import shutil
from src.nodes import utils
from src.nodes.models.Fr import utils as mutils


class FrModel(object):
    """
    """
    def __init__(self, predictors:list, label:str, order=1):
        """Instantiate the model

        Args:
            predictors (list): list of predictor features
            label (str): the target label to predict
            order (int, optional): _description_. Defaults to 1.
            1: calculate the main effects (weight of each feature)
            2: calculate the pairwise (2nd order) interactions 
            (not yet implemented)
        """
        
        # parametrize predictors
        self.formula = self.__create_classifier_model(predictors, label, order)
        self.predictors = predictors
        self.order=order
        
        
    def __create_classifier_model(self, predictors, label, order=1):
        """Create model GLM formula to predict
        label (Binomial distribution., e.g., proportions) 
        from predictors
        """
        # create main effects
        variables = ""
        for predictor in predictors:
            variables += " + " + str(predictor)
        
        # keep main effect only
        if order == 1:
            return f"""{label} ~ 1 {variables}"""
        elif order == 2:
            raise NotImplementedError
            # add all pairwise interactions
            #predictors2 = [[p_i] for p_i in predictors]
            #order2 = ""
            #for pair in mutils.pairs(*predictors2):
            #    order2 += " + " + str(pair[0]) + ":" + str(pair[1])
            #return f"""{label} ~ 1 {variables}{order2}"""

  
    def evaluate(self, dataset: pd.DataFrame, split=0.75, 
                 seeds=np.arange(0, 100, 1), scale_data=False, 
                 regularization="elastic_net", maxiter=100,
                 cnvrg_tol=1e-10, verbose=False):
        """Calculate the cross-validated performance metrics 
        (r-squared)
        """        
        # get cross-validated performance metrics
        metric_data = mutils.get_crossval_mcf_r2(
            dataset, self.formula, split_ratio=split,
            seeds=seeds, scale_data=scale_data,
            regularization=regularization, maxiter=maxiter, 
            cnvrg_tol=cnvrg_tol, verbose=verbose
        )

        # calculate metrics stats
        metric_stats = mutils.get_metrics_stats(metric_data)
        return {"metric_stats": metric_stats, "metric_data": metric_data}


    def evaluate_on_full_dataset(self, dataset: pd.DataFrame, 
                                 scale_data=False, 
                                 regularization="elastic_net",
                                 maxiter=100, 
                                 cnvrg_tol=1e-10, verbose=False):
            """train and test on full dataset and get 
            performance metrics (r-squared)
            """
            # get cross-validated performance metrics
            metric_data = mutils.evaluate_on_full_dataset(
                self.formula, dataset, 
                scale_data=scale_data,
                regularization=regularization, maxiter=maxiter, 
                cnvrg_tol=cnvrg_tol, verbose=verbose
            )

            return {"metric_data": metric_data}