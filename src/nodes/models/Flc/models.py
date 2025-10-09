"""
"""

import os
import spikeinterface as si
import spikeinterface.core.template_tools as ttools
from spikeinterface import comparison
from spikeinterface.qualitymetrics import compute_quality_metrics as qm
from spikeinterface import qualitymetrics
import pandas as pd
# from cebra import CEBRA
import numpy as np
import sklearn
import seaborn as sns
# import cebra.models
import shutil
from src.nodes import utils
from src.nodes.models.Flc import utils as mutils


class FlcModel(object):
    """
    """
    def __init__(self, predictors):
        
        # parametrize predictors
        self.formula = self.__create_classifier_model(predictors)
        self.predictors = predictors
        
        
    def __create_classifier_model(self, predictors):
        """Create model GLM formula to predict
        dataset column named "quality_label"
        """
        variables = ""
        for predictor in predictors:
            variables += " + " + str(predictor)
        return f"""quality_label ~ 1 {variables}"""
    
    
    def evaluate(self, dataset: pd.DataFrame, seeds=np.arange(0,100,1), scale_data=False):
        """cross-validated
        """
        
        # get performance metrics        
        metric_data = mutils.get_crossval_metrics(
            dataset=dataset,
            model_formula=self.formula,
            split_ratio=0.75,
            seeds=seeds,
            thresh=0.8,
            scale_data=scale_data,
        )
        # calculate metrics stats
        metric_stats = mutils.get_metrics_stats(metric_data)
        
        return {"metric_stats": metric_stats, "metric_data": metric_data}
    
        
    def evaluate_on_full_dataset(self, dataset: pd.DataFrame, 
                                    thresh=0.8,
                                    scale_data=False, 
                                    regularization="elastic_net",
                                    maxiter=100, 
                                    cnvrg_tol=1e-10, 
                                    verbose=False):
            """train and test on full dataset and get 
            performance metrics (r-squared)
            """
            # get cross-validated performance metrics
            metric_data = mutils.evaluate_on_full_dataset(
                self.formula, dataset, 
                thresh=thresh,
                scale_data=scale_data,
                regularization=regularization, 
                maxiter=maxiter, 
                cnvrg_tol=cnvrg_tol, 
                verbose=verbose
            )
            return {"metric_data": metric_data}