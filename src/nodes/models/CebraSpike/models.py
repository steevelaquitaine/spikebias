
"""CebraSpike models

author: steeve.laquitaine@epfl.ch

Returns:
    _type_: _description_
"""

import pandas as pd
import numpy as np
from src.nodes.models.CebraSpike import utils as mutils


class CebraSpike(object):
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        
        
    def evaluate(
        self, dataset: dict, model_path: str, 
        is_train: bool=False, seeds=np.arange(0, 100, 1), n_neighbors: int=2
        ):
        """cross-validated evaluation of 
        classification performance with precision and recall metrics
        """
        # get metric data for all cross-validated samples
        metric_data = mutils.get_crossval_metrics(self.cfg, dataset, model_path, is_train, seeds, n_neighbors)

        # calculate metric stats
        metric_stats = mutils.get_metrics_stats(metric_data)
        return {"metric_stats": metric_stats, "metric_data": metric_data}