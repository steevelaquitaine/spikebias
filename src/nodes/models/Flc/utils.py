
import statsmodels.api as sm
import random
from statsmodels.tools.validation import float_like
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import logging
import logging.config
import yaml
import copy

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def get_crossval_metrics(
    dataset: pd.DataFrame,
    model_formula: str,
    split_ratio: float = 0.75,
    seeds: np.array = np.arange(0, 100, 1),
    thresh: float = 0.8,
    scale_data=False,
):
    """Calculate cross-validated mcfadden pseudo r-squared
    on test dataset

    Args:
        model_formula (str): glm model formula
        split_ratio (float, optional): _description_. Defaults to 0.75.
        seeds (np.array, optional): _description_. Defaults to np.arange(0, 100, 1).

    Returns:
        np.array: mcfadden pseudo r-squared
    """
    results = []
    for seed in range(len(seeds)):
        result = get_single_fold_flclassifier_metrics(
            model_formula,
            dataset,
            split_ratio=split_ratio,
            seed=seed,
            thresh=thresh,
            scale_data=scale_data,
        )
        results.append(result)
    return np.array(results)


def get_single_fold_flclassifier_metrics(
    model_formula: str,
    dataset: pd.DataFrame,
    split_ratio: float = 0.75,
    seed: int = 0,
    thresh: float = 0.8,
    scale_data=False,
):
    """Calculate the cross-validated precisions
    and recalls of a fractional logistic classifier
    trained to predict high-quality unit label-

    Args:
        model_formula

    Returns:
        precisions and recalls for each fold
    """
    # GET MCF R2 FOR TEST MODEL
    random.seed(seed)

    # unit-test
    # this is a ground truth label that should not
    # be in the dataset
    assert (
        not "sorting_accuracy" in dataset.columns
    ), "drop sorting_accuracy from dataset"

    # TRAIN -----------
    # calculate 75% of train
    n_train = np.round(split_ratio * dataset.shape[0]).astype(int)

    # sample n_train
    indices = np.arange(0, dataset.shape[0], 1).tolist()
    train_indices = random.sample(indices, n_train)
    train_dataset = dataset.iloc[train_indices, :]

    # apply scaling
    if scale_data:
        standard_scaler = StandardScaler()
        predictors = dataset.columns.tolist()
        predictors.remove("quality_label")
        train_dataset[predictors] = standard_scaler.fit_transform(
            train_dataset[predictors]
        )

    # train model on this fold
    try:
        model_1 = sm.GLM.from_formula(
            model_formula,
            family=sm.families.Binomial(),
            data=train_dataset,
        )
        result_1 = model_1.fit()
    except:
        raise ValueError("Model formula is wrong")

    # TEST -----------
    # create test dataset with remaining instances
    # make sure to drop quality label from test dataset
    test_indices = list(set(indices) - set(train_indices))
    test_dataset = dataset.iloc[test_indices, :]
    test_label = test_dataset["quality_label"]
    test_dataset = test_dataset.drop(columns=["quality_label"])

    # apply scaling
    if scale_data:
        test_dataset[predictors] = standard_scaler.fit_transform(
            test_dataset[predictors]
        )

    # unit-test
    assert not "quality_label" in test_dataset.columns, "drop quality label from test"

    # reorder test dataset features and add intercept
    # to make predictions
    features = result_1.params.index[1:]
    test_features = test_dataset.loc[:, features]
    test_features.insert(0, "intercept", 1)

    # unit-test
    assert not "quality_label" in features, "drop quality label from features"

    # predict -------------
    # thresholded binary predictions
    predictions = (result_1.predict(test_features) >= thresh).astype(int)
    precision = metrics.precision_score(test_label, predictions)
    recall = metrics.recall_score(test_label, predictions)
    return {"precision": precision, "recall": recall}


def get_metrics_stats(metric_data: dict):

    # make arrays
    precisions = []
    recalls = []
    for m_i in metric_data:
        precisions.append(m_i["precision"])
        recalls.append(m_i["recall"])

    # precision
    precision_median = np.nanmedian(precisions)
    precision_std = np.nanstd(precisions)
    precision_ci95 = 1.96 * np.std(precisions) / np.sqrt(len(precisions))

    # recall
    recall_median = np.nanmedian(recalls)
    recall_std = np.nanstd(recalls)
    recall_ci95 = 1.96 * np.std(recalls) / np.sqrt(len(recalls))
    return {
        "precision_median": precision_median, "precision_std": precision_std,
        "precision_ci95": precision_ci95, "recall_median": recall_median, 
        "recall_std": recall_std, "recall_ci95": recall_ci95}
    
    
def evaluate_on_full_dataset(
    model_formula: str,
    dataset: pd.DataFrame,
    scale_data=False,
    regularization="elastic_net",
    thresh=0.8,
    maxiter=100,
    cnvrg_tol=1e-10,
    verbose=False
):
    """Calculate precision and recall metrics for a single fold, sampling
    split_ratio instances of the dataset as train and 1-split_ratio as test

    Args:
        model_formula

    note:
        - only the elastic_net regularization method is currently implemented

    Returns:
        mcfadden pseudo r-squared (float)
    """
    label = "quality_label"

    # apply scaling
    if scale_data:
        logger.info("Z-scoring the dataset features...")
        dataset2 = copy.copy(dataset)
        standard_scaler = StandardScaler()
        predictors = dataset.columns.tolist()
        predictors.remove(label)
        dataset2[predictors] = standard_scaler.fit_transform(
            dataset[predictors]
        )
        logger.info("Done Z-scoring.")
    train_dataset = dataset2
    test_dataset = dataset2
        
    # train model on this fold
    try:
        # instantiate model
        model = sm.GLM.from_formula(
            model_formula,
            family=sm.families.Binomial(),
            data=train_dataset,
        )        
        # train the model
        # regularization does not produce pvalues (not implemented
        # in this version of statsmodel), but fit() does. So we 
        # initialize the fit() function with the regularized weight
        # set only one iteration, which does not change the weight
        # to obtain the pvalues.
        # Lasso regularization produces more well behaved weights, fit()
        # alone sometimes produce large weights and -inf r-squared, which 
        # are fixed by lasso regularization that may be due to feature 
        # correlations
        if verbose:
            logger.info("Training the model with lasso regularization...")
        result = model.fit_regularized(method=regularization, 
                                    maxiter=maxiter,
                                    cnvrg_tol=cnvrg_tol)            
        result = model.fit(start_params=result.params, maxiter=1)        
        logger.info("Done training the model.")
    except:
        if verbose:
            print("The model could not handle this fold -> returning nan")
        return {"model": np.nan, "r-squared": np.nan}

    # setup test dataset
    test_label = test_dataset[label]
    test_dataset = test_dataset.loc[:, predictors]
    test_dataset.insert(0, "intercept", 1)
    
    # evaluate *******************
    # unit-test
    assert not "quality_label" in test_dataset.columns, "drop quality label from test"

    # reorder test dataset features and add intercept
    # to make predictions
    features = result.params.index[1:]
    test_features = test_dataset.loc[:, features]
    test_features.insert(0, "intercept", 1)

    # unit-test
    assert not "quality_label" in features, "drop quality label from features"

    # predict -------------
    # thresholded binary predictions
    predictions = (result.predict(test_features) >= thresh).astype(int)
    precision = metrics.precision_score(test_label, predictions)
    recall = metrics.recall_score(test_label, predictions)
    return {"precision": precision, "recall": recall, "model": result}
    