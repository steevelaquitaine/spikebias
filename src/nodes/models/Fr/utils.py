"""Node functions for the fractional regression model

author: steeve.laquitaine@epfl.ch

Returns:
    _type_: _description_

Yields:
    _type_: _description_
"""

import copy
import statsmodels.api as sm
import pandas as pd
import numpy as np 
from statsmodels.tools.validation import float_like
import random
from sklearn.preprocessing import StandardScaler
from itertools import combinations, product
import itertools
import logging
import logging.config
import yaml

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def get_combinations(lst: list):
    """create combinations
    """
    combination = []
    for r in range(1, len(lst) + 1):
        combination.extend(itertools.combinations(lst, r))
    return combination


def pairs(*lists):
    """create all combination pairs
    of the list items

    Yields:
        _type_: _description_
    """
    for t in combinations(lists, 2):
        for pair in product(*t):
            yield pair


def count_combinations(features: list):
    """count combinations in list

    Args:
        features (list): _description_

    Returns:
        _type_: _description_
    """
    all_combinations = get_combinations(features)
    return len(all_combinations)
   
    
def loglike(fit_output, params, scale: float, exog: np.ndarray, endog: np.array):
    """Calculate the log likelihood of observing the true sorting accuracies "endog"
    given the fitted glm model "fit_output"
    you can get by inspecting result_1.model.loglike
    code = inspect.getsource(result_1.model.loglike)
    print(code)

    Args:
        fit_output: fitted glm model
        exog: predictive features used to make predictions
        - independent variables
        endog: true sorting accuracy (dependent variable)

    Note:
        Setting the args as below should produce llf == fit_output.llf
        that is the log likelihood nproduced from fitting the training
        data
        - exog = fit_output.model.exog
        - endog = fit_output.model.endog
    """
    scale = float_like(scale, "scale", optional=True)
    var_weights = np.ones(exog.shape[0])
    freq_weights = np.ones(exog.shape[0])

    # make predictions
    # - same as calling result.model.predict(params, exog)
    linear_preds = np.dot(exog, params) + fit_output.model._offset_exposure
    expval = fit_output.model.family.link.inverse(linear_preds)
    if scale is None:
        scale = fit_output.model.estimate_scale(expval)

    # calculate loglikelihood of data
    llf = fit_output.model.family.loglike(
        endog,  # true sorting accuracy
        expval,  # predicted sorting accuracy
        var_weights,  # 1 by default
        freq_weights,  # 1 by default
        scale,
    )
    return llf


def get_single_fold_mcf_r2_regularized(
    model_formula: str, dataset: pd.DataFrame, split_ratio: float = 0.75, 
    seed: int = 0, scale_data=False, regularization="elastic_net", 
    maxiter=100, cnvrg_tol=1e-10, verbose=False
):
    """Calculate mcfadden pseudo r-squared for a single fold, sampling
    split_ratio instances of the dataset as train and 1-split_ratio as test

    Args:
        model_formula

    note:
        - mcfadden r2 formula: (1 - result_1.llf / result_1.llnull)
        - produced by statsmodel r2 = test_model.pseudo_rsquared(kind="mcf")
        - only the elastic_net regularization method is currently implemented

    Returns:
        mcfadden pseudo r-squared (float)
    """
    # GET MCF R2 FOR TEST MODEL
    random.seed(seed)
    label = "sorting_accuracy"

    # TRAIN -----------
    # calculate 75% of train
    n_train = np.round(split_ratio * dataset.shape[0]).astype(int)

    # sample n_train
    indices = np.arange(0, dataset.shape[0], 1).tolist()
    train_indices = random.sample(indices, n_train)
    train_dataset = dataset.iloc[train_indices, :]

    # apply scaling
    if scale_data:
        if verbose:
            logger.info("z-scoring dataset..")
        standard_scaler = StandardScaler()
        predictors = dataset.columns.tolist()
        predictors.remove(label)
        train_dataset[predictors] = standard_scaler.fit_transform(
            train_dataset[predictors]
        )
        if verbose:
            logger.info("z-scoring done.")
        
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
        if verbose:
            logger.info("Done training the model.")
    except:
        if verbose:
            logger.info("The model fit could not handle this fold -> returning nan")
        return {"model": np.nan, "r-squared": np.nan}

    # TEST *******************
    # create test dataset with remaining instances
    test_indices = list(set(indices) - set(train_indices))
    test_dataset = dataset.iloc[test_indices, :]

    # apply scaling
    if scale_data:
        test_dataset[predictors] = standard_scaler.fit_transform(
            test_dataset[predictors]
        )
    test_label = test_dataset[label]
    test_dataset = test_dataset.loc[:, predictors]
    test_dataset.insert(0, "intercept", 1)
    
    # evaluate *******************
    # calculate model's loglikelihood
    llf = loglike(
        result,
        result.params,
        None,
        exog=test_dataset,
        endog=test_label,
    )

    # get the r-squared of the null model **********
    # train
    null_model = sm.GLM.from_formula(
        f"{label} ~ 1",
        family=sm.families.Binomial(),
        data=train_dataset,
    )
    null_model = null_model.fit()

    # test and eval
    ll_null = loglike(
        null_model,
        null_model.params,
        None,
        exog=np.array([test_dataset["intercept"]]).T,
        endog=test_label,
    )

    # return r-squared
    if llf > 0 and ll_null == 0:
        # fix r-squared in case ll_null==0
        return {"model": result, "r-squared": np.nan}
    else:
        return {"model": result, "r-squared": 1 - llf / ll_null}


def evaluate_on_full_dataset(
    model_formula: str,
    dataset: pd.DataFrame,
    scale_data=False,
    regularization="elastic_net",
    maxiter=100,
    cnvrg_tol=1e-10,
    verbose=False
):
    """Calculate mcfadden pseudo r-squared for a single fold, sampling
    split_ratio instances of the dataset as train and 1-split_ratio as test

    Args:
        model_formula

    note:
        - mcfadden r2 formula: (1 - result_1.llf / result_1.llnull)
        - produced by statsmodel r2 = test_model.pseudo_rsquared(kind="mcf")
        - only the elastic_net regularization method is currently implemented

    Returns:
        mcfadden pseudo r-squared (float)
    """
    # GET MCF R2 FOR TEST MODEL
    label = "sorting_accuracy"

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
    if verbose:
        logger.info("Calculating r-squared...")
    
    # calculate model's loglikelihood
    llf = loglike(
        result,
        result.params,
        None,
        exog=test_dataset,
        endog=test_label,
    )

    # get the r-squared of the null model **********
    # train
    null_model = sm.GLM.from_formula(
        f"{label} ~ 1",
        family=sm.families.Binomial(),
        data=train_dataset,
    )
    null_model = null_model.fit()

    # test and eval
    ll_null = loglike(
        null_model,
        null_model.params,
        None,
        exog=np.array([test_dataset["intercept"]]).T,
        endog=test_label,
    )
    
    # return r-squared
    if llf > 0 and ll_null == 0:
        # fix r-squared in case ll_null=0
        if verbose:
            logger.info("loglikel > 0 and loglikel_null =0 -> returning nan")
        return {"model": result, "r-squared": np.nan}
    else:
        if verbose:
            logger.info("Done calculating r-squared")
        return {"model": result, "r-squared": 1 - llf / ll_null}
    
    
def get_single_fold_mcf_r2(
    model_formula: str, dataset: pd.DataFrame, 
    split_ratio: float = 0.75, seed: int = 0, 
    verbose=False
):
    """Calculate mcfadden pseudo r-squared for a single fold, sampling
    split_ratio instances of the dataset as train and 1-split_ratio as test

    Args:
        model_formula

    note:
        - mcfadden r2 formula: (1 - result_1.llf / result_1.llnull)
        - produced by statsmodel r2 = test_model.pseudo_rsquared(kind="mcf")

    Returns:
        mcfadden pseudo r-squared (float)
    """
    # GET MCF R2 FOR TEST MODEL

    random.seed(seed)

    # TRAIN -----------
    # calculate 75% of train
    n_train = np.round(split_ratio * dataset.shape[0]).astype(int)

    # sample n_train
    indices = np.arange(0, dataset.shape[0], 1).tolist()
    train_indices = random.sample(indices, n_train)
    train_dataset = dataset.iloc[train_indices, :]

    # train model on this fold
    model_1 = sm.GLM.from_formula(
        model_formula,
        family=sm.families.Binomial(),
        data=train_dataset,
    )
    result_1 = model_1.fit()

    # TEST -----------
    # create test dataset with remaining instances
    test_indices = list(set(indices) - set(train_indices))
    test_dataset = dataset.drop(
        columns=[
            "exc_mini_frequency",
            "dynamics_holding_current",
            "model_template",
        ]
    ).iloc[test_indices, :]

    # reorder test dataset features and add intercept
    # to make predictions and get loglikelihood
    features = result_1.params.index[1:]
    test_features = test_dataset.loc[:, features]
    test_features.insert(0, "intercept", 1)

    # test and eval
    llf = loglike(
        result_1,
        result_1.params,
        None,
        exog=test_features,
        endog=test_dataset["sorting_accuracy"],
    )

    # GET MCF R2 FOR NULL MODEL

    # train
    null_model = sm.GLM.from_formula(
        "sorting_accuracy ~ 1",
        family=sm.families.Binomial(),
        data=train_dataset,
    )
    null_model = null_model.fit()

    # test and eval
    ll_null = loglike(
        null_model,
        null_model.params,
        None,
        exog=np.array([test_features["intercept"]]).T,
        endog=test_dataset["sorting_accuracy"],
    )

    # fix r-squared in case ll_null==0
    if llf > 0 and ll_null == 0:
        return np.nan
    else:
        return 1 - llf / ll_null


def get_crossval_mcf_r2(
    dataset: pd.DataFrame,
    model_formula: str,
    split_ratio: float = 0.75,
    seeds: np.array = np.arange(0, 100, 1),
    scale_data=False,
    regularization="elastic_net",
    maxiter=100,    
    cnvrg_tol=1e-10,
    verbose=False
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
    # note: can fail if SVD did not converge in Linear Least Squares
    # because of wrong covariance matrix
    cv_datas = []
    for seed in range(len(seeds)):
        cv_data = get_single_fold_mcf_r2_regularized(
            model_formula, dataset, split_ratio=split_ratio, 
            seed=seed, scale_data=scale_data,
            regularization=regularization, maxiter=maxiter, 
            cnvrg_tol=cnvrg_tol, verbose=verbose
        )
        cv_datas.append(cv_data)
    return np.array(cv_datas)


def get_metrics_stats(metric_data: dict):
    """Calculate r-squared's median, std, and 95% 
    confidence interval

    Args:
        metric_data (dict): _description_

    Returns:
        _type_: _description_
    """
    r2 = []
    for m_i in metric_data:
        r2.append(m_i["r-squared"])
    r2_median = np.nanmedian(r2)
    r2_std = np.nanstd(r2)
    r2_ci95 = 1.96 * np.nanstd(r2) / np.sqrt(np.sum(~np.isnan(r2)))
    return {
        "r2_median": r2_median, "r2_std": r2_std,
        "r2_ci95": r2_ci95}