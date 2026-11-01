"""Models that curate the "good" sorted single-units (above 80% sorting accuracy), evaluated with nested
cross-validation: the two best models found so far.

* a logistic regression on the 10 quality metrics + the firing pattern features (interpretable)
* an extra-trees classifier on the 10 quality metrics + all the new features (best overall)

The decision threshold of each model is chosen on the training units only (5-fold out-of-fold
probabilities): the lowest threshold whose out-of-fold precision reaches TARGET_PRECISION."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics as skm
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from src.nodes.models.Flc import extra_features as ef

LABEL = "quality_label"
TARGET_PRECISION = 0.92
LOGISTIC = "logistic regression"
TREES = "extra trees"


def feature_sets(base_predictors):
    """features of each model"""
    all_new = [c for cols in ef.GROUPS.values() for c in cols]
    return {
        LOGISTIC: list(base_predictors) + ef.GROUPS["firing pattern"],
        TREES: list(base_predictors) + all_new,
    }


def pipeline(kind):
    if kind == LOGISTIC:
        # features are replaced by their percentile rank (uniform on [0, 1]): the model is linear in the rank, so the
        # extreme values of the heavy-tailed features cannot dominate (a normal-score transform gave a much lower recall)
        qt = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
        return make_pipeline(qt, LogisticRegression(C=0.3, max_iter=5000))
    qt = QuantileTransformer(n_quantiles=100, output_distribution="normal")     # heavy-tailed features
    return make_pipeline(qt, ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, random_state=0))


def calibrate(p_oof, y_true, target=TARGET_PRECISION, min_pred=5):
    """lowest threshold whose out-of-fold precision reaches the target (largest recall)"""
    order = np.argsort(-p_oof)
    ys, ps = y_true[order], p_oof[order]
    tp = np.cumsum(ys)
    n = np.arange(1, len(ys) + 1)
    ok = np.where((tp / n >= target) & (n >= min_pred))[0]
    return ps[ok[-1]] if len(ok) else ps[min(min_pred, len(ps)) - 1]


def _one_fold(k, tr, te, X_by_model, y):
    rows = []
    for kind, X in X_by_model.items():
        est = pipeline(kind)
        oof = np.zeros(len(tr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=k).split(X[tr], y[tr]):
            oof[b] = clone(est).fit(X[tr][a], y[tr][a]).predict_proba(X[tr][b])[:, 1]
        thr = calibrate(oof, y[tr])
        p = clone(est).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        pred = (p >= thr).astype(int)
        rows.append(dict(fold=k, model=kind, precision=skm.precision_score(y[te], pred, zero_division=0),
                         recall=skm.recall_score(y[te], pred, zero_division=0),
                         average_precision=skm.average_precision_score(y[te], p), threshold=thr))
    return rows


def nested_cv(dataset, base_predictors, models=None, n_repeats=10, n_jobs=1):
    """per-fold precision and recall of the two models (repeated stratified 4-fold: 75% train / 25% test).
    models: list of models to evaluate (default: all of them, [LOGISTIC, TREES]).
    n_jobs=1 by default: in a notebook that already started worker processes (spikeinterface) the joblib workers fail with
    BrokenProcessPool; the 40 folds take a few minutes sequentially."""
    y = dataset[LABEL].values.astype(int)
    X_by_model = {kind: dataset[cols].values for kind, cols in feature_sets(base_predictors).items() if models is None or kind in models}
    folds = list(RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=0).split(np.zeros(len(y)), y))
    res = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(k, tr, te, X_by_model, y) for k, (tr, te) in enumerate(folds))
    return pd.DataFrame([r for rows in res for r in rows])


def summarize(results):
    """median and 95% CI (of the mean) of precision and recall over the folds"""
    ci = lambda x: 1.96 * np.std(x) / np.sqrt(len(x))
    rows = []
    for kind, d in results.groupby("model", sort=False):
        rows.append({"model": kind, "n_folds": len(d),
                     "precision (median)": d.precision.median(), "precision 95% CI": ci(d.precision),
                     "recall (median)": d.recall.median(), "recall 95% CI": ci(d.recall),
                     "average precision (median)": d.average_precision.median()})
    return pd.DataFrame(rows).set_index("model").round(3)
