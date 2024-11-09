import numpy as np
from src.nodes import utils
from cebra import CEBRA
import cebra
import sklearn
import pickle 

def save_dict(data, file_path: str):
    
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
def cv_split(label, split, seed):

    # ensure reproducibility
    np.random.seed(seed)

    # get good unit indices
    good_ix = np.where(label)[0]
    n_tr = int(np.floor(len(good_ix) * split))
    shuffled = np.random.permutation(good_ix)
    g_tr_ix = shuffled[:n_tr]

    # get poor unit indices
    poor_ix = np.where(label == 0)[0]
    n_tr = int(np.floor(len(poor_ix) * split))
    shuffled = np.random.permutation(poor_ix)
    p_tr_ix = shuffled[:n_tr]

    # get train and test indices
    tr_ix = np.hstack([g_tr_ix, p_tr_ix])
    all_ix = np.arange(0, len(label), 1)
    test_ix = np.where(~np.isin(all_ix, tr_ix))[0]
    return tr_ix, test_ix

def fit(model_cfg: dict, dataset):
    """train discrete label-supervised CEBRA
    
    Args
        dataset (dict):
        - "data": spike data
        - "label": supervised labels
        max_iter (int): number of training iterations

    Returns:

    """
    # instantiate model
    CebraPooled = CEBRA(**model_cfg)

    # train model
    CebraPooled.fit(dataset["data"], dataset["label"])

    # get embedding
    CebraPooled_em = CebraPooled.transform(dataset["data"])
    return {"model": CebraPooled, "embedding": CebraPooled_em}


def fit_or_load(model_cfg: dict, train: bool, model_path: str, dataset):
    """train discrete label-supervised CEBRA
    
    TODO: MOVE TO CebraSpike model module
    
    Args
        train (bool)
        model_path (str): model load/save path
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



def decode(embed_train, embed_test, label_train:int, label_test:int, n_neighbors:int=2):
    """decoding using a k-Nearest Neighbor clustering technique
    We use the fixed number of neighbors 2
    """
    
    # ensure labels are integers
    # else the knn can return continuous values
    label_train= label_train.astype(int)
    label_test= label_test.astype(int)
    
    # predict
    decoder = cebra.KNNDecoder(n_neighbors=n_neighbors, metric="cosine")

    # train kNN on training embedding
    decoder.fit(embed_train, label_train)

    # decode test embedding
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
    
    
def get_crossval_metrics(cfg: dict, dataset: dict, save_path: str,
                         is_train=False, seeds=np.arange(0, 10, 1),
                         n_neighbors: int=2):
    """_summary_

    Args:
        cfg (dict): model parameters
        dataset (dict): _description_
        save_path (str): cross-validation data, including models
        nfolds (_type_): _description_
    """

    # cross-validation: loop over nfolds random
    # seeds to get generate different subsets of
    # train/test units    
    # precision = []
    # recall = []
    # test_labels = []
    # predictions_all = []
    # models = []
    # tr_ix_all = []
    # test_ix_all = []
    metric_data = []
    
    # loop over sets
    for seed_i in seeds:

        # get train and test
        tr_ix, test_ix = cv_split(dataset["label"], split=0.8, seed=seed_i)

        # training data
        train_data = dict()
        train_data["data"] = dataset["data"][tr_ix, :]
        train_data["label"] = dataset["label"][tr_ix]
        train_data["unit_ids"] = dataset["unit_ids"][tr_ix]

        # test data
        test_data = dict()
        test_data["data"] = dataset["data"][test_ix, :]
        test_data["label"] = dataset["label"][test_ix]
        test_data["unit_ids"] = dataset["unit_ids"][test_ix]

        # train *******************

        model = fit(cfg, train_data)

        # decode *******************

        # get embeddings
        train_embed = model["embedding"]
        test_embed = model["model"].transform(test_data["data"])

        # decode
        eval_rez = decode(
            train_embed,
            test_embed,
            train_data["label"],
            test_data["label"],
            n_neighbors=n_neighbors
        )

        # record
        metric_data.append(
            {"precision": eval_rez["metrics"]["precision"], 
             "recall": eval_rez["metrics"]["recall"],
             "model": model,
             "train_unit_ix": tr_ix,
             "test_unit_ix": test_ix,
             "test_label": test_data["label"],
             "predictions": eval_rez["prediction"]
             }
            )
    
    # save
    save_dict(metric_data, save_path)
    return metric_data


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