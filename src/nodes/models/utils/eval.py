import numpy as np

def get_tp_tn_fp_fn(prediction, label):
    """evaluate individual predictions as
    TP, TN, FP, FN

    Args:
        prediction (np.array): model prediction (1 or 0)
        label (np.array): actual label (1 or 0)

    Returns:
        np.array[str]: array of TP, TN, ...
    """

    model_eval = np.zeros(len(label)).astype(str)

    # true positives
    model_eval[(prediction == 1) & (label == 1)] = str("TP")
    # true negatives
    model_eval[(prediction == 0) & (label == 0)] = str("TN")
    # false positive
    model_eval[(prediction == 1) & (label == 0)] = str("FP")
    # false negative
    model_eval[(prediction == 0) & (label == 1)] = str("FN")
    return model_eval