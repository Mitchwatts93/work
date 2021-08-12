"""functions for running evaluation of predictions"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR)
from misc import constants

################################################################################
# metrics

def get_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """get the precision score using sklearn precision. Does NOT use recommender
    type precision using lists (i.e. precision @k) as this was not the result 
    requested."""
    prec = metrics.precision_score(y_true=y_true, y_pred=y_pred)
    return prec


def get_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """get the recall score using sklearn precision. Does NOT use recommender
    type recall using lists (i.e. recall @k) as this was not the result 
    requested."""
    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
    return recall


def get_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """get the f1 score using sklearn precision."""
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)
    return f1


def get_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """get the area under the receiver operator characteristic curve."""
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    return auc


################################################################################

def convert_pred_prob_to_bool(
    preds: np.ndarray, threshold: float = constants.default_threshold
) -> np.ndarray:
    """convert the predicted probabilities to boolean predictions based on the 
    threshold. values less than or equal to the threshold will be converted to
    False, greater than to True."""
    bool_preds = np.where(preds <= threshold, False, True)
    return bool_preds

################################################################################
# metrics dict to fetch

metric_fun_dict = {
    "f1":get_f1_score,
    "precision":get_precision,
    "recall":get_recall,
    "auc_roc":get_auc_roc,
} # keys will be keys for results, values are functions to get metrics

################################################################################

def merge_predictions_labels(
    predictions: pd.DataFrame, labels: pd.DataFrame, 
):
    """get a dictionary of metrics of interest (f1, prec, recall, auc_roc) for
    a set of predictions and labels. predictions and labels will be merged 
    using an inner product on productId and customerId, any missing values
    will not be evaluated.
    Args:
        predictions: pandas df with customer ids in column with 
            name=constants.customer_id_str, product ids in column with 
            name=constants.product_id_str. Prediction probabilities [0-1] 
            stored in column with name=constant.probabilities_str.
        labels: pandas df with customer ids in column with 
            name=constants.customer_id_str, product ids in column with 
            name=constants.product_id_str. Actual purchase boolean values 
            stored in column with name=constant.purchased_label_str.
    Returns: 
        final_results: pd dataframe with real values as 
            constants.purchased_label_str and predicted values as 
            constants.predicted_purchased_str
    """
    missing_preds = labels.loc[labels.index.difference(predictions.index), :] # 
    # missing predictions
    missing_preds.loc[:, constants.probabilities_str] = constants.default_value_missing # for missing 
    # predictions, we just predict value of constants.default_value_missing

    predictions_full = pd.concat([predictions, missing_preds])
    
    predictions_full.rename({constants.probabilities_str:constants.predicted_purchased_str}, inplace=True, axis=1)
    final_results = pd.merge(
        labels, 
        predictions_full, 
        on=[constants.product_id_str, constants.customer_id_str], 
        how='inner'
    )
    return final_results


def get_predictions_labels(
    predictions: pd.DataFrame, labels: pd.DataFrame, 
    threshold: float = constants.default_threshold
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the predictions and labels dataframes into numpy arrays with the 
    predictions cast as bools according to the threshold float, where <= is set 
    to False, and > is True.
    Args:
        predictions: pandas df with customer ids in column with 
            name=constants.customer_id_str, product ids in column with 
            name=constants.product_id_str. Prediction probabilities [0-1] 
            stored in column with name=constant.probabilities_str.
        labels: pandas df with customer ids in column with 
            name=constants.customer_id_str, product ids in column with 
            name=constants.product_id_str. Actual purchase boolean values 
            stored in column with name=constant.purchased_label_str.
        threshold: probabilities below or equal to threshold will be cast to 
            False, greater than will be True.
    Returns:
        reals, preds: both boolean np arrays. Actual and predicted labels.
    """
    final_results = merge_predictions_labels(
        labels=labels, predictions=predictions
    )

    reals = final_results.loc[:, constants.purchased_label_str].values # the 
    # boolean labels as np array
    preds = final_results.loc[:, constants.predicted_purchased_str].values # the
    # predicted probabilities as np array
    
    threshold_preds = convert_pred_prob_to_bool(preds, threshold=threshold) # 
    # cast probabilities to bols accoridng to threshold
    return reals, threshold_preds


def get_metric_dict(
    predictions: pd.DataFrame, labels: pd.DataFrame, 
    threshold: float = constants.default_threshold
) -> Dict:
    """get a dictionary of metrics of interest (f1, prec, recall, auc_roc) for
    a set of predictions and labels. predictions and labels will be merged 
    using an inner product on productId and customerId, any missing values
    will not be evaluated.
    Args:
        predictions: pandas df with customer ids in column with 
            name=constants.customer_id_str, product ids in column with 
            name=constants.product_id_str. Prediction probabilities [0-1] 
            stored in column with name=constant.probabilities_str.
        labels: pandas df with customer ids in column with 
            name=constants.customer_id_str, product ids in column with 
            name=constants.product_id_str. Actual purchase boolean values 
            stored in column with name=constant.purchased_label_str.
        threshold: probabilities below or equal to threshold will be cast to 
            False, greater than will be True.
    Returns:
        metric_results_dict: dictionary with keys equal to 
            metric_fun_dict.keys(). Values the corresponding metrics calculated 
            from supplied predictions and labels.
    """
    reals, preds = get_predictions_labels(
        labels=labels, predictions=predictions, threshold=threshold
    )

    metric_results_dict = {} # now get all the metrics
    for metric_key in metric_fun_dict.keys():
        metric_fun_dict[metric_key](y_true=reals, y_pred=preds)

    return metric_results_dict


################################################################################

if __name__ == "__main__":
    metric_dict = get_metric_dict()
    print(metric_dict)
