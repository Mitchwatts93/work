from typing import Dict

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

def get_precision(y_true, y_pred):
    prec = metrics.precision_score(y_true=y_true, y_pred=y_pred)
    return prec


def get_recall(y_true, y_pred):
    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
    return recall


def get_f1_score(y_true, y_pred):
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)
    return f1


def get_auc_roc(y_true, y_score):
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    return auc


################################################################################

def convert_pred_prob_to_bool(preds, threshold):
    bool_preds = np.where(preds < threshold, False, True)
    return bool_preds

################################################################################

# TODO thresholding?
def get_metric_dict(predictions: pd.DataFrame, labels: pd.DataFrame, threshold: float = constants.default_threshold) -> Dict:
    # predictions and labels must have the same indexes, although do not need 
    # #to be in same order
    
    
    #clipped_labels = labels.loc[predictions.index, :] # maybe not all the preds could be made?
    missing_preds = labels.loc[labels.index.difference(predictions.index)] # missing predictions
    missing_preds.purchased = 0 # for missing predictions, we just predict 0 # TODO: correct?

    predictions_full = pd.concat([predictions, missing_preds])
    
    labels.rename({'purchased':'actual_purchased'}, inplace=True, axis=1)
    final_results = pd.merge(labels, predictions_full, on=['productId', 'customerId'], how='inner')
    reals = final_results.actual_purchased.values

    preds = final_results.purchased.values
    threshold_preds = convert_pred_prob_to_bool(preds, threshold=threshold) # TODO: task wants prob, but evaluation needs bool, so convert here using 0.5 threshold - could decide this in the model predictions

    precision = get_precision(y_true=reals, y_pred=threshold_preds)
    recall = get_recall(y_true=reals, y_pred=threshold_preds)
    f1_score = get_f1_score(y_true=reals, y_pred=threshold_preds)
    auc = get_auc_roc(y_true=reals, y_score=preds)

    metric_dict = {
        "f1":f1_score,
        "precision":precision,
        "recall":recall,
        "auc_roc":auc,
    }
    return metric_dict


################################################################################


if __name__ == "__main__":
    # TODO argparse file locations.
    mae = get_metric_dict()