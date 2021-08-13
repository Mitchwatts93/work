"""functions for plotting evaluation results from models.
"""
# TODO mkdir to save plots in tidier format

import os, sys
from typing import Tuple
import gzip, pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from processing import split_data
from recmetrics.plots import class_separation_plot, precision_recall_plot

################################################################################

def get_ytrue_ypred(
    pred_filename: os.PathLike
) -> Tuple[np.ndarray, np.ndarray]:
    """load the predictions and corresponding real values, and if predictions 
    are missing then just omit them.
    Args:
        pred_filename: name of prediction file in constants.PREDICTIONS_PATH dir
    Returns:
        y_true: np array of true labels (as bools)
        y_pred: np array of predicted probabilities
    """
    # check is valid file
    if not os.path.isfile(
        os.path.join(constants.PREDICTIONS_PATH, pred_filename)
    ):
        raise FileNotFoundError("this isn't a file")
    if not pred_filename.endswith('.gzip'):
        raise FileNotFoundError("this isn't a gzip file")

    # load predictions
    with gzip.open(
        os.path.join(constants.PREDICTIONS_PATH, pred_filename), 
        'rb'
    ) as f:
        preds = pickle.load(f)

    # load labels
    _train_labels_training, val_labels_training, _test_labels_training = \
        split_data.get_split_labels_training_df()

    # merge the two and align their indices
    suffix = "label"
    val_labels_training.rename(
        {
            constants.probabilities_str:f"{constants.probabilities_str}_{suffix}"
        }, 
        inplace=True, 
        axis=1
    ) # rename the purchased column with _label as suffix so we don't get a 
    # conflict of column names
    joined_df = pd.merge(
        preds, 
        val_labels_training, 
        on=[
            constants.customer_id_str, 
            constants.product_id_str
        ], 
        how='inner'
    ) # merge them so that the rows match correctly. This gets rid of any 
    # missing preds

    # convert to np array
    y_true = joined_df[constants.purchased_label_str].values
    y_pred = joined_df[constants.probabilities_str].values
    return y_true, y_pred

################################################################################

def plot_roc(
    pred_filename : os.PathLike, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """plot the roc curve for these predictions."""

    fpr, tpr, _thresholds = roc_curve(
        y_true=y_true.astype(float), y_score=y_pred
    ) # get the false positive rate (fpr) and true positive rate (tpr) for 
    # roc plot
    
    plt.close() # in case plots already open

    roc_auc = auc(fpr, tpr) # get roc score

    # plot data
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc
    ) # plot roc
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') # plot 
    # diagonal line for random classifier

    # format fig
    plt.title('Receiver operating curve')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate')")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # save plot
    save_path = os.path.join(
        constants.PLOTS_PATH, 
        f"roc_{pred_filename.split('.')[0]}.png"
    )
    plt.savefig(save_path)
    
    plt.close()


def plot_precision_recall_plot(
    pred_filename : os.PathLike, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """plot the precision recall graph and save it in plots dir. uses 
    recmetrics function"""
    save_path = os.path.join(
        constants.PLOTS_PATH, 
        f"prec_rec_{pred_filename.split('.')[0]}.png"
    )
    precision_recall_plot(preds=y_pred, targs=y_true, save_label=save_path)
    plt.close()


def plot_class_separation(
    pred_filename : os.PathLike, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """plot the class separation using remetrics function"""
    pred_df = pd.DataFrame(
        data=np.vstack(y_true, y_pred), 
        columns=['predicted', 'truth']
    ) # recmetrics wants these columns
    #model_name = pred_filename.split('.')[0] # TODO mkdir to save plots
    save_path = os.path.join(
        constants.PLOTS_PATH, 
        f"class_sep_{pred_filename.split('.')[0]}.png"
    )
    class_separation_plot(pred_df=pred_df, save_label=save_path)


################################################################################

def plot_all_preds() -> None:
    """look through all the saved prediction file and make a plot for each"""
    for pred_filename in os.listdir(constants.PREDICTIONS_PATH):
        try:
            y_true, y_pred = get_ytrue_ypred()
        except FileNotFoundError:
            # the file is not a prediction file
            continue

        plot_class_separation(pred_filename, y_true, y_pred)
        plot_precision_recall_plot(pred_filename, y_true, y_pred)
        plot_roc(pred_filename, y_true, y_pred)

################################################################################

if __name__ == "__main__":
    plot_all_preds()
