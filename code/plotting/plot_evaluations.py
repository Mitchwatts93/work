import os, sys
import gzip, pickle

import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from recmetrics.plots import class_separation_plot, precision_recall_plot #, roc_plot


for pred_file in os.listdir(constants.PREDICTIONS_PATH):
    if not os.path.isfile(os.path.join(constants.PREDICTIONS_PATH, pred_file)):
        continue
    if not pred_file.endswith('.gzip'):
        continue

    with gzip.open(os.path.join(constants.PREDICTIONS_PATH, pred_file), 'rb') as f:
        preds = pickle.load(f)

    with gzip.open(os.path.join(constants.RAW_DATA_DIR, 'split_labels_training.gzip'), 'rb') as f: 
        real=pickle.load(f)
        val = real[1]

    val.rename({'purchased':'purchased_label'}, inplace=True, axis=1) # rename the purchased columns
    train_df = pd.merge(preds, val, on=['customerId', 'productId'], how='inner') # merge them so now it all matches
    pred_df = train_df[["purchased", "purchased_label"]]
    #pred_df = pd.concat([preds.purchased.astype(float), val.purchased.astype(float)], axis=1)
    pred_df.columns = ['predicted', 'truth']

    save_path = os.path.join(constants.PLOTS_PATH, f"class_sep_{pred_file.split('.')[0]}.png")
    class_separation_plot(pred_df=pred_df, save_label=save_path)

    save_path = os.path.join(constants.PLOTS_PATH, f"prec_rec_{pred_file.split('.')[0]}.png")
    precision_recall_plot(preds=pred_df.predicted.values, targs=pred_df.truth.values, save_label=save_path)

    # save_path = os.path.join(constants.PLOTS_PATH, f"roc_{pred_file.split('.')[0]}.png")
    #roc_plot(actual=val.purchased.values, model_probs=[preds.purchased.values,], model_names=[pred_file.split('.')[0],], save_label=save_path)
    # this was very slow so skipped it

    # TODO add ROC plot
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = metrics.roc_curve(y_true=pred_df.truth.values.astype(float), y_score=pred_df.predicted.values)
    breakpoint()
    import matplotlib.pyplot as plt
    plt.close()
    save_path = os.path.join(constants.PLOTS_PATH, f"roc_{pred_file.split('.')[0]}.png")
    lw = 2
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2]
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend()
    plt.title('Receiver operating curve')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate')")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

