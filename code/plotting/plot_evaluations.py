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
        

    pred_df = pd.concat([preds.purchased.astype(float), val.purchased.astype(float)], axis=1)
    pred_df.columns = ['predicted', 'truth']

    save_path = os.path.join(constants.PLOTS_PATH, f"class_sep_{pred_file.split('.')[0]}.png")
    class_separation_plot(pred_df=pred_df, save_label=save_path)

    save_path = os.path.join(constants.PLOTS_PATH, f"prec_rec_{pred_file.split('.')[0]}.png")
    print(save_path)
    if preds.isnull().any().any() or val.isnull().any().any():
        breakpoint()
    precision_recall_plot(preds=preds.purchased.values, targs=val.purchased.values, save_label=save_path)

    # save_path = os.path.join(constants.PLOTS_PATH, f"roc_{pred_file.split('.')[0]}.png")
    #roc_plot(actual=val.purchased.values, model_probs=[preds.purchased.values,], model_names=[pred_file.split('.')[0],], save_label=save_path)
    # this was very slow so skipped it