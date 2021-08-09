import os
import gzip, pickle

import pandas as pd
from recmetrics.plots import long_tail_plot, class_separation_plot, roc_plot, precision_recall_plot

CDIR = os.path.dirname(os.path.abspath(__file__))

for pred_file in os.listdir(CDIR):
    if not os.path.isfile(os.path.join(CDIR, pred_file)):
        continue
    if not pred_file.endswith('.gzip'):
        continue

    with gzip.open(os.path.join(CDIR, pred_file), 'rb') as f:
        preds = pickle.load(f)

    with gzip.open(os.path.join(CDIR, '../raw_data/split_labels_training.gzip'), 'rb') as f: 
        real=pickle.load(f)
        val = real[1]

    pred_df = pd.concat([preds.purchased.astype(float), val.purchased.astype(float)], axis=1)
    pred_df.columns = ['predicted', 'truth']
    class_separation_plot(pred_df=pred_df, save_label=f"class_sep_{pred_file.split('.')[0]}.png")

    precision_recall_plot(preds=preds.purchased.values, targs=val.purchased.values, save_label=f"prec_rec_{pred_file.split('.')[0]}.png")

    #roc_plot(actual=val.purchased.values, model_probs=[preds.purchased.values,], model_names=[pred_file.split('.')[0],], save_label=f"roc_{pred_file.split('.')[0]}.png")
    # this was very slow so skipped it

# general dataset plots
import matplotlib.pyplot as plt

# plot long tail distribution
plt.close()
item_id_column = 'productId'
volume_df = val[item_id_column].value_counts().reset_index()
volume_df.columns = [item_id_column, "volume"]
plt.plot(volume_df.productId, volume_df.volume)
plt.savefig('tail_hist.png')
# plot class balance
plt.close()
val.purchased.value_counts().plot.bar()
plt.savefig('class_balance.png')