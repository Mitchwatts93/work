import os, sys
import gzip, pickle

import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants


with gzip.open(os.path.join(constants.RAW_DATA_DIR, 'split_labels_training.gzip'), 'rb') as f: 
    real=pickle.load(f)
    val = real[1]
        
# general dataset plots
import matplotlib.pyplot as plt

# plot long tail distribution
plt.close()
item_id_column = 'productId'
volume_df = val[item_id_column].value_counts().reset_index()
volume_df.columns = [item_id_column, "volume"]
plt.plot(volume_df.productId, volume_df.volume)
save_path = os.path.join(constants.PLOTS_PATH, 'tail_hist.png')
plt.savefig(save_path)

# plot class balance
plt.close()
val.purchased.value_counts().plot.bar()
save_path = os.path.join(constants.PLOTS_PATH, 'class_balance.png')
plt.savefig(save_path)
