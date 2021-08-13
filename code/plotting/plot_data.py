"""functions for generating some plots to explore the data
"""
import os, sys
import matplotlib.pyplot as plt
import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from processing import split_data

################################################################################

def get_validation_dataset() -> pd.DataFrame:
    """get the validation set labels training data"""
    real_data = split_data.get_split_labels_training_df()
    validation_data = real_data[1]
    return validation_data

################################################################################

def plot_long_tail_dist(
    validation_data: pd.DataFrame, plot_name: str = 'tail_hist.png'
) -> None:
    """plot the long tail distribution of products.
    """
    plt.close() # in case one is already open
    volume_df = validation_data[constants.product_id_str].\
        value_counts().reset_index() # counts of each productId
    volume_str = "volume"
    volume_df.columns = [constants.product_id_str, volume_str] # select cols
    plt.plot(volume_df[constants.product_id_str], volume_df[volume_str]) # plot
    save_path = os.path.join(constants.PLOTS_PATH, plot_name)
    plt.savefig(save_path)
    plt.close()


def plot_class_balance(
    validation_data: pd.DataFrame, plot_name: str = 'class_balance.png'
) -> None:
    """plot the long tail distribution of products.
    """
    plt.close()
    validation_data[constants.purchased_label_str].value_counts().plot.bar()
    save_path = os.path.join(constants.PLOTS_PATH, plot_name)
    plt.savefig(save_path)

################################################################################

def main():
    """couple of plots for EDA"""
    validation_data = get_validation_dataset() # get validation dataset
            
    plot_long_tail_dist(validation_data) # plot long tail distribution
    plot_class_balance(validation_data) # plot class balance
    

if __name__ == "__main__":
    main()