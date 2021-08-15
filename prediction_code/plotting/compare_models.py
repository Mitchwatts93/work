"""short script to make a plot comparing all models for each metric of interest
"""

import os, sys
import pandas as pd

import matplotlib.pyplot as plt

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)
sys.path.append(PDIR) # rather than force you to add package to path in bash, 
# I've done this for robustness

from misc import constants
from models import common_funcs

################################################################################

def main():
    """load metrics dict, for each metric make barplot of all different models
    """
    model_scores_dict = common_funcs.load_master_scores_dict(constants.VAL_SCORES_DICT)
    model_score_df = pd.DataFrame(model_scores_dict).T

    for metric in model_score_df.columns:
        model_score_df.plot.bar(y=metric)
        plt.tight_layout()
        save_path = os.path.join(constants.PLOTS_PATH, f'{metric}.png')
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    main()
