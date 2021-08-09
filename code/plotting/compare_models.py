import os, sys
import pandas as pd

import matplotlib.pyplot as plt

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from models import common_funcs

def main():
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
