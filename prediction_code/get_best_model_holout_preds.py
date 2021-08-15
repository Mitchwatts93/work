import os, sys
import gzip, pickle

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))
sys.path.append(PPDIR) # rather than force you to add package to path in bash, 
# I've done this for robustness

from misc import constants

################################################################################

def main():
    holout_set_save_path = os.path.join(constants.PREDICTIONS_PATH, 'nnv1_holdout_set.gzip')
    with gzip.open(holout_set_save_path, 'rb') as f:
        overall_holdout_set_preds = pickle.load(f)

    return overall_holdout_set_preds

if __name__ == "__main__":
    overall_holdout_set_preds = main()

