import os, sys
import gzip, pickle

import pandas as pd

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

def fill_missing_preds_as_nan(overall_holdout_set_preds):
    # NOTE: my predictions are missing some of the holdout set.
    # to get them do this but fill with nan where missing
    holdout = pd.read_csv("../data/raw_data/labels_predict.txt")
    merged = pd.merge(holdout, overall_holdout_set_preds, on=["customerId", "productId"], how="outer")
    final = merged[["customerId", "productId", "purchase_probability_y"]]
    final.rename({"purchase_probability_y":"purchase_probability"}, axis=1, inplace=True)
    return final


if __name__ == "__main__":
    overall_holdout_set_preds = main()
    missing_preds_as_nan = fill_missing_preds_as_nan(overall_holdout_set_preds)
    breakpoint()
    missing_preds_as_nan.to_csv("../data/predictions/holdout_predictions.txt", index=False) # save
    
    