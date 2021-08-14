"""random baseline model
"""
import os, sys

import numpy as np
import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from models import common_funcs

################################################################################

def get_purchase_probabilities(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """just set all the predicted probabilities as random floats [0-1] of the 
    correct shape
    Args:
        train_df: pd dataframe containing training data. Note this is necessary 
            for this function to be compatible with 
            common_funcs.generate_and_cache_preds
        test_df: pd dataframe containing test data. Note this is necessary 
            for this function to be compatible with 
            common_funcs.generate_and_cache_preds
    Returns:
        test_df: change the column for labels to be predicted values instead
    """
    probs = np.random.rand(len(test_df))
    test_df.loc[:, constants.probabilities_str] = probs
    return test_df

################################################################################

def main():
    """get preds and cache them"""
    model_name = "random_baseline"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(
        model_name=model_name, 
        model_fetching_func=get_purchase_probabilities, 
        dataset_being_evaluated=dataset_being_evaluated
    ) # get the set predictions
    labels = common_funcs.get_labels(
        dataset_to_fetch=dataset_being_evaluated
    ) # get the set labels
    scores_dict = common_funcs.get_scores(
        predictions, 
        labels, 
        model_name=model_name, 
        dataset_being_evaluated=dataset_being_evaluated
    )

    common_funcs.cache_scores_to_master_dict(
        dataset_being_evaluated=dataset_being_evaluated, 
        scores_dict=scores_dict, 
        model_name=model_name
    ) # update the master scores dict
    
################################################################################

if __name__ == "__main__":
    main()
    