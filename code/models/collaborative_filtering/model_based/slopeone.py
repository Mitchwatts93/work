"""functions for training and predicting using slopeone algorithm"""

from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.slope_one import SlopeOne

import numpy as np
import pandas as pd

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPPDIR = os.path.dirname(os.path.dirname(os.path.dirname(CDIR)))

sys.path.append(PPPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from models import common_funcs

################################################################################

def get_SlopeOne_probs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """fit slopeone model to train_df and then make predictions on test_df"""
    # build surprise datasets
    train_df = train_df.iloc[:int(len(train_df) / 3)] # RAM ISSUES
    train_data = Dataset.load_from_df(
        train_df, 
        reader=Reader(rating_scale=(0,1))
    )
    val_data = Dataset.load_from_df(test_df, reader=Reader(rating_scale=(0,1)))

    # fit model
    algo = SlopeOne()
    algo.fit(train_data.build_full_trainset())

    # make predictions
    probs = algo.test(val_data.build_full_trainset().build_testset())
    df = pd.DataFrame(probs)
    predictions = df.est.values

    # save predictions in df
    test_df.loc[:, constants.probabilities_str] = predictions
    return test_df

################################################################################

def main() -> None:
    model_name = "SlopeOne"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(
        model_name=model_name, 
        model_fetching_func=get_SlopeOne_probs, 
        dataset_being_evaluated=dataset_being_evaluated
    )
    labels = common_funcs.get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores_dict = common_funcs.get_scores(
        predictions=predictions, 
        labels=labels, 
        model_name=model_name, 
        dataset_being_evaluated=dataset_being_evaluated
    )
    
    common_funcs.cache_scores_to_master_dict(
        dataset_being_evaluated=dataset_being_evaluated,
        scores_dict=scores_dict,
        model_name=model_name
    )
################################################################################

if __name__ == "__main__":
    main()
