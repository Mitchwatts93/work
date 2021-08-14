"""functions for training and predicting using NMF algorithm"""
from typing import Tuple

from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.matrix_factorization import NMF

import pandas as pd

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPPDIR = os.path.dirname(os.path.dirname(os.path.dirname(CDIR)))

sys.path.append(PPPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from models import common_funcs

################################################################################

def trim_train_test_dfs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """trim the datasets due to ram issues, and due to missing product or 
    customer indices from train or test df"""

    train_df = train_df.iloc[:int(len(train_df) / 3)] # RAM ISSUES

    # discard items and products which have all zeros
    # as surprise package struggles with these
    product_counts = train_df.groupby(
            constants.product_id_str
        )[constants.purchased_label_str].sum()
    valid_products = product_counts[product_counts > 0].index
    customer_counts = train_df.groupby(
            constants.customer_id_str
        )[constants.purchased_label_str].sum()
    valid_customers = customer_counts[customer_counts > 0].index

    # trim the datasets
    train_df = train_df[
        (train_df[constants.customer_id_str].isin(valid_customers)) & \
            (train_df[constants.product_id_str].isin(valid_products))
    ] # trim the df down, because if not in the set, it fails
    test_df = test_df[
        (test_df[constants.customer_id_str].isin(valid_customers)) &\
             (test_df[constants.product_id_str].isin(valid_products))
    ] # trim the df down, because if not in the set, it fails

    return train_df, test_df

################################################################################

def get_NMF_probs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """fit nmf model using train_df and then make predictions on test_df"""

    # trim the dfs because of product and customer id issues with algorithm
    train_df, test_df = trim_train_test_dfs(train_df=train_df, test_df=test_df)

    # build surprise datasets
    train_data = Dataset.load_from_df(
        train_df, 
        reader=Reader(rating_scale=(0,1))
    )
    val_data = Dataset.load_from_df(
        test_df, 
        reader=Reader(rating_scale=(0,1))
    )

    # fit model
    algo = NMF()
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
    model_name = "NMF"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(
        model_name=model_name, 
        model_fetching_func=get_NMF_probs, 
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
