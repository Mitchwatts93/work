from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.matrix_factorization import NMF

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

def get_NMF_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    train_df = train_df.iloc[:int(len(train_df) / 3)] # RAM ISSUES
    # discard items and products which have all zeros
    # as surprise package struggles with these
    product_counts = train_df.groupby('productId')['purchased'].sum()
    valid_products = product_counts[product_counts > 0].index
    customer_counts = train_df.groupby('customerId')['purchased'].sum()
    valid_customers = customer_counts[customer_counts > 0].index

    train_df = train_df[(train_df.customerId.isin(valid_customers)) & (train_df.productId.isin(valid_products))]
    test_df = test_df[(test_df.customerId.isin(valid_customers)) & (test_df.productId.isin(valid_products))]

    # build surprise datasets
    train_data = Dataset.load_from_df(train_df, reader=Reader(rating_scale=(0,1)))
    val_data = Dataset.load_from_df(test_df, reader=Reader(rating_scale=(0,1)))

    # fit model
    algo = NMF()
    algo.fit(train_data.build_full_trainset())

    # make predictions
    probs = algo.test(val_data.build_full_trainset().build_testset()) # TODO this doesn't work?

    df = pd.DataFrame(probs)
    predictions = df.est.values

    # save predictions in df
    test_df['purchased'] = predictions # NOTE: same name column as labels
    return test_df

################################################################################

def main():
    model_name = "NMF"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(model_name=model_name, model_fetching_func=get_NMF_probs, dataset_being_evaluated=dataset_being_evaluated)
    labels = common_funcs.get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores = common_funcs.get_scores(predictions, labels, model_name=model_name, dataset_being_evaluated=dataset_being_evaluated)
    
    if dataset_being_evaluated == "val":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.VAL_SCORES_DICT)
    elif dataset_being_evaluated == "test":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.TEST_SCORES_DICT)

################################################################################

if __name__ == "__main__":
    main()
