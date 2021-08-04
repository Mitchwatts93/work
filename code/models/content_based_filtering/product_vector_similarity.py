from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.knns import KNNBasic

import numpy as np
import pandas as pd

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from models import common_funcs

################################################################################


def get_vector_content_sim_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    product_vectors = encode_products(test_df['productId'])
    

    test_df['purchased'] = predictions # NOTE: same name column as labels
    return test_df

################################################################################

def main():
    model_name = "vector_content_sim"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(model_name=model_name, model_fetching_func=get_vector_content_sim_probs, dataset_being_evaluated=dataset_being_evaluated)
    labels = common_funcs.get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores = common_funcs.get_scores(predictions, labels, model_name=model_name, dataset_being_evaluated=dataset_being_evaluated)
    
    if dataset_being_evaluated == "val":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.VAL_SCORES_DICT)
    elif dataset_being_evaluated == "test":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.TEST_SCORES_DICT)

################################################################################

if __name__ == "__main__":
    main()
