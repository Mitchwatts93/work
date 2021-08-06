from tensorflow import keras


class simple_NN(keras.model):

    def __init__(self) -> None:
        super().__init__()

        self.prod_fc1 = keras.layers.Dense(units=10, activation='relu')

        self.cust_fc1 = keras.layers.Dense(units=10, activation='relu')

        self.fc2 = keras.layers.Dense(units=10, activation='relu')
        self.out = keras.layers.Dense(units=1, activation=None)


    
    def call(self, X):
        product_vec = X[0]
        customer_vec = X[1]

        p_hid = self.prod_fc1(product_vec)
        c_hid = self.cust_fc1(customer_vec)

        concatted = tf.concat([p_hid, c_hid])

        hid2 = self.fc2(concatted)
        out = self.out(hid2)

        return out


from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.knns import KNNBasic

import numpy as np
import pandas as pd
from tqdm import tqdm

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from models.content_based_filtering import product_vector_similarity, customer_vector_similarity

################################################################################


def get_content_nn_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    row_lookup, encoded_customers = customer_vector_similarity.encode_customer()
    row_lookup, encoded_customers = product_vector_similarity.encode_products()

    breakpoint()
    test_df['purchased'] = predictions # NOTE: same name column as labels
    return test_df

################################################################################

def main():
    model_name = "content_nn"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(model_name=model_name, model_fetching_func=get_content_nn_probs, dataset_being_evaluated=dataset_being_evaluated)
    labels = common_funcs.get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores = common_funcs.get_scores(predictions, labels, model_name=model_name, dataset_being_evaluated=dataset_being_evaluated)
    
    if dataset_being_evaluated == "val":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.VAL_SCORES_DICT)
    elif dataset_being_evaluated == "test":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.TEST_SCORES_DICT)

################################################################################

if __name__ == "__main__":
    main()
