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

from misc import constants
from models import common_funcs

from processing import data_loading

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
################################################################################

def encode_customer():
    customer_df = data_loading.get_customers_df()

    row_lookup = dict(zip(customer_df.customerId, range(len(customer_df)))) # faster for later - rather than use df so we can keep sparse encoded

    customer_df = customer_df[['isFemale', 'country', 'yearOfBirth', 'isPremier']]

    customer_df.loc[:, "isFemale"] = customer_df.isFemale.astype(bool)
    customer_df.loc[:, "isPremier"] = customer_df.isPremier.astype(bool)

    oldest_person_alive_birthyear = 1903
    customer_df.loc[customer_df.yearOfBirth < oldest_person_alive_birthyear, "yearOfBirth"] = np.nan
    customer_df.yearOfBirth.fillna(customer_df.yearOfBirth.median(), inplace=True)

    customer_df["isFemale"] = customer_df["isFemale"].astype('category')
    customer_df["country"] = customer_df["country"].astype('category')
    customer_df["isPremier"] = customer_df["isPremier"].astype('category')
    customer_df["yearOfBirth"] = customer_df["yearOfBirth"].astype('category')

    enc = OneHotEncoder()
    encoded_customers = enc.fit_transform(customer_df)

    return row_lookup, encoded_customers


def get_customers_who_bought_product(train_df, productId):
    customerIds = train_df[train_df.productId == productId]["customerId"].values
    return customerIds


def get_vector_content_sim_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    row_lookup, encoded_customers = encode_customer()

    grouped_products = train_df.groupby('productId')
    product_customer_dict = grouped_products['customerId'].apply(np.array).to_dict()

    predictions = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        
        this_customer_id = row.customerId

        try:
            row_lookup[this_customer_id]
        except KeyError:
            predictions.append(0.5)
            continue 

        this_customer_vector = encoded_customers[row_lookup[this_customer_id], :].toarray()

        try:
            product_customer_dict[row.productId]
        except KeyError:
            predictions.append(0.5)
            continue 

        customerIds = product_customer_dict[row.productId]
        customerIds = customerIds[np.where(customerIds != this_customer_id)[0]] # TODO add back in

        if len(customerIds) == 0:
            predictions.append(0.5)
            continue # no other customers have bought this product!

        #customerIds =  np.random.choice(customerIds, 5, replace=True) # TODO remove this, just picking random 5 for now
        customerIds = [id for id in customerIds if id in row_lookup]
        
        customer_encoder_rows = [row_lookup[id] for id in customerIds]
        customer_vectors = encoded_customers[customer_encoder_rows, :].toarray()

        # why are all the similarities so high?
        #customer_similarities = np.dot(customer_vectors, this_customer_vector.reshape(-1))
        customer_similarities = cosine_similarity(customer_vectors, this_customer_vector)
        #customer_similarities = (np.dot(customer_vectors, this_customer_vector.reshape(-1)) / np.linalg.norm(this_customer_vector)) / np.linalg.norm(customer_vectors, axis=1)
        mean_sim = np.mean(customer_similarities)
        predictions.append(mean_sim)

    test_df['purchased'] = predictions # NOTE: same name column as labels
    return test_df

################################################################################

def main():
    model_name = "vector_content_sim_customer"
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
