"""functions to get content-based filtering predicitons using customers."""
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))
sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from models import common_funcs
from processing import data_loading

################################################################################

def engineer_customer_df_features(customer_df: pd.DataFrame) -> pd.DataFrame:
    customer_df = customer_df[
        ['isFemale', 'country', 'yearOfBirth', 'isPremier']
    ]
    customer_df.fillna(customer_df.median(), inplace=True)

    # feature engineering
    # TODO feature engineering:
    # could encode country based on continent or similar? or at least additional
    # feature?

    customer_df.loc[:, "isFemale"] = customer_df.isFemale.astype(bool)
    customer_df.loc[:, "isPremier"] = customer_df.isPremier.astype(bool)

    oldest_person_alive_birthyear = 1903
    customer_df.loc[
            customer_df.yearOfBirth < oldest_person_alive_birthyear,
            "yearOfBirth"
        ] = np.nan
    customer_df.yearOfBirth.fillna(
        customer_df.yearOfBirth.median(), inplace=True
    )

    customer_df["isFemale"] = customer_df["isFemale"].astype('category')
    customer_df["country"] = customer_df["country"].astype('category')
    customer_df["isPremier"] = customer_df["isPremier"].astype('category')
    customer_df["yearOfBirth"] = customer_df["yearOfBirth"].astype('category')

    return customer_df


def encode_customer() -> Tuple[Dict, np.ndarray]:
    customer_df = data_loading.get_customers_df() # get the customers 
    # information

    # form lookup dict for getting rows of encoded matrix based on customer id
    row_lookup = dict(
        zip(
            customer_df[constants.customer_id_str], range(len(customer_df))
        )
    ) # 
    # faster for later - rather than use df so we can keep sparse encoded

    # feature engineering
    customer_df = engineer_customer_df_features(customer_df)    

    #encode the features, some as categorical, some as numerical
    enc = ColumnTransformer([
        ("categorical", OneHotEncoder(), ['isFemale', 'country', 'isPremier']),
        ("numerical", StandardScaler(), ['yearOfBirth']),
    ])
    encoded_customers = enc.fit_transform(customer_df)

    return row_lookup, encoded_customers


def get_customers_who_bought_product(
    train_df: pd.DataFrame, productId: int
) -> np.ndarray:
    customerIds = train_df[
            train_df[constants.product_id_str] == productId
        ][constants.customer_id_str].values
    return customerIds


def get_vector_content_sim_probs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:

    # get row lookup dict and encoded matrix of customers
    row_lookup, encoded_customers = encode_customer()


    grouped_products = train_df.groupby(constants.product_id_str)
    product_customer_dict = grouped_products[
        constants.customer_id_str].apply(np.array).to_dict()

    # TODO vectorise
    # TODO change variable names
    predictions = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        
        this_customer_id = row[constants.customer_id_str]

        try:
            row_lookup[this_customer_id]
        except KeyError:
            predictions.append(0.5)
            continue 

        this_customer_vector = encoded_customers[
            row_lookup[this_customer_id], :].toarray()

        try:
            product_customer_dict[row[constants.product_id_str]]
        except KeyError:
            predictions.append(0.5)
            continue 

        customerIds = product_customer_dict[row[constants.product_id_str]]
        customerIds = customerIds[np.where(customerIds != this_customer_id)[0]] 

        if len(customerIds) == 0:
            predictions.append(0.5)
            continue # no other customers have bought this product!

        customerIds = [id for id in customerIds if id in row_lookup]
        
        customer_encoder_rows = [row_lookup[id] for id in customerIds]
        customer_vectors = encoded_customers[customer_encoder_rows, :].toarray()

        customer_similarities = cosine_similarity(
            customer_vectors, this_customer_vector
        )
        mean_sim = np.mean(customer_similarities)
        predictions.append(mean_sim)


    test_df.loc[:, constants.probabilities_str] = predictions 
    return test_df

################################################################################

def main() -> None:
    model_name = "vector_content_sim_customer"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(
        model_name=model_name, 
        model_fetching_func=get_vector_content_sim_probs, 
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
