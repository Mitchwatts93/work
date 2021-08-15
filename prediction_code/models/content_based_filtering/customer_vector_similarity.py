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
sys.path.append(PPDIR) # rather than force you to add package to path in bash, 
# I've done this for robustness

from misc import constants
from models import common_funcs
from processing import data_loading

################################################################################

def engineer_customer_df_features(customer_df: pd.DataFrame) -> pd.DataFrame:
    """feature engineering of cutomer df.
    # TODO feature engineering:
    # could encode country based on continent or similar? or at least additional
    # feature?
    """
    # select cols
    customer_df = customer_df[
        ['isFemale', 'country', 'yearOfBirth', 'isPremier']
    ]
    # fill nans
    customer_df.fillna(customer_df.median(), inplace=True)

    # set bool types
    customer_df.loc[:, "isFemale"] = customer_df.isFemale.astype(bool)
    customer_df.loc[:, "isPremier"] = customer_df.isPremier.astype(bool)

    # remove too old birth years
    oldest_person_alive_birthyear = 1903
    customer_df.loc[
            customer_df.yearOfBirth < oldest_person_alive_birthyear,
            "yearOfBirth"
        ] = np.nan
    # fill nans birthyears
    customer_df.yearOfBirth.fillna(
        customer_df.yearOfBirth.median(), inplace=True
    )

    # set types
    customer_df["isFemale"] = customer_df["isFemale"].astype('category')
    customer_df["country"] = customer_df["country"].astype('category')
    customer_df["isPremier"] = customer_df["isPremier"].astype('category')
    customer_df["yearOfBirth"] = customer_df["yearOfBirth"].astype('category')

    return customer_df


def encode_customer() -> Tuple[Dict, np.ndarray]:
    """encode the customer data as vectors, a row for each customer"""
    
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
    """get the customers who previously have bought product == productId"""
    customerIds = train_df[
            train_df[constants.product_id_str] == productId
        ][constants.customer_id_str].values
    return customerIds


def get_vector_content_sim_probs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """get the similarity scores based on content vectors between customers"""
    # get row lookup dict and encoded matrix of customers
    row_lookup, encoded_customers = encode_customer()

    # get a dictionary mapping products to any customers that have previously 
    # bought that product
    grouped_products = train_df.groupby(constants.product_id_str)
    product_customer_mapping = grouped_products[
        constants.customer_id_str].apply(np.array).to_dict()

    # TODO vectorise
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # get customer id for this customer product pair
        this_customer_id = row[constants.customer_id_str]

        # if this customer id isn't in our encoded customers, then predict 
        # default
        try:
            row_lookup[this_customer_id]
        except KeyError:
            predictions.append(constants.default_value_missing)
            continue 

        # get the vector for this customer
        this_customer_vector = encoded_customers[
            row_lookup[this_customer_id], :].toarray()

        # check if the product is in our product-customer dictionary
        # if not, then just predict prob as the default
        try:
            product_customer_mapping[row[constants.product_id_str]]
        except KeyError:
            predictions.append(constants.default_value_missing)
            continue 

        # get the ids of customers that have bought this product before
        customerIds_who_bought_this_product = product_customer_mapping[
                row[constants.product_id_str]
            ]
        customerIds_who_bought_this_product = \
            customerIds_who_bought_this_product[
                np.where(
                    customerIds_who_bought_this_product != this_customer_id
                )[0]
            ] # filter out this customers id though

        # if there are not customers that have bought it, then just predict the 
        # default
        if len(customerIds_who_bought_this_product) == 0:
            predictions.append(constants.default_value_missing)
            continue # no other customers have bought this product!
        
        # filter out customers that aren't encoded
        customerIds_who_bought_this_product = [
            id for id in customerIds_who_bought_this_product if id in row_lookup
        ]
        
        # get the vectors for those customers
        customer_encoder_rows = [row_lookup[id] for id in customerIds_who_bought_this_product]
        customer_vectors = encoded_customers[customer_encoder_rows, :].toarray()

        # get the similarity between each customer and this customer
        customer_similarities = cosine_similarity(
            customer_vectors, this_customer_vector
        )
        mean_sim = np.mean(customer_similarities) # average those similarities
        predictions.append(mean_sim) # that is our prediction

    # set column as predictions
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
