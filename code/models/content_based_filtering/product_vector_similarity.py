from typing import Dict, Tuple
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

###############################################################################

def engineer_product_features(product_df: pd.DataFrame) -> pd.DataFrame:
    """feature engineering of product df.
    TODO     
    # could bin prices?
    """
    # select cols
    product_df = product_df[
        ['brand', 'price', 'productType', 'onSale', 'dateOnSite']
    ]
    # fill nans
    product_df.fillna(product_df.median(), inplace=True)

    # set bool types
    product_df.loc[:, "onSale"] = product_df.onSale.astype(bool)

    # set types
    product_df["brand"] = product_df["brand"].astype('category')
    product_df["productType"] = product_df["productType"].astype('category')
    product_df["onSale"] = product_df["onSale"].astype('category')

    # change to pd datetime
    product_df["dateOnSite"] = pd.to_datetime(
        product_df.dateOnSite, errors='coerce'
    )
    # fill nan dates
    product_df.dateOnSite.fillna(product_df.dateOnSite.mean(), inplace=True)
    # instead, get the number of days after the earliest date - to change to a 
    # numerical feature that is useful
    product_df["days_on_site"] = (
            product_df["dateOnSite"] - product_df["dateOnSite"].min()
        ).dt.days


    return product_df


def encode_products() -> Tuple[Dict, np.ndarray]:
    """encode the product data as vectors, a row for each product"""

    product_df = data_loading.get_products_df() # get product info

    # form lookup dict for getting rows of encoded matrix based on product id
    row_lookup = dict(
        zip(
            product_df[constants.product_id_str], 
            range(len(product_df))
        )
    ) # 
    # faster for later - rather than use df so we can keep sparse encoded

    # feature engineering
    product_df = engineer_product_features(product_df=product_df)

    # encode the features, some as categorical, some as numerical
    enc = ColumnTransformer([
        ("categorical", OneHotEncoder(), ['brand', 'productType', 'onSale']),
        ("numerical", StandardScaler(), ['price','days_on_site']),
    ])
    encoded_products = enc.fit_transform(product_df)

    return row_lookup, encoded_products


def get_products_customer_bought(
    train_df: pd.DataFrame, customerId: int
) -> np.ndarray:
    """get the product previously bought by customer == customerId"""
    productIds = train_df[
            train_df[constants.customer_id_str] == customerId
        ][constants.product_id_str].values
    return productIds


def get_vector_content_sim_product_probs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """get the similarity scores based on content vectors between products"""
    
    # get row lookup dict and encoded matrix of products
    row_lookup, encoded_products = encode_products()

    # get a dictionary mapping products to any products that have previously 
    # been bought that customer
    grouped_customers = train_df.groupby(constants.customer_id_str)
    customer_product_mapping = grouped_customers[
        constants.product_id_str].apply(np.array).to_dict()

    # TODO vectorise
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # get product id for this customer product pair
        this_product_id = row[constants.product_id_str]

        # if this product id isn't in our encoded products, then predict 
        # default
        try:
            row_lookup[this_product_id]
        except KeyError:
            predictions.append(constants.default_value_missing)
            continue 

        # get the vector for this product
        this_product_vector = encoded_products[
            row_lookup[this_product_id], :].toarray()

        # check if the customer is in our customer-product dictionary
        # if not, then just predict prob as the default
        try:
            customer_product_mapping[row[constants.product_id_str]]
        except KeyError:
            predictions.append(constants.default_value_missing)
            continue 

        # get the ids of products that have been bought by this customer before
        productIds_bought_by_this_customer = customer_product_mapping[
                row[constants.customer_id_str]
            ]
        productIds_bought_by_this_customer = productIds_bought_by_this_customer[
            np.where(productIds_bought_by_this_customer != this_product_id)[0]
        ] # filter out this product id though

        # if there are not product that have been bought, then just predict the 
        # default
        if len(productIds_bought_by_this_customer) == 0:
            predictions.append(constants.default_value_missing)
            continue # no other customers have bought this product!

        # filter out products that aren't encoded
        productIds_bought_by_this_customer = [
            id for id in productIds_bought_by_this_customer if id in row_lookup
        ]
        
        # get the vectors for those products
        product_encoder_rows = [
            row_lookup[id] for id in productIds_bought_by_this_customer
        ]
        product_vectors = encoded_products[product_encoder_rows, :].toarray()

        # get the similarity between each product and this product
        product_similarities = cosine_similarity(
            product_vectors, this_product_vector
        )
        mean_sim = np.mean(product_similarities) # average those similarities
        predictions.append(mean_sim) # that is our prediction
   
    # set column as predictions
    test_df.loc[:, constants.probabilities_str] = predictions
    return test_df

################################################################################

def main():
    model_name = "vector_content_sim_product"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(
        model_name=model_name, 
        model_fetching_func=get_vector_content_sim_product_probs, 
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
