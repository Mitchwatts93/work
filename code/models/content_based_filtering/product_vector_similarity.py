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

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from models import common_funcs
from processing import data_loading

###############################################################################

def engineer_product_features(product_df: pd.DataFrame) -> pd.DataFrame:
    product_df = product_df[
        ['brand', 'price', 'productType', 'onSale', 'dateOnSite']
    ]
    product_df.fillna(product_df.median(), inplace=True)


    product_df.loc[:, "onSale"] = product_df.onSale.astype(bool)

    product_df["brand"] = product_df["brand"].astype('category')
    product_df["productType"] = product_df["productType"].astype('category')
    product_df["onSale"] = product_df["onSale"].astype('category')

    product_df["dateOnSite"] = pd.to_datetime(product_df.dateOnSite, errors='coerce')
    product_df.dateOnSite.fillna(product_df.dateOnSite.mean(), inplace=True)
    product_df["days_on_site"] = (
            product_df["dateOnSite"] - product_df["dateOnSite"].min()
        ).dt.days

    # could bin prices?

    return product_df


def encode_products() -> Tuple[Dict, np.ndarray]:
    product_df = data_loading.get_products_df() # get product info

    # form lookup dict for getting rows of encoded matrix based on product id
    row_lookup = dict(zip(product_df[constants.product_id_str], range(len(product_df)))) # 
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
    productIds = train_df[train_df[constants.customer_id_str] == customerId][constants.product_id_str].values
    return productIds


def get_vector_content_sim_product_probs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    row_lookup, encoded_products = encode_products()

    grouped_customers = train_df.groupby(constants.customer_id_str)
    customer_product_dict = grouped_customers[constants.product_id_str].apply(np.array).to_dict()

    # TODO vectorise
    # TODO change variable names
    predictions = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        
        this_product_id = row[constants.product_id_str]

        try:
            row_lookup[this_product_id]
        except KeyError:
            predictions.append(0.5)
            continue 

        this_product_vector = encoded_products[row_lookup[this_product_id], :].toarray()

        try:
            customer_product_dict[row[constants.product_id_str]]
        except KeyError:
            predictions.append(0.5)
            continue 

        productIds = customer_product_dict[row[constants.customer_id_str]]
        productIds = productIds[np.where(productIds != this_product_id)[0]]

        if len(productIds) == 0:
            predictions.append(0.5)
            continue # no other customers have bought this product!

        productIds = [id for id in productIds if id in row_lookup]
        
        product_encoder_rows = [row_lookup[id] for id in productIds]
        product_vectors = encoded_products[product_encoder_rows, :].toarray()

        product_similarities = cosine_similarity(product_vectors, this_product_vector)
        mean_sim = np.mean(product_similarities)
        predictions.append(mean_sim)
   
    test_df.loc[:, constants.probabilities_str] = predictions # NOTE: same name column as labels
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
