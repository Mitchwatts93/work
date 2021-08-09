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

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
################################################################################

def encode_products():
    product_df = data_loading.get_products_df()

    # add missing customer ids
    #full_ids = np.arange(product_df.productId.max())
    #missing_ids = np.setdiff1d(full_ids, product_df.productId)
    #missing_vals = np.full(shape=(len(missing_ids), len(product_df.columns)), fill_value=np.nan) # TODO can't put nans here, so put them after encoding, just conatenate to np array
    #missing_vals[:, 0] = missing_ids
    #missing_ids_df = pd.DataFrame(data=missing_vals, columns=product_df.columns)
    #product_df = pd.concat([product_df, missing_ids_df])
    #product_df.loc[:, "productId"] = product_df.productId.astype(int)

    row_lookup = dict(zip(product_df.productId, range(len(product_df)))) # faster for later - rather than use df so we can keep sparse encoded

    product_df = product_df[['brand', 'price', 'productType', 'onSale', 'dateOnSite']]
    product_df.fillna(product_df.median(), inplace=True)


    product_df.loc[:, "onSale"] = product_df.onSale.astype(bool)

    product_df["brand"] = product_df["brand"].astype('category')
    product_df["productType"] = product_df["productType"].astype('category')
    product_df["onSale"] = product_df["onSale"].astype('category')

    product_df["dateOnSite"] = pd.to_datetime(product_df.dateOnSite, errors='coerce')
    product_df.dateOnSite.fillna(product_df.dateOnSite.mean(), inplace=True)
    product_df["days_on_site"] = (product_df["dateOnSite"] - product_df["dateOnSite"].min()).dt.days

    # could bin prices?

    enc = ColumnTransformer([
        ("categorical", OneHotEncoder(), ['brand', 'productType', 'onSale']),
        ("numerical", StandardScaler(), ['price','days_on_site']), # TODO standard scale them!
    ])
    
    encoded_products = enc.fit_transform(product_df)

    return row_lookup, encoded_products


def get_products_customer_bought(train_df, customerId):
    productIds = train_df[train_df.customerId == customerId]["productId"].values
    return productIds


def get_vector_content_sim_product_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    row_lookup, encoded_products = encode_products()

    grouped_customers = train_df.groupby('customerId')
    customer_product_dict = grouped_customers['productId'].apply(np.array).to_dict()

    predictions = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        
        this_product_id = row.productId

        try:
            row_lookup[this_product_id]
        except KeyError:
            predictions.append(0.5)
            continue 

        this_product_vector = encoded_products[row_lookup[this_product_id], :].toarray()

        try:
            customer_product_dict[row.productId]
        except KeyError:
            predictions.append(0.5)
            continue 

        productIds = customer_product_dict[row.customerId]
        productIds = productIds[np.where(productIds != this_product_id)[0]] # TODO add back in

        if len(productIds) == 0:
            predictions.append(0.5)
            continue # no other customers have bought this product!

        #customerIds =  np.random.choice(customerIds, 5, replace=True) # TODO remove this, just picking random 5 for now
        productIds = [id for id in productIds if id in row_lookup]
        
        product_encoder_rows = [row_lookup[id] for id in productIds]
        product_vectors = encoded_products[product_encoder_rows, :].toarray()

        # why are all the similarities so high?
        #customer_similarities = np.dot(customer_vectors, this_customer_vector.reshape(-1))
        product_similarities = cosine_similarity(product_vectors, this_product_vector)
        #customer_similarities = (np.dot(customer_vectors, this_customer_vector.reshape(-1)) / np.linalg.norm(this_customer_vector)) / np.linalg.norm(customer_vectors, axis=1)
        mean_sim = np.mean(product_similarities)
        predictions.append(mean_sim)
   
    test_df['purchased'] = predictions # NOTE: same name column as labels
    return test_df

################################################################################

def main():
    model_name = "vector_content_sim_product"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(model_name=model_name, model_fetching_func=get_vector_content_sim_product_probs, dataset_being_evaluated=dataset_being_evaluated)
    labels = common_funcs.get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores = common_funcs.get_scores(predictions, labels, model_name=model_name, dataset_being_evaluated=dataset_being_evaluated)
    
    if dataset_being_evaluated == "val":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.VAL_SCORES_DICT)
    elif dataset_being_evaluated == "test":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.TEST_SCORES_DICT)

################################################################################

if __name__ == "__main__":
    main()
