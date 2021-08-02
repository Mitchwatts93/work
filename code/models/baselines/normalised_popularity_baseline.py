import json

import numpy as np
import pandas as pd

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc.caching import load_or_make_wrapper
from misc import constants

from processing.split_data import get_split_labels_training_df, get_split_purchases_df
from evaluation.evaluate import get_metric_dict


################################################################################

def normalised_popularity(purchases_df):
    number_purchases = purchases_df["productId"].value_counts()
    most_popular_product = number_purchases.max()

    purchases_df["normalised_popularity"] = number_purchases[purchases_df["productId"].values].values / most_popular_product

    normalised_popularities = purchases_df["normalised_popularity"]
    normalised_popularities.index = purchases_df["productId"]

    normalised_popularities.drop_duplicates(inplace=True)

    return normalised_popularities


def get_normalised_popularities(dataset_being_evaluated="val"):
    train_purchases_df, val_purchases_df, test_purchases_df = get_split_purchases_df()

    if dataset_being_evaluated == "train":
        purchases_df = pd.concat([val_purchases_df, test_purchases_df])
    elif dataset_being_evaluated == "val":
        purchases_df = pd.concat([train_purchases_df, test_purchases_df])
    else:
        purchases_df = pd.concat([train_purchases_df, val_purchases_df])

    cache_path = os.path.join(constants.RAW_DATA_DIR, f"normalised_popularity_{dataset_being_evaluated}.gzip")
    normalised_popularities = load_or_make_wrapper(
        maker_func=normalised_popularity, 
        filepath=cache_path, cache=True, 
        purchases_df=purchases_df,
    )
    return normalised_popularities


################################################################################

def fill_missing_popularities(product_ids, normalised_popularities, fill_method="mean", fill_value=None):
    missing_products = product_ids[~product_ids.isin(normalised_popularities.index)].unique()
    if fill_method is None and fill_value is None:
        raise constants.InputError("fill_method and fill_value cannot both be None")
    elif fill_method is not None:
        fill_value = pd.eval(f"normalised_popularities.{fill_method}()")
    
    missing_products_series = pd.Series(index=missing_products, data=fill_value)

    filled_popularities = pd.concat([normalised_popularities, missing_products_series])

    return filled_popularities


################################################################################

def get_purchase_probabilities(inputs_df: pd.DataFrame, dataset_being_evaluated: str = "val") -> np.ndarray:
    product_ids = inputs_df["productId"]
    normalised_popularities = get_normalised_popularities(dataset_being_evaluated=dataset_being_evaluated)
    filled_normalised_popularities = fill_missing_popularities(
        product_ids, normalised_popularities, fill_method="mean",
    ) # there might be products in the test set not seen in the train set - so give these a default value
    # NOTE: most products haven't been seen before! - so probably using "all" datasets
    # NOTE: also this is for views also, so there are still many products never seen before

    product_popularity = filled_normalised_popularities[product_ids]

    inputs_df['purchased'] = product_popularity.values # NOTE: same name column as labels

    return inputs_df

################################################################################

def get_labels(dataset_to_fetch="val"):
    train_df, val_df, test_df = get_split_labels_training_df()
    datasets = {
        "train":train_df,
        "val":val_df,
        "test":test_df,
    }
    dataset = datasets[dataset_to_fetch]
    return dataset


def generate_and_cache_preds(dataset_being_evaluated="val"):
    dataset = get_labels(dataset_to_fetch=dataset_being_evaluated)

    cache_path = os.path.join(constants.PREDICTIONS_PATH, f"normalised_popularity_{dataset_being_evaluated}.gzip")
    predictions = load_or_make_wrapper(
        maker_func=get_purchase_probabilities, 
        filepath=cache_path, cache=True, 
        inputs_df=dataset,
        dataset_being_evaluated=dataset_being_evaluated,
    )
    return predictions

################################################################################
# TODO move common fucntions to a master file

def get_scores(predictions, labels):
    cache_path = os.path.join(constants.SCORES_PATH, f"normalised_popularity_{dataset_being_evaluated}.gzip")
    scores_dict = load_or_make_wrapper(
        maker_func=get_metric_dict, 
        filepath=cache_path, cache=True, 
        predictions=predictions,
        labels=labels,
    )
    return scores_dict


################################################################################

def load_master_scores_dict(model_dict_path):
    if not os.path.isfile(model_dict_path):
        return {}
    # Read data from file:
    master_dict = json.load(open(model_dict_path))
    return master_dict


def save_master_scores_dict(dict, model_dict_path):
    # Serialize data into file:
    json.dump(dict, open(model_dict_path, 'w'), indent=4)


################################################################################

def add_scores_to_master_dict(scores_dict, model="normalised_popularity_baseline", model_dict_path=constants.VAL_SCORES_DICT):
    master_dict = load_master_scores_dict(model_dict_path)
    master_dict[model] = scores_dict
    save_master_scores_dict(master_dict, model_dict_path)


################################################################################


if __name__ == "__main__":
    dataset_being_evaluated = "val"
    predictions = generate_and_cache_preds(dataset_being_evaluated=dataset_being_evaluated)
    labels = get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores = get_scores(predictions, labels)
    if dataset_being_evaluated == "val":
        add_scores_to_master_dict(scores, model_dict_path=constants.VAL_SCORES_DICT)
    elif dataset_being_evaluated == "test":
        add_scores_to_master_dict(scores, model_dict_path=constants.TEST_SCORES_DICT)