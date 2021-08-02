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

from processing.split_data import get_split_labels_training_df
from evaluation.evaluate import get_metric_dict

################################################################################

def get_purchase_probabilities(inputs_df: pd.DataFrame) -> np.ndarray:
    probs = np.random.rand(len(inputs_df))
    inputs_df['purchased'] = probs # NOTE: same name column as labels
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

    cache_path = os.path.join(constants.PREDICTIONS_PATH, f"random_{dataset_being_evaluated}.gzip")
    predictions = load_or_make_wrapper(
        maker_func=get_purchase_probabilities, 
        filepath=cache_path, cache=True, 
        inputs_df=dataset,
    )
    return predictions


################################################################################

def get_scores(predictions, labels):
    cache_path = os.path.join(constants.SCORES_PATH, f"random_{dataset_being_evaluated}.gzip")
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

def add_scores_to_master_dict(scores_dict, model="random_baseline", model_dict_path=constants.VAL_SCORES_DICT):
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