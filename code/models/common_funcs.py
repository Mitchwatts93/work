from typing import Callable, Dict
import json
import os, sys
import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc.caching import load_or_make_wrapper
from misc import constants

from processing.split_data import get_split_labels_training_df
from evaluation.evaluate import get_metric_dict

################################################################################

def get_labels(dataset_to_fetch: str = "val") -> Dict:
    """get the labels for the specified dataset"""
    train_df, val_df, test_df = get_split_labels_training_df()
    # store in dict
    datasets = {
        "train":train_df,
        "val":val_df,
        "test":test_df,
    }
    # get the dataset we want as specified by dataset_to_fetch
    dataset = datasets[dataset_to_fetch]
    return dataset


def generate_and_cache_preds(
    model_name: str, 
    model_fetching_func: Callable, 
    dataset_being_evaluated: str = "val", 
    cache: bool = True,
    **additional_kwargs_for_model
) -> pd.DataFrame:
    """generate the predictions using the function specified, and cache it if 
    specified.
    Args:
        model_name: a string for what the model will be stored as in cache
        model_fetching_func: function which has signature train_df, test_df, and
            any additional kwargs as specified by additional_kwargs_for_model
        dataset_being_evaluated: string indicating which dataset to evaluate,  
            either train, val or test
        cache: bool to indicate if the returned values from the 
            model_fetching_func should be saved to cache
        **additional_kwargs_for_model: additional keyword arguments to pass to 
            model_fetching_func
    Returns:
        predictions: a df which was returned from the cache or the 
            model_fetching_func
    """

    # get the labels for train and test set (note test set can be val set also)
    train_dataset = get_labels(dataset_to_fetch="train")
    test_dataset = get_labels(dataset_to_fetch=dataset_being_evaluated)

    # path to cache results
    cache_path = os.path.join(
        constants.PREDICTIONS_PATH, 
        f"{model_name}_{dataset_being_evaluated}.gzip"
    )
    # get the predictions, either from cache or by computing them
    predictions = load_or_make_wrapper(
        maker_func=model_fetching_func, 
        filepath=cache_path, 
        cache=cache, 
        train_df=train_dataset,
        test_df=test_dataset,
        **additional_kwargs_for_model
    )
    return predictions

################################################################################

def get_scores(
    predictions: pd.DataFrame, 
    labels: pd.DataFrame, 
    model_name: str, 
    dataset_being_evaluated: str = "val",
    cache: bool = True,
) -> Dict:
    """gets score dict either from cache or by computing it, and optionally caches it.
    Args:
        predictions: pd dataframe with predicted probabilities under column 
            name constants.probabilities_str
        labels: pd dataframe with purchase labels under column name 
            constants.purchased_label_str
        model_name: string for model name to save/load scores from cache
        dataset_being_evaluated: train, val or test string, to indicate which 
            dataset is being evaluated
        cache: bool to indicate whether to cache values if they are not 
            currently cached
    Returns:
        scores_dict: dictionary containing metric scores for this model based 
            on predictions and labels.
    """
    cache_path = os.path.join(
        constants.SCORES_PATH, 
        f"{model_name}_{dataset_being_evaluated}.gzip"
    )
    scores_dict = load_or_make_wrapper(
        maker_func=get_metric_dict, 
        filepath=cache_path, 
        cache=cache, 
        predictions=predictions,
        labels=labels,
    )
    return scores_dict

################################################################################

def load_master_scores_dict(model_dict_path: os.PathLike) -> Dict:
    """load the dict containing all scores for each model."""
    if not os.path.isfile(model_dict_path):
        return {}
    # Read data from file:
    master_dict = json.load(open(model_dict_path))
    return master_dict


def save_master_scores_dict(dict: Dict, model_dict_path: os.PathLike) -> None:
    """save the dict containing scores for all models"""
    # Serialize data into file:
    json.dump(dict, open(model_dict_path, 'w'), indent=4)

################################################################################

def add_scores_to_master_dict(
    scores_dict: Dict, 
    model_name: str, 
    model_dict_path: os.PathLike = constants.VAL_SCORES_DICT
) -> None:
    """add this score dict to the existing master score dict. and recache"""
    master_dict = load_master_scores_dict(model_dict_path)
    master_dict[model_name] = scores_dict
    save_master_scores_dict(master_dict, model_dict_path)

################################################################################

def cache_scores_to_master_dict(
    dataset_being_evaluated: str, 
    scores_dict: Dict, 
    model_name: str
) -> None:
    """cache the model to the master scores dict depending on whether it is val 
    or test set."""
    if dataset_being_evaluated == "val":
        add_scores_to_master_dict(
            scores_dict, 
            model_name=model_name, 
            model_dict_path=constants.VAL_SCORES_DICT
        )
    elif dataset_being_evaluated == "test":
        add_scores_to_master_dict(
            scores_dict, 
            model_name=model_name, 
            model_dict_path=constants.TEST_SCORES_DICT
        )

################################################################################
