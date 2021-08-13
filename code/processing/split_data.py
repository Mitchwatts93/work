"""Functions for splitting the data into train, val and test sets."""

import os, sys
from typing import Tuple
from datetime import datetime, timezone
import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from processing import data_loading
from misc import constants
from misc.caching import load_or_make_wrapper

################################################################################
# dates for splitting sets

TRAINING_START_DATE = datetime(day=1, month=1, year=2017, tzinfo=timezone.utc)
VALIDATION_START_DATE = datetime(
    day=22, month=1, year=2017, tzinfo=timezone.utc
)
TEST_START_DATE = datetime(day=27, month=1, year=2017, tzinfo=timezone.utc)

################################################################################
# this is the main splitting, since it determines which labels in the training
# labels will be in each set.

def split_purchases_df() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split the purchases data into train, val, test
    Returns:
        train_purchases, val_purchases, test_purchases: pd dataframes split from
            the raw purchases data according to TRAINING_START_DATE, 
            VALIDATION_START_DATE and TEST_START_DATE.
    """
    purchases_df = data_loading.get_purchases_df()

    train_purchases = purchases_df[purchases_df.date < VALIDATION_START_DATE]
    val_purchases = purchases_df[
        (purchases_df.date >= VALIDATION_START_DATE) & \
            (purchases_df.date < TEST_START_DATE)
    ]
    test_purchases = purchases_df[purchases_df.date >= TEST_START_DATE]

    return train_purchases, val_purchases, test_purchases

################################################################################

def split_labels_training_df(
    train_purchases: pd.DataFrame, 
    val_purchases: pd.DataFrame, 
    test_purchases: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split the labels training data into train, val, test sets according to 
    the split purchases data.

    Args:
        train_purchases: pd dataframe of the training set for purchases data.
        val_purchases: pd dataframe of the validation set for purchases data.
        test_purchases: pd dataframe of the test set for purchases data.
    Returns:
        train_labels_training, val_labels_training, test_labels_training: pd 
            dataframes split from the raw training labels data according to 
            the split purchases data  
    """
    labels_training_df = data_loading.get_labels_training_df()

    # for each of the below, split according to which productIds and customerIds
    # are in the purchases data - for the train, val and test sets accordingly.
    train_labels_training = labels_training_df[
        (
            labels_training_df["customerId"] + 
            labels_training_df["productId"]
        ).isin(
            train_purchases["customerId"] + train_purchases["productId"]
        )
    ]
    val_labels_training = labels_training_df[
        (
            labels_training_df["customerId"] + 
            labels_training_df["productId"]
        ).isin(
            val_purchases["customerId"] + val_purchases["productId"]
        )
    ]
    test_labels_training = labels_training_df[
        (
            labels_training_df["customerId"] + 
            labels_training_df["productId"]
        ).isin(
            test_purchases["customerId"] + test_purchases["productId"]
        )
    ]

    return train_labels_training, val_labels_training, test_labels_training

################################################################################
# this set of functions all make use of the load_or_make_wrapper function, 
# where if the data is already cached it is loaded, else it is computed.

def get_split_purchases_df(
    cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """either load from cache or compute and then return the pd dataframes for
    train, val and test sets for purchases data.
    Args:
        cache: bool of whether to cache the data if it wasn't already cached.
    Returns:
        train_purchases, val_purchases, test_purchases: pd dataframes split from
            the raw purchases data according to TRAINING_START_DATE, 
            VALIDATION_START_DATE and TEST_START_DATE.
    """
    cache_path = constants.SPLIT_PURCHASES_PATH
    train_purchases, val_purchases, test_purchases = load_or_make_wrapper(
        maker_func=split_purchases_df, 
        filepath=cache_path, cache=cache, 
    )
    return train_purchases, val_purchases, test_purchases


def get_split_labels_training_df(
    cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """either load from cache or compute and then return the pd dataframes for
    train, val and test sets for labels training data.
    Args:
        cache: bool of whether to cache the data if it wasn't already cached.
    Returns:
        train_labels_training, val_labels_training, test_labels_training: pd 
            dataframes split from the raw training labels data according to 
            the split purchases data .
    """

    train_purchases, val_purchases, test_purchases = get_split_purchases_df() #
    # this will retrieve this data from cache

    cache_path = constants.SPLIT_LABELS_TRAINING_PATH
    train_labels_training, val_labels_training, test_labels_training = load_or_make_wrapper(
        maker_func=split_labels_training_df, 
        filepath=cache_path, cache=cache, 
        train_purchases=train_purchases,
        val_purchases=val_purchases,
        test_purchases=test_purchases,
    )
    return train_labels_training, val_labels_training, test_labels_training

################################################################################

def main():
    """get these datasets here, but do nothing with them. If they're not 
    already cached then this will just serve to cache them for faster loading 
    next time"""
    train_purchases, val_purchases, test_purchases = get_split_purchases_df()    
    train_labels_training, val_labels_training, test_labels_training = get_split_labels_training_df()


if __name__ == "__main__":
    main()

