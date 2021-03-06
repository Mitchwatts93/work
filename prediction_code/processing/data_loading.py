"""functions to load raw data
"""
import os, sys
import pandas as pd


CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)
sys.path.append(PDIR) # rather than force you to add package to path in bash, 
# I've done this for robustness

from misc import constants
from processing import processors


################################################################################

def get_data_df(filepath: os.PathLike) -> pd.DataFrame:
    """load the data as a pd dataframe
    Args:
        filepath: filepath to the file to load (must be a csv format file) 
    Returns:
        df: the data as a dataframe
    """
    df = pd.read_csv(filepath)
    return df

################################################################################

def get_views_df() -> pd.DataFrame:
    """returns the views data as a pd dataframe"""
    df = get_data_df(constants.VIEWS_PATH)
    df['date'] = processors.convert_to_datetime(df['date'])
    return df


def get_purchases_df() -> pd.DataFrame:
    """returns the purchases data as a pd dataframe"""
    df = get_data_df(constants.PURCHASES_PATH)
    df['date'] = processors.convert_to_datetime(df['date']) # timezone is blank 
    # because I assume utc - I could check customerId and get a country for 
    # each time, but I'm assuming its all utc
    return df


def get_products_df() -> pd.DataFrame:
    """returns the products data as a pd dataframe"""
    df = get_data_df(constants.PRODUCTS_PATH)
    return df


def get_customers_df() -> pd.DataFrame:
    """returns the customers data as a pd dataframe"""
    df = get_data_df(constants.CUSTOMERS_PATH)
    return df


def get_labels_training_df() -> pd.DataFrame:
    """returns the labels training data as a pd dataframe"""
    df = get_data_df(constants.LABELS_TRAINING_PATH)
    return df


def get_labels_predict_df() -> pd.DataFrame:
    """returns the labels holdout data as a pd dataframe"""
    df = get_data_df(constants.LABELS_PREDICT_PATH)
    return df

################################################################################