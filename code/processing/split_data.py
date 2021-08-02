

# we will split the data based on labels_training.txt - into a train val and test with 60:30:10 split.
# the reason we use this is because this is the final labels
# we will stratify on customers and products.
# Should we hold back some customers and products for unseen evaluation? seems like a good idea...
# Okay so split stratify will only preserve the distribution of customers, not what we want...
# maybe we just split on unique customers, and go from there? assume fixed product base for now? or also split on products?

# once we have our list of customers, we should select them from labels_training.txt, and shuffle

# datasets:
# views: contains all of the purchases data

# purchases contains everything in labels and more
# customers is from labels
# products is from labels

# we could split on customers, and split on products
# we can stratify based on the features in each, to get balanced sets
# then we can select the combination? but how do we know the combination is correct?
# I think we don't, and it will cause problems. So I think a good way to go about this is to split customers
# then select the train set from labels, and select products from that? - and hope product distributions are okay?
# then we can use anything in purchases based on customers also, and then on views too?

# the instructions say to be careful of data leakage, especially considering future events are wanted...
# so maybe the thing to do is split on time? and to do that, you can go on purchases?
# without this, there would be the possibility of data leakage, since they would have bought things before and after a product, which might 
# influence the predictions -> the evaluation.
# so to do it based on date, we split on date in purchases - which contains all of labels
# therefore labels will be split by this straight away
# also views will be split by this
# the products and customers can be split in the same way?
# 


# views df is only for january! and purchases runs until the end of jan...

# split on labels, since this is jan 2017 only, thats all we need...
# purchases goes back further which is good, so we can use historical also?
# can we use purchases for customers not in labels? yes?
# 


import os, sys
from typing import Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from processing import data_loading
from misc import constants
from misc.caching import load_or_make_wrapper


################################################################################
# constants

TRAINING_START_DATE = datetime(day=1, month=1, year=2017, tzinfo=timezone.utc)
VALIDATION_START_DATE = datetime(day=22, month=1, year=2017, tzinfo=timezone.utc)
TEST_START_DATE = datetime(day=27, month=1, year=2017, tzinfo=timezone.utc)

################################################################################

def split_purchases_df() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    purchases_df = data_loading.get_purchases_df()

    train_purchases = purchases_df[purchases_df.date < VALIDATION_START_DATE]
    val_purchases = purchases_df[(purchases_df.date >= VALIDATION_START_DATE) & (purchases_df.date < TEST_START_DATE)]
    test_purchases = purchases_df[purchases_df.date >= TEST_START_DATE]

    return train_purchases, val_purchases, test_purchases

################################################################################

def split_views_df():
    views_df = data_loading.get_views_df()

    train_views = views_df[(views_df.date >= TRAINING_START_DATE) & (views_df.date < VALIDATION_START_DATE)]
    val_views = views_df[(views_df.date >= VALIDATION_START_DATE) & (views_df.date < TEST_START_DATE)]
    test_views = views_df[(views_df.date >= TEST_START_DATE)]

    return train_views, val_views, test_views


def split_products_df(train_purchases, val_purchases, test_purchases):
    products_df = data_loading.get_products_df()

    train_products = products_df[products_df.productId.isin(train_purchases.productId.unique())]
    val_products = products_df[products_df.productId.isin(val_purchases.productId.unique())]
    test_products = products_df[products_df.productId.isin(test_purchases.productId.unique())]

    return train_products, val_products, test_products


def split_customers_df(train_purchases, val_purchases, test_purchases):
    customers_df = data_loading.get_customers_df()

    train_customers = customers_df[customers_df.customerId.isin(train_purchases.customerId.unique())]
    val_customers = customers_df[customers_df.customerId.isin(val_purchases.customerId.unique())]
    test_customers = customers_df[customers_df.customerId.isin(test_purchases.customerId.unique())]

    return train_customers, val_customers, test_customers


def split_labels_training_df(train_purchases, val_purchases, test_purchases):
    labels_training_df = data_loading.get_labels_training_df()

    train_labels_training = labels_training_df[(labels_training_df["customerId"] + labels_training_df["productId"]).isin(train_purchases["customerId"] + train_purchases["productId"])]
    val_labels_training = labels_training_df[(labels_training_df["customerId"] + labels_training_df["productId"]).isin(val_purchases["customerId"] + val_purchases["productId"])]
    test_labels_training = labels_training_df[(labels_training_df["customerId"] + labels_training_df["productId"]).isin(test_purchases["customerId"] + test_purchases["productId"])]

    return train_labels_training, val_labels_training, test_labels_training


################################################################################

def get_split_purchases_df():
    cache_path = constants.SPLIT_PURCHASES_PATH
    train_purchases, val_purchases, test_purchases = load_or_make_wrapper(
        maker_func=split_purchases_df, 
        filepath=cache_path, cache=True, 
    )
    return train_purchases, val_purchases, test_purchases


def get_split_views_df():
    cache_path = constants.SPLIT_VIEWS_PATH
    train_views, val_views, test_views = load_or_make_wrapper(
        maker_func=split_views_df, 
        filepath=cache_path, cache=True, 
    )
    return train_views, val_views, test_views


def get_split_products_df():
    train_purchases, val_purchases, test_purchases = get_split_purchases_df()

    cache_path = constants.SPLIT_PRODUCTS_PATH
    train_products, val_products, test_products = load_or_make_wrapper(
        maker_func=split_products_df, 
        filepath=cache_path, cache=True, 
        train_purchases=train_purchases, 
        val_purchases=val_purchases, 
        test_purchases=test_purchases
    )
    return train_products, val_products, test_products


def get_split_customers_df():
    train_purchases, val_purchases, test_purchases = get_split_purchases_df()

    cache_path = constants.SPLIT_CUSTOMERS_PATH
    train_customers, val_customers, test_customers = load_or_make_wrapper(
        maker_func=split_customers_df, 
        filepath=cache_path, cache=True, 
        train_purchases=train_purchases, 
        val_purchases=val_purchases, 
        test_purchases=test_purchases
    )
    return train_customers, val_customers, test_customers


def get_split_labels_training_df():
    train_purchases, val_purchases, test_purchases = get_split_purchases_df()

    cache_path = constants.SPLIT_LABELS_TRAINING_PATH
    train_labels_training, val_labels_training, test_labels_training = load_or_make_wrapper(
        maker_func=split_labels_training_df, 
        filepath=cache_path, cache=True, 
        train_purchases=train_purchases,
        val_purchases=val_purchases,
        test_purchases=test_purchases,
    )
    return train_labels_training, val_labels_training, test_labels_training



################################################################################


def main():

    """load_or_make_wrapper"""
    train_purchases, val_purchases, test_purchases = get_split_purchases_df()
    
    train_views, val_views, test_views = get_split_views_df()

    train_products, val_products, test_products = get_split_products_df()

    train_customers, val_customers, test_customers = get_split_customers_df()
    
    train_labels_training, val_labels_training, test_labels_training = get_split_labels_training_df()


if __name__ == "__main__":
    main()

