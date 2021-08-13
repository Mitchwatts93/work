"""normalised popularity baseline
"""
import numpy as np
import pandas as pd

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc.caching import load_or_make_wrapper
from misc import constants
from processing.split_data import get_split_purchases_df
from models import common_funcs

################################################################################
# constants
normalised_popularity_str = "normalised_popularity"
################################################################################

def normalised_popularity(purchases_df: pd.DataFrame) -> pd.DataFrame:
    """returns normalised probability saved in pd dataframe.
    normalised popularity of a product is the number of times that product has
    been purchased divided by the number of times the most popular product has
    been purchased. So is a value [0-1], which 0 has never been purchased, and
    1 has been purchased as many or more times than any product.
    Args:
        purchases_df: dataframe containing productIds with each row a purchase.
    Returns:
        normalised_popularities: pd dataframe containing a normalised popularity
            score for each productId.
    """
    n_purchases = purchases_df["productId"].value_counts() # no. times each
    # product purchased
    n_purchases_most_popular_product = n_purchases.max()

    purchases_df[normalised_popularity_str] = n_purchases[
            purchases_df[constants.product_id_str].values
        ].values / n_purchases_most_popular_product # get the normalised 
    # popularity scores

    normalised_popularities = purchases_df["normalised_popularity"] # rename col
    normalised_popularities.index = purchases_df["productId"] # index reset as 
    # product id

    normalised_popularities.drop_duplicates(inplace=True) # shouldn't be any 
    # duplicates but just in case

    return normalised_popularities


def get_normalised_popularities(
    dataset_being_evaluated: str = "val"
) -> pd.DataFrame:
    """get the normalised popularity df, either from cache or from computing 
    the data.
    Args:
        dataset_being_evaluated: string to indicate which dataset we are 
            evaluating. Either train, val or test
    Returns:
        normalised_popularities: pd dataframe containing a normalised popularity
            score for each productId.
    """
    
    # load data
    train_purchases_df, val_purchases_df, test_purchases_df = \
        get_split_purchases_df()

    # join the datasets to cheat a little and make larger sets. This should be 
    # deleted. TODO fix this?
    if dataset_being_evaluated == "train":
        purchases_df = pd.concat([val_purchases_df, test_purchases_df])
    elif dataset_being_evaluated == "val":
        purchases_df = pd.concat([train_purchases_df, test_purchases_df])
    else:
        purchases_df = pd.concat([train_purchases_df, val_purchases_df])


    cache_path = os.path.join(
        constants.RAW_DATA_DIR, 
        f"normalised_popularity_{dataset_being_evaluated}.gzip"
    )
    normalised_popularities = load_or_make_wrapper(
        maker_func=normalised_popularity, 
        filepath=cache_path, cache=True, 
        purchases_df=purchases_df,
    )
    return normalised_popularities

################################################################################

def fill_missing_popularities(
    product_ids: pd.Series, 
    normalised_popularities: pd.DataFrame, 
    fill_method: str = "mean", 
    fill_value: bool = None
):
    """There will be some popularities which are missing from predictions, 
    because they aren't in their training set. Set these values accordingly.
    Args:
        product_ids: pd series containing product ids in the set to be predicted
        normalised_popularities: normalised popularity scores for the set to be
            predicted, may be missing productIds from its index compared to 
            product_ids df.
        fill_method: how to fill in missing values
        fill_value: if not using a fill method, what to fill missing values as.
            if fill_method is not None then this won't be used.
    Returns:
        filled_popularities: pd dataframe containing original 
            normalised_popularities data, with additional filled data for 
            missing productIds that are in product_ids.
    """
    # which ids are missing?
    missing_products = product_ids[~product_ids.isin(
            normalised_popularities.index
        )].unique()
    # check inputs are correct
    if fill_method is None and fill_value is None:
        raise constants.InputError(
            "fill_method and fill_value cannot both be None"
        )
    elif fill_method is not None:
        # use that fill method
        fill_value = pd.eval(f"normalised_popularities.{fill_method}()")
    # otherwise we will just use the fill_value

    # fill in the missing values for the missing ids
    missing_products_series = pd.Series(index=missing_products, data=fill_value)

    # join the two into the same df as the input
    filled_popularities = pd.concat(
        [normalised_popularities, missing_products_series]
    )

    return filled_popularities

################################################################################

def get_normalised_popularity_purchase_probabilities(
    train_df: pd.DataFrame, test_df: pd.DataFrame, 
    dataset_being_evaluated: str = "val",
    fill_method: str = "mean"
) -> np.ndarray:
    """returns the purchase probabilities using the normalised popularity 
    method.
    Args:
        train_df: pd dataframe containing training data. Note this is necessary 
            for this function to be compatible with 
            common_funcs.generate_and_cache_preds
        test_df: pd dataframe containing test data. Note this is necessary 
            for this function to be compatible with 
            common_funcs.generate_and_cache_preds
    Returns:
        test_df: change the column for labels to be predicted values instead
    """
    # get the ids to predict
    product_ids = test_df[constants.product_id_str]
    normalised_popularities = get_normalised_popularities(
        dataset_being_evaluated=dataset_being_evaluated
    ) # get the scores
    filled_normalised_popularities = fill_missing_popularities(
        product_ids=product_ids, 
        normalised_popularities=normalised_popularities, 
        fill_method=fill_method,
    ) # fill in missing ids using the specified method

    # the np array of predicted probs
    product_popularities = filled_normalised_popularities[product_ids].values

    # put into input df
    test_df.loc[:, constants.probabilities_str] = product_popularities

    return test_df

################################################################################

def main():
    """get and cache predictions using normalised popularity baseline model"""
    model_name = "normalised_popularity_baseline"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(
        model_name=model_name,
         model_fetching_func=get_normalised_popularity_purchase_probabilities, 
        dataset_being_evaluated=dataset_being_evaluated, 
        additional_kwargs_for_model={
            "dataset_being_evaluated":dataset_being_evaluated
        }
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
    ) # update the master scores dict


if __name__ == "__main__":
    main()
    