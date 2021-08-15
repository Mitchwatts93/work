"""functions to help with saving to and loading from disk.
"""

import os
import gzip, pickle
from collections.abc import Callable
from typing import Dict

################################################################################

def is_cached(filepath: os.PathLike) -> bool:
    """check if file is already cached"""
    return os.path.isfile(filepath)


def load_from_cache(filepath: os.PathLike):
    """load file from cache using gzip/pickle"""
    with gzip.open(filepath, 'rb') as f:
        file = pickle.load(f)
    return file


def cache_file(file, filepath: os.PathLike) -> None:
    """save file to disk using pickle/gzip"""
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(file, f, protocol=4)


def load_or_make_wrapper(
    maker_func: Callable, filepath: os.PathLike, cache: bool = True, 
    **inputs: Dict
):
    """if file is already cached, then load from file, otherwise get the data 
    by running the callable function. if cache is true then save the data to 
    disk.
    Args:
        maker_func: function that returns data of interest. Additionally takes 
            inputs dict as kwargs input.
        filepath: filepath from which to check cache, or to save to cache.
        cache: boolean of whether to save to cache or not.
        **inputs: dictionary of additional keyword arg inputs to maker_func
    Returns:
        data: type of data is unknown, same as type returned from maker_func.
    """
    if is_cached(filepath=filepath):
        data = load_from_cache(filepath=filepath)
    else:
        if 'additional_kwargs_for_model' in inputs:
            additional_args = inputs.pop('additional_kwargs_for_model')
            inputs.update(additional_args)
        data = maker_func(**inputs)
        if cache:
            cache_file(file=data, filepath=filepath)
    return data

