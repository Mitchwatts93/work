import os
import gzip, pickle
from collections.abc import Callable

################################################################################

def is_cached(filepath: os.PathLike) -> bool:
    return os.path.isfile(filepath)


def load_from_cache(filepath: os.PathLike):
    with gzip.open(filepath, 'rb') as f:
        file = pickle.load(f)
    return file


def cache_file(file, filepath: os.PathLike) -> None:
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(file, f, protocol=4)


def load_or_make_wrapper(
    maker_func: Callable, filepath: os.PathLike, cache: bool = True, **inputs
):
    if is_cached(filepath=filepath):
        data = load_from_cache(filepath=filepath)
    else:
        data = maker_func(**inputs)
        if cache:
            cache_file(file=data, filepath=filepath)
    return data

