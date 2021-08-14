import numpy as np
import pandas as pd

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from misc import constants
from models import common_funcs

from processing import data_loading

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
################################################################################


def get_embedding_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    
    test_df['purchased'] = predictions # NOTE: same name column as labels
    return test_df

################################################################################

def main() -> None:
    model_name = "embedding"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(model_name=model_name, model_fetching_func=get_embedding_probs, dataset_being_evaluated=dataset_being_evaluated)
    labels = common_funcs.get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores_dict = common_funcs.get_scores(predictions, labels, model_name=model_name, dataset_being_evaluated=dataset_being_evaluated)

    common_funcs.cache_scores_to_master_dict(
        dataset_being_evaluated=dataset_being_evaluated,
        scores_dict=scores_dict,
        model_name=model_name
    )
    
################################################################################

if __name__ == "__main__":
    main()
