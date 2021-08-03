from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPPDIR = os.path.dirname(os.path.dirname(os.path.dirname(CDIR)))

sys.path.append(PPPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from processing.split_data import get_split_labels_training_df

train_df, val_df, test_df = get_split_labels_training_df()

data = Dataset.load_from_df(train_df, reader=Reader(rating_scale=(0,1)))

algo = SVD()

#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

algo.fit(data.build_full_trainset())
breakpoint()

################################################################################



################################################################################


################################################################################

def main():
    pass

################################################################################
if __name__ == "__main__":
    main()