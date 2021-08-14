"""functions for making predictions using manual memory-based approach"""
import os, sys
from typing import Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

CDIR = os.path.dirname(os.path.abspath(__file__))
PPPDIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            CDIR
        )
    )
)

sys.path.append(PPPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from processing.purchases import get_purchases_df
from processing import split_data
from misc import caching, constants

################################################################################

def get_co_occurance_df(purchases_df: pd.DataFrame) -> pd.DataFrame:
    co_occurances_multiind = purchases_df.groupby(['customerId', 'productId']).size()
    co_occurances = pd.DataFrame(co_occurances_multiind.to_frame().to_records())
    return co_occurances


def make_sparse_utility_matrix(co_occurances_df: pd.DataFrame) -> sparse.csr_matrix:
    col_product_labels = co_occurances_df.productId.unique()
    col_ind_dict = dict(zip(col_product_labels, range(len(co_occurances_df))))

    row_customer_labels = co_occurances_df.customerId.unique()
    row_ind_dict = dict(zip(row_customer_labels, range(len(co_occurances_df))))
    
    zeros = np.zeros((co_occurances_df.customerId.unique().shape[0], co_occurances_df.productId.unique().shape[0]), dtype='int32')
    prefilled_arr = zeros

    for i, row in tqdm(co_occurances_df.iterrows(), total=len(co_occurances_df)):
        row_ind = row_ind_dict[row.customerId]
        col_ind = col_ind_dict[row.productId]
        prefilled_arr[row_ind, col_ind] += row['0']
    
    sparse_utility_mat = sparse.csr_matrix(prefilled_arr).T
    row_product_labels, col_customer_labels = col_product_labels, row_customer_labels # because of the .T
    return sparse_utility_mat, row_product_labels, col_customer_labels


def make_similarity_matrix():
    train_purchases, _val_purchases, _test_purchases = split_data.get_split_purchases_df()
    co_occurances_df = get_co_occurance_df(train_purchases)
    sparse_utility_mat, row_product_labels, _col_customer_labels = get_sparse_utility_matrix(co_occurances_df)
    sim_mat = cosine_similarity(sparse_utility_mat) # requires a ~500gb ram
    # TODO why does cosine similarity not keep it sparse??

    sparse_sim_mat = sparse.csr_matrix(sim_mat) # row_product_labels x row_product_labels shape
    return sparse_sim_mat, row_product_labels


################################################################################


def get_sparse_utility_matrix(co_occurances_df: pd.DataFrame) -> sparse.coo_matrix:
    cache_filepath = os.path.join(
        constants.MODEL_FILES_DIR, 
        'collaborative_filtering', 
        'sparse_utility_matrix.gzip'
    )
    sim_matrix, col_product_labels, row_customer_labels = caching.load_or_make_wrapper(
        maker_func=make_sparse_utility_matrix, filepath=cache_filepath,
        co_occurances_df=co_occurances_df
    )
    return sim_matrix, col_product_labels, row_customer_labels


def get_similarity_matrix():
    cache_filepath = os.path.join(
        constants.MODEL_FILES_DIR, 
        'collaborative_filtering', 
        'similarity_matrix.gzip'
    )
    sim_matrix, product_labels = caching.load_or_make_wrapper(
        maker_func=make_similarity_matrix, filepath=cache_filepath
    )
    return sim_matrix, product_labels

################################################################################

def get_customer_previous_purchases(customerId):
    purchases_df = get_purchases_df()

    previous_product_ids = purchases_df[purchases_df['customerId'] == customerId]['productId'].to_numpy()
    return previous_product_ids


def make_pred(sim_mat, sim_mat_labels, previous_product_ids, new_product_id):
    old_product_inds = [np.argwhere(np.array(sim_mat_labels) == p_label) for p_label in previous_product_ids]
    new_product_ind = np.argwhere(np.array(sim_mat_labels) == new_product_id)

    similarities = [sim_mat[new_product_ind[0][0], old_product_ind[0][0]] for old_product_ind in old_product_inds]
    # NOTE: if nothing is there, then by default its zero?
    breakpoint()
    #similarities = [sim for sim in similarities if sim != 0]
    mean_sim = np.nanmean(similarities)

    return mean_sim


################################################################################

def are_args_valid(args_dict: Dict) -> bool:

    if args_dict.filepath is None and (args_dict.customerId is None and args_dict.productId is None):
        return False
    else:
        return True


def main():
    """
    parser = argparse.ArgumentParser(description='get predictions for '
        'memory-based collaborative filtering algorithm')
    parser.add_argument('--filepath',  '-f', type=str, default=None,
                        help='absolute path to file to be evaluated. file must '
                        'be csv with the header: customerId, productid')
    parser.add_argument('--customerId', '-cid', type=int, default=None,
                        help='absolute path to file to be evaluated. file must '
                        'be csv with the header: customerId, productid')
    parser.add_argument('--productId', '-pid', type=int, default=None,
                        help='absolute path to file to be evaluated. file must '
                        'be csv with the header: customerId, productid')
    args_dict = parser.parse_args()

    if not are_args_valid(args_dict):
        raise constants.InputError("the inputs are invalid")

    """


    sim_mat, product_labels = get_similarity_matrix()

    
    _train_labels_training, val_labels_training, test_labels_training = split_data.get_split_labels_training_df()

    def make_preds(df):
        products_set = set(product_labels)
        pred_df = df.copy()
        pred_df["guess"] = False
        for customer_id in df.customerId.unique():
            print("customer: ", customer_id)
            previous_product_ids = get_customer_previous_purchases(customerId=customer_id)
            product_ids = df[df.customerId == customer_id].productId.unique()
            for product_id in product_ids:
                row = pred_df[(pred_df.customerId == customer_id) & (pred_df.productId == product_id)]
                if product_id not in products_set:
                    mean_sim = np.random.rand(1)[0] # random guess
                    row.loc["guess"] = True
                else:
                    mean_sim = make_pred(sim_mat, sim_mat_labels=product_labels, previous_product_ids=previous_product_ids, new_product_id=product_id)
                row.loc["purchased"] = mean_sim

        breakpoint()
        return pred_df
    
    make_preds(df=val_labels_training)
    breakpoint()
    make_preds(df=test_labels_training)
    breakpoint()
    

    """
    # TODO this is so slow!
    if args_dict['filepath'] is not None:
        breakpoint()
        prod_df = pd.read_csv(args_dict.filepath)
        scores = []
        for i, row in prod_df.iterrows():
            previous_product_ids = get_customer_previous_purchases(customerId=args_dict.cid)
            mean_sim = make_pred(sim_mat, product_labels, cid=row.customerId, pid=row.productId)
            scores.append(mean_sim)
    else:
        previous_product_ids = get_customer_previous_purchases(customerId=args_dict.cid)
        mean_sim = make_pred(sim_mat, sim_mat_labels=product_labels, previous_product_ids=previous_product_ids, new_product_id=args_dict.pid)
    
    """

if __name__ == "__main__":
    main() 