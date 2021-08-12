

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere
from models.content_based_filtering import product_vector_similarity, customer_vector_similarity
from misc import constants
from models import common_funcs
from processing import split_data, data_loading

from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from scipy.sparse import coo_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

def get_interaction_matrix(df, df_column_as_row, df_column_as_col, df_column_as_value, row_indexing_map, 
                          col_indexing_map):
    
    row = df[df_column_as_row].apply(lambda x: row_indexing_map[x]).values
    col = df[df_column_as_col].apply(lambda x: col_indexing_map[x]).values
    value = df[df_column_as_value].values
    
    return coo_matrix((value, (row, col)), shape = (len(row_indexing_map), len(col_indexing_map)))

def id_mappings(user_list, item_list, feature_list):
    """
    
    Create id mappings to convert user_id, item_id, and feature_id
    
    """
    user_to_index_mapping = {}
    index_to_user_mapping = {}
    for user_index, user_id in enumerate(user_list):
        user_to_index_mapping[user_id] = user_index
        index_to_user_mapping[user_index] = user_id
        
    item_to_index_mapping = {}
    index_to_item_mapping = {}
    for item_index, item_id in enumerate(item_list):
        item_to_index_mapping[item_id] = item_index
        index_to_item_mapping[item_index] = item_id
        
    feature_to_index_mapping = {}
    index_to_feature_mapping = {}
    for feature_index, feature_id in enumerate(feature_list):
        feature_to_index_mapping[feature_id] = feature_index
        index_to_feature_mapping[feature_index] = feature_id
        
        
    return user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping

def main():

    train_df = common_funcs.get_labels(dataset_to_fetch="train")
    val_df = common_funcs.get_labels(dataset_to_fetch="val")

    
    customer_df = data_loading.get_customers_df()
    product_df = data_loading.get_products_df()
    

    train_df = train_df[(train_df.customerId.isin(customer_df.customerId)) & (train_df.productId.isin(product_df.productId))]
    val_df = val_df[(val_df.customerId.isin(customer_df.customerId)) & (val_df.productId.isin(product_df.productId))]

    user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping = id_mappings(customer_df.customerId.values, product_df.productId.values, product_df)

    user_to_product_interaction_train = get_interaction_matrix(train_df, "customerId", "productId", "purchased", row_indexing_map=user_to_index_mapping, col_indexing_map=item_to_index_mapping)
    user_to_product_interaction_val = get_interaction_matrix(val_df, "customerId", "productId", "purchased", row_indexing_map=user_to_index_mapping, col_indexing_map=item_to_index_mapping)


    model_without_features = LightFM(loss = "warp")

    #===================

    model_without_features.fit(user_to_product_interaction_train,
              user_features=None, 
              item_features=None, 
              sample_weight=None, 
              epochs=1, 
              num_threads=4,
              verbose=False
            )

    auc_without_features = auc_score(model = model_without_features, 
                            test_interactions = user_to_product_interaction_val,
                            num_threads = 4, check_intersections = False)

    breakpoint()

    val_df = val_df[(val_df.customerId.isin(train_df.customerId)) & (val_df.productId.isin(train_df.productId))]
    preds = model_without_features.predict(val_df.customerId.map(user_to_index_mapping).values, val_df.productId.map(item_to_index_mapping).values)
    breakpoint()


if __name__ == "__main__":
    main()