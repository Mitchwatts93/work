"""various constants used elsewhere.
"""
import os 

################################################################################
# paths

CDIR = os.path.dirname(os.path.abspath(__file__)) # current directory absolute
# path


# all data
DATA_DIR = os.path.join(CDIR, '..', '..', 'data') # absolute path to data
# directory

# raw data
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data") # the raw data
CUSTOMERS_PATH = os.path.join(RAW_DATA_DIR, "customers.txt") # the customers 
#file
PRODUCTS_PATH = os.path.join(RAW_DATA_DIR, "products.txt") # the products file
PURCHASES_PATH = os.path.join(RAW_DATA_DIR, "purchases.txt") # the purchases 
#file
VIEWS_PATH = os.path.join(RAW_DATA_DIR, "views.txt") # the views data
LABELS_TRAINING_PATH = os.path.join(RAW_DATA_DIR, "labels_training.txt") # the 
#labels training data
LABELS_PREDICT_PATH = os.path.join(RAW_DATA_DIR, "labels_predict.txt") # the 
#labels holdout set

# split data
SPLIT_DATA_DIR = os.path.join(DATA_DIR, "split_data") # the split pickled data
SPLIT_PURCHASES_PATH = os.path.join(SPLIT_DATA_DIR, "split_purchases.gzip") # 
# the split purchases pickled file
SPLIT_VIEWS_PATH = os.path.join(SPLIT_DATA_DIR, "split_views.gzip") # the split 
# views pickled file
SPLIT_LABELS_TRAINING_PATH = os.path.join(
    SPLIT_DATA_DIR, "split_labels_training.gzip"
) # the split labels pickled file
#SPLIT_CUSTOMERS_PATH = os.path.join(SPLIT_DATA_DIR, "split_customers.gzip")#the 
#SPLIT_PRODUCTS_PATH = os.path.join(SPLIT_DATA_DIR, "split_products.gzip")

# model files
MODEL_FILES_DIR = os.path.join(DATA_DIR, 'model_files') # data for model 
# related file
NORMALISED_POPULARITY_PATH = os.path.join(
    MODEL_FILES_DIR, 
    "normalised_popularity_val.gzip"
) # path to normalised popularity file

# predictions and scores
PREDICTIONS_PATH = os.path.join(DATA_DIR, 'predictions') # dir to store 
# predictions
SCORES_PATH = os.path.join(DATA_DIR, 'scores') # dir to store all metric 
# evaluation results
VAL_SCORES_DICT = os.path.join(SCORES_PATH, "model_val_scores.json") # path to 
# validation scores
TEST_SCORES_DICT = os.path.join(SCORES_PATH, "model_test_scores.json") # path 
# to test scores

# plots
PLOTS_PATH = os.path.join(DATA_DIR, "plots") # path to dir to store plots

################################################################################
# custom exceptions

class InputError(Exception):
    pass

################################################################################
# default values

default_threshold = 0.5 # default threshold when evaluating - 
# probabilities > 0.5 are True, <=0.5 are False
default_value_missing = 0 # the default value to set for missing values. Set as
# zero as the dataset is imbalanced, so this is likely to be the class missing
product_id_str = "productId"
customer_id_str = "customerId"
probabilities_str = "purchased"
predicted_purchased_str = "predicted_purchased"
purchased_label_str = "purchased"
################################################################################