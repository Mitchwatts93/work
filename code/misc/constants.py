import os 

CDIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(CDIR, '..', '..', 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
CUSTOMERS_PATH = os.path.join(RAW_DATA_DIR, "customers.txt")
SPLIT_CUSTOMERS_PATH = os.path.join(RAW_DATA_DIR, "split_customers.gzip")
PRODUCTS_PATH = os.path.join(RAW_DATA_DIR, "products.txt")
SPLIT_PRODUCTS_PATH = os.path.join(RAW_DATA_DIR, "split_products.gzip")
PURCHASES_PATH = os.path.join(RAW_DATA_DIR, "purchases.txt")
SPLIT_PURCHASES_PATH = os.path.join(RAW_DATA_DIR, "split_purchases.gzip")
VIEWS_PATH = os.path.join(RAW_DATA_DIR, "views.txt")
SPLIT_VIEWS_PATH = os.path.join(RAW_DATA_DIR, "split_views.gzip")
LABELS_TRAINING_PATH = os.path.join(RAW_DATA_DIR, "labels_training.txt")
SPLIT_LABELS_TRAINING_PATH = os.path.join(RAW_DATA_DIR, "split_labels_training.gzip")
LABELS_PREDICT_PATH = os.path.join(RAW_DATA_DIR, "labels_predict.txt")

MODEL_FILES_DIR = os.path.join(DATA_DIR, 'model_files')

PREDICTIONS_PATH = os.path.join(DATA_DIR, 'predictions')

SCORES_PATH = os.path.join(DATA_DIR, 'scores')
VAL_SCORES_DICT = os.path.join(SCORES_PATH, "model_val_scores.json")
TEST_SCORES_DICT = os.path.join(SCORES_PATH, "model_test_scores.json")

class InputError(Exception):
    pass


default_threshold = 0.5