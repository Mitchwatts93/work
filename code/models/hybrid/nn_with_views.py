from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import pandas as pd
import os, sys
import gzip, pickle

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from models.content_based_filtering import product_vector_similarity, customer_vector_similarity
from misc import constants
from models import common_funcs
from processing import split_data, data_loading
from models.hybrid.nn.misc import lr_plotter
from models.hybrid.nn.nn_models import simple_NN_views, deep_NN_views

################################################################################
# constants
PLOT_LR = False # set to True to plot lr graph
DEFAULT_LR = 0.03 # NOTE: I found this from doing the lr_plotter 
DEFAULT_BATCH_SIZE = 50_000
DEFAULT_BALANCE_DATASET = False
DEFAULT_CLASS_WEIGHTING = 0.5
DEFAULT_SHUFFLE_FRAC = 0.3 # NOTE: didn't spend much time thinking about this, 
# probably because of oversampling want this to be larger
DEFAULT_PREFETCH = 2

NN_PLOT_DIR = os.path.join(CDIR, "nn_plots")

################################################################################

def plot_metrics_and_loss_learning(
    history: keras.callbacks.History, model_name_str: str
) -> None:
    """just plotting losses and metrics stored in history object"""

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    save_path = os.path.join(NN_PLOT_DIR, f"{model_name_str}_loss.png")
    plt.savefig(save_path)
    plt.close()

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    save_path = os.path.join(NN_PLOT_DIR, f"{model_name_str}_acc.png")
    plt.savefigsave_path)
    plt.close()

    plt.plot(history.history['auc'], label='train')
    plt.plot(history.history['val_auc'], label='val')
    plt.legend()
    save_path = os.path.join(NN_PLOT_DIR, f"{model_name_str}_auc.png")
    plt.savefig(save_path)
    plt.close()

    plt.plot(history.history['precision'], label='train')
    plt.plot(history.history['val_precision'], label='val')
    plt.legend()
    save_path = os.path.join(NN_PLOT_DIR, f"{model_name_str}_pred.png")
    plt.savefig(save_path)
    plt.close()

    plt.plot(history.history['recall'], label='train')
    plt.plot(history.history['val_recall'], label='val')
    plt.legend()
    save_path = os.path.join(NN_PLOT_DIR, f"{model_name_str}_recall.png")
    plt.savefig(save_path)
    plt.close()


def get_default_bias(train_df: pd.DataFrame) -> float:
    """the bias to set on the final nn layer - according to the bias in the 
    class labels"""
    # set the bias manually to speed up learning
    n_pos = len(train_df[train_df.loc[:, constants.purchased_label_str]])
    n_neg = len(train_df[~train_df.loc[:, constants.purchased_label_str]])
    b0 = np.log([n_pos/n_neg])
    return b0


def construct_train_dataset(
    balance_dataset: bool, train_df: pd.DataFrame, val_df: pd.DataFrame, 
    feature_cols: List,
    pos_weight: float = DEFAULT_CLASS_WEIGHTING, 
    neg_weight: float = DEFAULT_CLASS_WEIGHTING,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_frac: float = DEFAULT_SHUFFLE_FRAC,
    prefetch_n: int = DEFAULT_PREFETCH
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """construct the trianing dataset for features and labels. if balancing, 
    then do so by oversampling the rarer class - i.e. by making the class 
    balance according to the positive and negative labels.
    Args:
        balance_dataset: bool to decide whether to balance the dataset or not
        train_df: df to train on 
        val_df: df for early stopping
        pos_weight: positive class weight [0-1], pos_weight+neg_weight must = 1
        neg_weight: negative class weight [0-1], pos_weight+neg_weight must = 1
        batch_size: batch size for training
        shuffle_frac: what proportion of the dataset size should be set as 
            shuffle buffer
        prefetch_n: how many batches to prefetch when training
    Returns:
        train_dataset, val_dataset: tf datasets for train and val set - ready
            to be used in .fit method
    """
    # NOTE: we have an imbalanced dataset, rather than use class-weights, I'll use oversampling, as this will be a smoother evolution (more positive samples in each batch rather than one heavily weighted sample)

    if pos_weight + neg_weight != 1:
        raise constants.IncorrectWeightError(
            "the positive and negative class weights must add to 1"
        )

    if balance_dataset:
        pos_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                train_df[train_df.purchased][
                    [feature_cols]
                ].values, 
                train_df[train_df.purchased].purchased.values
            )
        )
        neg_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                train_df[~train_df.purchased][
                    [feature_cols]
                ].values, 
                train_df[~train_df.purchased].purchased.values
            )
        )
        pos_val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                val_df[val_df.purchased][
                    [feature_cols]
                ].values, 
                val_df[val_df.purchased].purchased.values
            )
        )
        neg_val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                val_df[~val_df.purchased][
                    [feature_cols]
                ].values, 
                val_df[~val_df.purchased].purchased.values
            )
        )

        resampled_train_dataset = tf.data.experimental.sample_from_datasets(
            [pos_train_dataset, neg_train_dataset], 
            weights=[pos_weight, neg_weight]
        )
        resampled_val_dataset = tf.data.experimental.sample_from_datasets(
            [pos_val_dataset, neg_val_dataset], 
            weights=[pos_weight, neg_weight]
        )
    else:
        resampled_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                train_df[
                    [feature_cols]
                ].values, 
                train_df.purchased.values
            )
        )
        resampled_val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                val_df[
                    [feature_cols]
                ].values, 
                val_df.purchased.values
            )
        )

    train_dataset = resampled_train_dataset.shuffle(
            int(len(train_df) * shuffle_frac) # shuffle buffer size
        ).batch(batch_size).prefetch(prefetch_n) 
    val_dataset = resampled_val_dataset.shuffle(
            int(len(val_df) * shuffle_frac) # shuffle buffer size
        ).batch(batch_size).prefetch(prefetch_n)

    return train_dataset, val_dataset


def train_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, 
    feature_cols: List,
    learning_rate: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    balance_dataset: bool = DEFAULT_BALANCE_DATASET,
    pos_weight: float = DEFAULT_CLASS_WEIGHTING, 
    neg_weight: float = DEFAULT_CLASS_WEIGHTING,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """train the model on train_df, return the model and history object
    Args:
        learning_rate: learning rate for fitting
        balance_dataset: bool to decide whether to balance the dataset or not
        train_df: df to train on 
        val_df: df for early stopping
        pos_weight: positive class weight [0-1], pos_weight+neg_weight must = 1
        neg_weight: negative class weight [0-1], pos_weight+neg_weight must = 1
        batch_size: batch size for training
    Returns:
        model: trained keras model
        history: history object from training
    """


    # get encoded matrices for customers and products, as well as lookup dicts 
    # so we can index rows properly
    row_lookup_customers, encoded_customers = customer_vector_similarity.encode_customer()
    row_lookup_products, encoded_products = product_vector_similarity.encode_products()

    # TODO tidy up
    views = data_loading.get_views_df()
    non_repeating_inds = views[['customerId', 'productId']].drop_duplicates().index # we have some duplciate customer-product ids, each pair only results in one purchase, so I think this is due to a purchase before the end of the month, then more views after? if so we don't care about that. Otherwise maybe its incorrect data
    views = views.loc[non_repeating_inds, :] # all pairs are now unique
    views = views[['customerId','productId', 'viewOnly','changeThumbnail','imageZoom','viewCatwalk','view360','sizeGuide']]
    std_scaler = StandardScaler()
    views[['viewOnly','changeThumbnail','imageZoom' , 'viewCatwalk','view360','sizeGuide']] = std_scaler.fit_transform(views[['viewOnly','changeThumbnail','imageZoom','viewCatwalk','view360','sizeGuide']].values)

    train_df = pd.merge(views, train_df, on=['customerId', 'productId'], how='inner') # merge views with train_df
    test_df = pd.merge(views, test_df, on=['customerId', 'productId'], how='inner') # views with test_df

    # make sure train and test only contain customers and products that are 
    # encoded
    train_df = train_df[
        (
            train_df[constants.customer_id_str].\
                isin(row_lookup_customers.keys())
        ) & (
            train_df[constants.product_id_str].\
                isin(row_lookup_products.keys())
        )
    ]
    test_df = test_df[
        (
            test_df[constants.customer_id_str].\
                isin(row_lookup_customers.keys())
        ) & (
            test_df[constants.product_id_str].\
                isin(row_lookup_products.keys())
        )
    ]
    
    # map the ids to the indices in the encoded matrixes
    train_df.loc[:, constants.customer_id_str] = train_df[
        constants.customer_id_str].map(row_lookup_customers) # now the ids are 
    # the row indexes of the encoder matrix
    train_df.loc[:, constants.product_id_str] = train_df[
        constants.product_id_str].map(row_lookup_products)


    max_prod_ind = max(train_df.productId.max(), test_df.productId.max()) + 1
    max_cust_ind = max(train_df.customerId.max(), test_df.customerId.max()) + 1

    # form a validation set for early stopping
    train_df = train_df.iloc[:int(len(train_df) * 0.8)]
    val_df = train_df.iloc[int(len(train_df) * 0.8):]

    # construct the datasets, 
    train_dataset, val_dataset = construct_train_dataset(
        balance_dataset=balance_dataset, 
        train_df=train_df, val_df=val_df, 
        pos_weight=pos_weight, neg_weight=neg_weight,
        batch_size=batch_size,
        feature_cols=feature_cols,
    )

    # init model
    model = simple_NN_views(
        highest_customer_ind=max_cust_ind, 
        highest_product_ind=max_prod_ind, 
        encoded_customers=encoded_customers.todense(), 
        encoded_products=encoded_products.todense(), 
        output_bias=get_default_bias(train_df=train_df)
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc', 
        mode='max',
        patience=4,
        restore_best_weights=True,
    )
    
    if PLOT_LR: # plot lr graph
        x_small = list(train_dataset.as_numpy_iterator())[0][0][:1000]
        y_small = list(train_dataset.as_numpy_iterator())[0][1][:1000]
        lr_plotter(model, x_small, y_small)

    model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        metrics=[
            'accuracy', 
            keras.metrics.Precision(), 
            keras.metrics.Recall(), 
            keras.metrics.AUC(), 
            keras.metrics.BinaryAccuracy()
        ]
    )

    # fit the model
    history = model.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=20,
        callbacks=[early_stopping],
        shuffle=True,
    )

    return model, history, row_lookup_customers, row_lookup_products


def make_model_preds(
    test_df: pd.DataFrame, 
    model: keras.Model,
    row_lookup_customers: Dict,
    row_lookup_products: Dict,
    feature_cols: List,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """construct dataset from test_df and used the passed model to make 
    predictions"""

    test_df_mapped = test_df.copy() 
    test_df_mapped.loc[:, constants.customer_id_str] = \
        test_df_mapped[constants.customer_id_str].map(row_lookup_customers) # 
    # now the ids are the row indexes of the encoder matrix 
    # NOTE: missing customers will go!?
    
    test_df_mapped.loc[:, constants.product_id_str] = \
        test_df_mapped[constants.product_id_str].map(row_lookup_products)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            test_df_mapped[
                [feature_cols]
            ].values, 
            test_df_mapped.purchased.values
        )
    )
    test_dataset = test_dataset.batch(batch_size) # no shuffle!
    predictions = model.predict(test_dataset)

    return predictions


def predict_and_save_holdout_sets(
    row_lookup_customers: Dict, row_lookup_products: Dict, 
    trained_model: keras.Model, 
    feature_cols: List,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> None:

    train, val, holdout_test_set_df = split_data.get_split_labels_training_df()
    views = data_loading.get_views_df()


    holdout_test_set_df = pd.merge(
        views, 
        holdout_test_set_df, 
        on=[constants.customer_id_str, constants.product_id_str], 
        how='inner'
    ) # views with test_set_df

    holdout_test_set_df = holdout_test_set_df[
        (
            holdout_test_set_df[constants.customer_id_str].isin(row_lookup_customers.keys())
        ) & (
            holdout_test_set_df[constants.product_id_str].isin(row_lookup_products.keys())
        )
    ]

    holdout_test_predictions = make_model_preds(
        test_df=holdout_test_set_df, 
        model=trained_model,
        feature_cols=feature_cols,
        row_lookup_customers=row_lookup_customers,
        row_lookup_products=row_lookup_products,
        batch_size=batch_size,
    )
    holdout_test_set_df['purchased'] = holdout_test_predictions
    holout_set_save_path = os.path.join(constants.PREDICTIONS_PATH, 'nnv2_test_set.gzip')
    with gzip.open(holout_set_save_path, 'wb') as f:
        pickle.dump(holdout_test_set_df, f, protocol=4)


    overall_holdout_set = data_loading.get_labels_predict_df()

    holdout_test_set_df = pd.merge(
        views, 
        holdout_test_set_df, 
        on=[constants.customer_id_str, constants.product_id_str], 
        how='inner'
    ) # views with test_set_df

    overall_holdout_set = overall_holdout_set[
        (
            overall_holdout_set[constants.customer_id_str].isin(row_lookup_customers.keys())
        ) & (
            overall_holdout_set[constants.product_id_str].isin(row_lookup_products.keys())
        )
    ]

    overall_holdout_set_predictions = make_model_preds(
        test_df=overall_holdout_set, 
        model=trained_model,
        feature_cols=feature_cols,
        row_lookup_customers=row_lookup_customers,
        row_lookup_products=row_lookup_products,
        batch_size=batch_size,
    )
    overall_holdout_set['purchase_probability'] = overall_holdout_set_predictions
    
    holout_set_save_path = os.path.join(constants.PREDICTIONS_PATH, 'nnv2_holdout_set.gzip')
    with gzip.open(holout_set_save_path, 'wb') as f:
        pickle.dump(overall_holdout_set, f, protocol=4)
    # NOTE missing about 4k preds


################################################################################

def get_hybrid_nn_probs(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    model_name_str: str,
    learning_rate: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    balance_dataset: bool = DEFAULT_BALANCE_DATASET,
    pos_weight: float = DEFAULT_CLASS_WEIGHTING, 
    neg_weight: float = DEFAULT_CLASS_WEIGHTING,
    predict_holdout_sets: bool = True,
) -> pd.DataFrame:
    # setup devices. only using 1 gpu, edit for more or for none.
    physical_devices = tf.config.list_physical_devices('GPU')[:1]
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    feature_cols = [constants.product_id_str, constants.customer_id_str, "viewOnly", "changeThumbnail", "imageZoom", "viewCatwalk", "view360", "sizeGuide"]

    # get trained model and history
    trained_model, history, row_lookup_customers, row_lookup_products = train_model(
        train_df=train_df, test_df=test_df, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        balance_dataset=balance_dataset,
        pos_weight=pos_weight, 
        neg_weight=neg_weight,
        feature_cols=feature_cols,
    )

    # plot everything
    plot_metrics_and_loss_learning(
        history=history, model_name_str=model_name_str
    )

    # make predicitons on test set
    predictions = make_model_preds(
        test_df=test_df, model=trained_model,
        row_lookup_customers=row_lookup_customers,
        row_lookup_products=row_lookup_products,
        batch_size=batch_size,
        feature_cols=feature_cols,
    )
    test_df.loc[:, constants.probabilities_str] = predictions # NOTE: same name column as labels
    
    if predict_holdout_sets:
        predict_and_save_holdout_sets(
            row_lookup_customers=row_lookup_customers, 
            row_lookup_products=row_lookup_products, 
            trained_model=trained_model, 
            batch_size=batch_size,
            feature_cols=feature_cols,
        )
        
    return test_df


################################################################################

def main():
    model_name = "hybrid_nn_views_simple_unbalanced"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(
        model_name=model_name, 
        model_fetching_func=get_hybrid_nn_probs, 
        dataset_being_evaluated=dataset_being_evaluated,
        model_name_str=model_name,
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
    )

################################################################################

if __name__ == "__main__":
    main()
