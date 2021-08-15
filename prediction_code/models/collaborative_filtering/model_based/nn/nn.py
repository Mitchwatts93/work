"""nn model functions"""

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

import os, sys
CDIR = os.path.dirname(os.path.abspath(__file__))
PPPPDIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(CDIR)
        )
    )
)
sys.path.append(PPPPDIR) # rather than force you to add package to path in bash, 
# I've done this for robustness

from misc import constants
from models import common_funcs
from models.collaborative_filtering.model_based.nn.nn_models import simple_NN, deep_NN
from models.collaborative_filtering.model_based.nn.nn_misc import lr_plotter

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
    plt.savefig(save_path)
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


def construct_train_dataset(
    balance_dataset: bool, train_df: pd.DataFrame, val_df: pd.DataFrame, 
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
                train_df[train_df[constants.purchased_label_str]][
                    [constants.product_id_str, constants.customer_id_str]
                ].values, 
                train_df[train_df[constants.purchased_label_str]][constants.purchased_label_str].values
            )
        )
        neg_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                train_df[~train_df[constants.purchased_label_str]][
                    [constants.product_id_str, constants.customer_id_str]
                ].values, 
                train_df[~train_df[constants.purchased_label_str]][constants.purchased_label_str].values
            )
        )
        pos_val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                val_df[val_df[constants.purchased_label_str]][
                    [constants.product_id_str, constants.customer_id_str]
                ].values, 
                val_df[val_df[constants.purchased_label_str]][constants.purchased_label_str].values
            )
        )
        neg_val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                val_df[~val_df[constants.purchased_label_str]][
                    [constants.product_id_str, constants.customer_id_str]
                ].values, 
                val_df[~val_df[constants.purchased_label_str]][constants.purchased_label_str].values
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
                    [constants.product_id_str, constants.customer_id_str]
                ].values, 
                train_df[constants.purchased_label_str].values
            )
        )
        resampled_val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                val_df[
                    [constants.product_id_str, constants.customer_id_str]
                ].values, 
                val_df[constants.purchased_label_str].values
            )
        )

    train_dataset = resampled_train_dataset.shuffle(
            int(len(train_df) * shuffle_frac) # shuffle buffer size
        ).batch(batch_size).prefetch(prefetch_n) 
    val_dataset = resampled_val_dataset.shuffle(
            int(len(val_df) * shuffle_frac) # shuffle buffer size
        ).batch(batch_size).prefetch(prefetch_n)

    return train_dataset, val_dataset


def get_default_bias(train_df: pd.DataFrame) -> float:
    """the bias to set on the final nn layer - according to the bias in the 
    class labels"""
    # set the bias manually to speed up learning
    n_pos = len(train_df[train_df.loc[:, constants.purchased_label_str]])
    n_neg = len(train_df[~train_df.loc[:, constants.purchased_label_str]])
    b0 = np.log([n_pos/n_neg])
    return b0


def train_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, 
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
    # get the sizes the embeddings should be set to
    max_prod_ind = max(train_df.productId.max(), test_df.productId.max()) + 1
    max_cust_ind = max(train_df.customerId.max(), test_df.customerId.max()) + 1

    # split the training df into a train and val, val used for early stopping
    train_df = train_df.iloc[:int(len(train_df) * 0.8)]
    val_df = train_df.iloc[int(len(train_df) * 0.8):]

    # construct the datasets, 
    train_dataset, val_dataset = construct_train_dataset(
        balance_dataset=balance_dataset, 
        train_df=train_df, val_df=val_df, 
        pos_weight=pos_weight, neg_weight=neg_weight,
        batch_size=batch_size,
    )

    model = deep_NN(
        highest_customer_ind=max_cust_ind, 
        highest_product_ind=max_prod_ind, 
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

    history = model.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=10,
        callbacks=[early_stopping],
        shuffle=True,
    )
    return model, history


def make_model_preds(
    test_df: pd.DataFrame, model: keras.Model, 
    batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """construct dataset from test_df and used the passed model to make 
    predictions"""
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            test_df.loc[:,
                [constants.product_id_str, constants.customer_id_str]
            ].values, 
            test_df.loc[:, constants.purchased_label_str].values
        )
    )
    test_dataset = test_dataset.batch(batch_size) # no shuffle!
    predictions = model.predict(test_dataset)
    return predictions

################################################################################

def get_collaborative_nn_probs(
    train_df: pd.DataFrame, test_df: pd.DataFrame, 
    model_name_str: str,
    learning_rate: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    balance_dataset: bool = DEFAULT_BALANCE_DATASET,
    pos_weight: float = DEFAULT_CLASS_WEIGHTING, 
    neg_weight: float = DEFAULT_CLASS_WEIGHTING,
) -> pd.DataFrame:
    """train a nn for collaborative filtering (i.e. trained embeddings plus FC 
    layers) and then make predictions for the test set.
    Args:
        model_name_str: name to save model predictions to cache/load from later
        balance_dataset: bool to decide whether to balance the dataset or not
        learning_rate: learning rate for fitting
        train_df: df to train on 
        test_df: df to make predictions with
        pos_weight: positive class weight [0-1], pos_weight+neg_weight must = 1
        neg_weight: negative class weight [0-1], pos_weight+neg_weight must = 1
        batch_size: batch size for training
    Returns:
        test_df: same as input but with predictions now instead of labels
    """
    # setup devices. only using 1 gpu, edit for more or for none.
    physical_devices = tf.config.list_physical_devices('GPU')[:1]
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    # train the model
    trained_model, history = train_model(
        train_df=train_df, test_df=test_df, balance_dataset=balance_dataset,
        pos_weight=pos_weight, neg_weight=neg_weight,
        learning_rate=learning_rate,
    )

    # plotting    
    plot_metrics_and_loss_learning(
        history=history, model_name_str=model_name_str
    )
    
    # make predictions with trained model
    predictions = make_model_preds(
        test_df=test_df, model=trained_model, batch_size=batch_size
    )

    test_df.loc[:, constants.probabilities_str] = predictions 
    return test_df


################################################################################

def main() -> None:
    model_name = "coll_nn_deep_unbalanced"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(
        model_name=model_name, 
        model_fetching_func=get_collaborative_nn_probs, 
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
