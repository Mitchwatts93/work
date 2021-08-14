from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.co_clustering import CoClustering
import gzip, pickle

class simple_NN(keras.Model):

    def __init__(self, highest_customer_ind, highest_product_ind, encoded_customers, encoded_products, output_bias=None):
        super().__init__()
        self.drop_rate = 0.01
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)

        self.encoded_customers = tf.convert_to_tensor(encoded_customers) #encoded_customers
        self.encoded_products = tf.convert_to_tensor(encoded_products)  #encoded_products

        self.prod_fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn_cont = keras.layers.BatchNormalization()
        self.prod_drop_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn_cont = keras.layers.BatchNormalization()
        self.cust_drop_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.views_hid_cont = keras.layers.Dense(units=10, activation='relu')
        self.views_bn_cont = keras.layers.BatchNormalization()
        self.views_drop_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.fc2_cont = keras.layers.Dense(units=1, activation='relu')
        self.fc2_bn_cont = keras.layers.BatchNormalization()
        self.fc2_drop_cont= keras.layers.Dropout(rate=self.drop_rate)

        ###

        self.fc1_tot = keras.layers.Dense(units=3, activation='relu')
        self.fc1_bn_drop = keras.layers.BatchNormalization()
        self.fc1_tot_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.out = keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)


    def call(self, X):
        product_id = tf.cast(X[:, 0], tf.int32)
        customer_id =tf.cast( X[:, 1], tf.int32)
        views = X[:, 2:-1] # this also includes the predictions
        cocluster_preds = X[:, -1:]

        ### content

        product_vec_cont = tf.nn.embedding_lookup(
            params=self.encoded_products,
            ids=product_id,
        )
        customer_vec_cont = tf.nn.embedding_lookup(
            params=self.encoded_customers,
            ids=customer_id,
        )

        p_hid_cont = self.prod_fc1_cont(product_vec_cont)
        p_hid_cont = self.prod_bn_cont(p_hid_cont)
        p_hid_cont = self.prod_drop_cont(p_hid_cont)

        c_hid_cont = self.cust_fc1_cont(customer_vec_cont)
        c_hid_cont = self.cust_bn_cont(c_hid_cont)
        c_hid_cont = self.cust_drop_cont(c_hid_cont)

        views_hid_cont = self.views_hid_cont(views)
        views_hid_cont = self.views_bn_cont(c_hid_cont)
        views_hid_cont = self.views_drop_cont(c_hid_cont)

        concatted_cont = tf.concat([p_hid_cont, c_hid_cont, views_hid_cont], axis=-1)

        hid2_cont = self.fc2_cont(concatted_cont) # this is now size 1 output
        hid2_cont = self.fc2_bn_cont(hid2_cont)
        hid2_cont = self.fc2_drop_cont(hid2_cont) 

        ###
        concatted_all = tf.concat([cocluster_preds, hid2_cont], axis=-1)
        hid_f1 = self.fc1_tot(concatted_all)
        hid_f1 = self.fc1_bn_drop(hid_f1)
        hid_f1 = self.fc1_tot_drop(hid_f1)

        out = self.out(hid_f1)

        return out


import pandas as pd

import os, sys

CDIR = os.path.dirname(os.path.abspath(__file__))
PPDIR = os.path.dirname(os.path.dirname(CDIR))

sys.path.append(PPDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from models.content_based_filtering import product_vector_similarity, customer_vector_similarity
from misc import constants
from models import common_funcs
from processing import split_data, data_loading

################################################################################

PLOT_LR = False # set to True to plot lr graph


def lr_plotter(model, x_small, y_small, epochs=12):
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/2))
    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    history = model.fit(x=x_small, y=y_small,
                epochs=epochs,
                callbacks=[lr_schedule]
    )
    plt.semilogx(history.history['lr'], history.history['loss'])
    plt.savefig('lr_plot.png')
    plt.close()


def get_hybrid_nn_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    # combine coclustering (the best collaborative we have) with content and views nn

    #physical_devices = tf.config.list_physical_devices('GPU')[:1]
    #tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.config.experimental.set_memory_growth(physical_devices[1], True)

    
    row_lookup_customers, encoded_customers = customer_vector_similarity.encode_customer()
    row_lookup_products, encoded_products = product_vector_similarity.encode_products()

    views = data_loading.get_views_df()
    non_repeating_inds = views[['customerId', 'productId']].drop_duplicates().index # we have some duplciate customer-product ids, each pair only results in one purchase, so I think this is due to a purchase before the end of the month, then more views after? if so we don't care about that. Otherwise maybe its incorrect data
    views = views.loc[non_repeating_inds, :] # all pairs are now unique
    views = views[['customerId','productId', 'viewOnly','changeThumbnail','imageZoom','viewCatwalk','view360','sizeGuide']]
    std_scaler = StandardScaler()
    views[['viewOnly','changeThumbnail','imageZoom' , 'viewCatwalk','view360','sizeGuide']] = std_scaler.fit_transform(views[['viewOnly','changeThumbnail','imageZoom','viewCatwalk','view360','sizeGuide']].values)

    train_df = pd.merge(views, train_df, on=['customerId', 'productId'], how='inner') # merge views with train_df
    test_df = pd.merge(views, test_df, on=['customerId', 'productId'], how='inner') # views with test_df

    train_df = train_df[(train_df.customerId.isin(row_lookup_customers.keys())) & (train_df.productId.isin(row_lookup_products.keys()))]
    test_df = test_df[(test_df.customerId.isin(row_lookup_customers.keys())) & (test_df.productId.isin(row_lookup_products.keys()))]

    train_df.loc[:, "customerId"] = train_df.customerId.map(row_lookup_customers) # now the ids are the row indexes of the encoder matrix
    train_df.loc[:, "productId"] = train_df.productId.map(row_lookup_products)

    max_prod_ind = max(train_df.productId.max(), test_df.productId.max()) + 1
    max_cust_ind = max(train_df.customerId.max(), test_df.customerId.max()) + 1

    train_df = train_df.iloc[:int(len(train_df) * 0.8)]
    val_df = train_df.iloc[int(len(train_df) * 0.8):]

    test_df_mapped = test_df.copy() 
    test_df_mapped.loc[:, "customerId"] = test_df_mapped.customerId.map(row_lookup_customers) # now the ids are the row indexes of the encoder matrix #NOTE: missing customers will go!?
    test_df_mapped.loc[:, "productId"] = test_df_mapped.productId.map(row_lookup_products)

    _train, _val, test_set_df = split_data.get_split_labels_training_df()
    test_set_df = pd.merge(views, test_set_df, on=['customerId', 'productId'], how='inner') # views with test_set_df
    test_set_df = test_set_df[(test_set_df.customerId.isin(row_lookup_customers.keys())) & (test_set_df.productId.isin(row_lookup_products.keys()))]
    test_set_df_mapped = test_set_df.copy() 
    test_set_df_mapped.loc[:, "customerId"] = test_set_df_mapped.customerId.map(row_lookup_customers) # now the ids are the row indexes of the encoder matrix #NOTE: missing customers will go!?
    test_set_df_mapped.loc[:, "productId"] = test_set_df_mapped.productId.map(row_lookup_products)

    holdout_set = data_loading.get_labels_predict_df()
    holdout_set = pd.merge(views, holdout_set, on=['customerId', 'productId'], how='inner') # views with holdout_set
    holdout_set = holdout_set[(holdout_set.customerId.isin(row_lookup_customers.keys())) & (holdout_set.productId.isin(row_lookup_products.keys()))]
    holdout_set_df_mapped = holdout_set.copy() 
    holdout_set_df_mapped.loc[:, "customerId"] = holdout_set_df_mapped.customerId.map(row_lookup_customers) # now the ids are the row indexes of the encoder matrix #NOTE: missing customers will go!?
    holdout_set_df_mapped.loc[:, "productId"] = holdout_set_df_mapped.productId.map(row_lookup_products)
    holdout_set_df_mapped.loc[:, "purchase_probability"] = 0

    # first get the coclustering predictions
    try:
        with gzip.open(os.path.join(constants.MODEL_FILES_DIR, "cocluster_preds.gzip"), 'rb') as f:
            train_predictions, val_predictions, test_predictions, test_set_predictions, holdout_predictions = pickle.load(f)
    except FileNotFoundError:
        train_data = Dataset.load_from_df(train_df[['customerId', 'productId', 'purchased']], reader=Reader(rating_scale=(0,1)))
        val_data = Dataset.load_from_df(val_df[['customerId', 'productId', 'purchased']], reader=Reader(rating_scale=(0,1)))
        test_data = Dataset.load_from_df(test_df_mapped[['customerId', 'productId', 'purchased']], reader=Reader(rating_scale=(0,1)))

        test_set_data = Dataset.load_from_df(test_set_df_mapped[['customerId', 'productId', 'purchased']], reader=Reader(rating_scale=(0,1)))
        holdout_set_data = Dataset.load_from_df(holdout_set_df_mapped[['customerId', 'productId', 'purchase_probability']], reader=Reader(rating_scale=(0,1)))

        # fit model
        algo = CoClustering()
        algo.fit(train_data.build_full_trainset())

        # make predictions
        # train
        train_preds = algo.test(train_data.build_full_trainset().build_testset()) 
        train_cocluster_df = pd.DataFrame(train_preds)
        train_predictions = train_cocluster_df.est.values
        # val
        val_preds = algo.test(val_data.build_full_trainset().build_testset()) 
        val_cocluster_df = pd.DataFrame(val_preds)
        val_predictions = val_cocluster_df.est.values
        # test
        test_preds = algo.test(test_data.build_full_trainset().build_testset()) 
        test_cocluster_df = pd.DataFrame(test_preds)
        test_predictions = test_cocluster_df.est.values
        # test set
        test_set_preds = algo.test(test_set_data.build_full_trainset().build_testset()) 
        test_set_cocluster_df = pd.DataFrame(test_set_preds)
        test_set_predictions = test_set_cocluster_df.est.values
        # holdout set
        holdout_preds = algo.test(holdout_set_data.build_full_trainset().build_testset()) 
        holdout_cocluster_df = pd.DataFrame(holdout_preds)
        holdout_predictions = holdout_cocluster_df.est.values

        # these are the coclustering predictions
        with gzip.open(os.path.join(constants.MODEL_FILES_DIR, "cocluster_preds.gzip"), 'wb') as f:
            pickle.dump([train_predictions, val_predictions, test_predictions, test_set_predictions, holdout_predictions], f, protocol=4)

    ## next join these predictions onto the train_data
    train_df["cocluster_pred"] = train_predictions
    val_df["cocluster_pred"] = val_predictions
    test_df_mapped["cocluster_pred"] = test_predictions

    test_set_df_mapped["cocluster_pred"] = test_set_predictions
    holdout_set_df_mapped["cocluster_pred"] = holdout_predictions

    ###

    # NOTE: we have an imbalanced dataset, rather than use class-weights, I'll use oversampling, as this will be a smoother evolution (more positive samples in each batch rather than one heavily weighted sample)
    balance = False
    feature_cols = ["productId", "customerId", "viewOnly", "changeThumbnail", "imageZoom", "viewCatwalk", "view360", "sizeGuide", "cocluster_pred"]
    if balance:
        pos_train_dataset = tf.data.Dataset.from_tensor_slices((train_df[train_df.purchased][feature_cols].values, train_df[train_df.purchased].purchased.values))
        neg_train_dataset = tf.data.Dataset.from_tensor_slices((train_df[~train_df.purchased][feature_cols].values, train_df[~train_df.purchased].purchased.values))
        pos_val_dataset = tf.data.Dataset.from_tensor_slices((val_df[val_df.purchased][feature_cols].values, val_df[val_df.purchased].purchased.values))
        neg_val_dataset = tf.data.Dataset.from_tensor_slices((val_df[~val_df.purchased][feature_cols].values, val_df[~val_df.purchased].purchased.values))

        resampled_train_dataset = tf.data.experimental.sample_from_datasets([pos_train_dataset, neg_train_dataset], weights=[0.5, 0.5])
        resampled_val_dataset = tf.data.experimental.sample_from_datasets([pos_val_dataset, neg_val_dataset], weights=[0.5, 0.5])
    else:
        resampled_train_dataset = tf.data.Dataset.from_tensor_slices((train_df[feature_cols].values, train_df.purchased.values)) # TODO: make dataset for +ve and -ve, combine them with weightings
        resampled_val_dataset = tf.data.Dataset.from_tensor_slices((val_df[feature_cols].values, val_df.purchased.values))

    BATCH_SIZE = 50_000
    train_dataset = resampled_train_dataset.shuffle(len(train_df) // 3).batch(BATCH_SIZE).prefetch(2) # NOTE: didn't spend much time thinking about this, probably because of oversampling want this to be larger
    val_dataset = resampled_val_dataset.shuffle(len(val_df) // 3).batch(BATCH_SIZE).prefetch(2)

    # set the bias manually
    n_pos = len(train_df[train_df.purchased])
    n_neg = len(train_df[~train_df.purchased])
    b0 = np.log([n_pos/n_neg])

    model = simple_NN( # deep_NN
        highest_customer_ind=max_cust_ind, 
        highest_product_ind=max_prod_ind, 
        encoded_customers=encoded_customers.todense(), 
        encoded_products=encoded_products.todense(), 
        output_bias=b0
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max',
        patience=6,
        restore_best_weights=True,
    )
    

    if PLOT_LR:
        x_small = list(train_dataset.as_numpy_iterator())[0][0][:1000]
        y_small = list(train_dataset.as_numpy_iterator())[0][1][:1000]
        lr_plotter(model, x_small, y_small)
    learning_rate = 0.001 # NOTE: I found this from doing the lr_plotter above

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(), keras.metrics.BinaryAccuracy()])

    history = model.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=40,
        callbacks=[early_stopping],
        shuffle=True,
    )
    
    model_name = "nn_views_coclustering_simple_unbalanced"

    def plot_all(history, model_name):
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()
        plt.savefig(f"{model_name}_loss.png")
        plt.close()

        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='val')
        plt.legend()
        plt.savefig(f"{model_name}_acc.png")
        plt.close()

        plt.plot(history.history['auc'], label='train')
        plt.plot(history.history['val_auc'], label='val')
        plt.legend()
        plt.savefig(f"{model_name}_auc.png")
        plt.close()

        plt.plot(history.history['precision'], label='train')
        plt.plot(history.history['val_precision'], label='val')
        plt.legend()
        plt.savefig(f"{model_name}_pred.png")
        plt.close()

        plt.plot(history.history['recall'], label='train')
        plt.plot(history.history['val_recall'], label='val')
        plt.legend()
        plt.savefig(f"{model_name}_recall.png")
        plt.close()
    
    plot_all(history, model_name)

    # TODO fix this bit
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df_mapped[feature_cols].values, test_df_mapped.purchased.values))
    test_dataset = test_dataset.batch(BATCH_SIZE) # no shuffle!
    predictions = model.predict(test_dataset)
    test_df['purchased'] = predictions # NOTE: same name column as labels


    ###
    def final_bit(history, model_name, model, views, row_lookup_customers, row_lookup_products, feature_cols, BATCH_SIZE, plot_all, test_set_df_mapped, holdout_set_df_mapped):
        
        test_set_dataset = tf.data.Dataset.from_tensor_slices((test_set_df_mapped[feature_cols].values, test_set_df_mapped.purchased.values))
        test_set_dataset = test_set_dataset.batch(BATCH_SIZE) # no shuffle!
        predictions = model.predict(test_set_dataset)

        test_set_df_mapped['purchased'] = predictions
        with gzip.open('nnv2_test_set.gzip', 'wb') as f: pickle.dump(test_set_df_mapped, f, protocol=4)

        holdout_set_dataset = tf.data.Dataset.from_tensor_slices((holdout_set_df_mapped[feature_cols].values, holdout_set_df_mapped.purchase_probability.values))
        holdout_set_dataset = holdout_set_dataset.batch(BATCH_SIZE) # no shuffle!
        predictions = model.predict(holdout_set_dataset)

        holdout_set['purchase_probability'] = predictions
        with gzip.open('nnv2_holdout_set.gzip', 'wb') as f: pickle.dump(holdout_set, f, protocol=4)
        # NOTE missing about 4k preds

        plot_all(history, model_name)

    final_bit(history, model_name, model, views, row_lookup_customers, row_lookup_products, feature_cols, BATCH_SIZE, plot_all, test_set_df_mapped, holdout_set_df_mapped)

    return test_df


################################################################################

def main():
    model_name = "nn_views_coclustering_simple_unbalanced"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(model_name=model_name, model_fetching_func=get_hybrid_nn_probs, dataset_being_evaluated=dataset_being_evaluated)
    labels = common_funcs.get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores = common_funcs.get_scores(predictions, labels, model_name=model_name, dataset_being_evaluated=dataset_being_evaluated)
    
    if dataset_being_evaluated == "val":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.VAL_SCORES_DICT)
    elif dataset_being_evaluated == "test":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.TEST_SCORES_DICT)

################################################################################

if __name__ == "__main__":
    main()
