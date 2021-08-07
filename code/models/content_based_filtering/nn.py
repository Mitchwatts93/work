from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class simple_NN(keras.Model):

    def __init__(self, encoded_customers, encoded_products):
        super().__init__()
        self.drop_rate = 0.1

        self.encoded_customers = tf.convert_to_tensor(encoded_customers) #encoded_customers
        self.encoded_products = tf.convert_to_tensor(encoded_products)  #encoded_products

        self.prod_fc1 = keras.layers.Dense(units=10, activation='relu')
        self.prod_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1 = keras.layers.Dense(units=10, activation='relu')
        self.cust_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.fc2 = keras.layers.Dense(units=10, activation='relu')
        self.fc2_drop = keras.layers.Dropout(rate=self.drop_rate)
        self.out = keras.layers.Dense(units=1, activation='sigmoid')


    def call(self, X):

        product_id = X[:, 0]

        product_vec = tf.nn.embedding_lookup(
            params=self.encoded_products,
            ids=product_id,
        )

        customer_id = X[:, 1]
        customer_vec = tf.nn.embedding_lookup(
            params=self.encoded_customers,
            ids=customer_id,
        )

        p_hid = self.prod_fc1(product_vec)
        p_hid = self.prod_drop(p_hid)
        c_hid = self.cust_fc1(customer_vec)
        c_hid = self.cust_drop(c_hid)

        concatted = tf.concat([p_hid, c_hid], axis=-1)

        hid2 = self.fc2(concatted)
        hid2 = self.fc2_drop(hid2)
        out = self.out(hid2)

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


def get_content_nn_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    row_lookup_customers, encoded_customers = customer_vector_similarity.encode_customer()
    row_lookup_products, encoded_products = product_vector_similarity.encode_products()

    train_df = train_df[(train_df.customerId.isin(row_lookup_customers.keys())) & (train_df.productId.isin(row_lookup_products.keys()))]
    test_df = test_df[(test_df.customerId.isin(row_lookup_customers.keys())) & (test_df.productId.isin(row_lookup_products.keys()))]
    
    train_df.loc[:, "customerId"] = train_df.customerId.map(row_lookup_customers) # now the ids are the row indexes of the encoder matrix
    train_df.loc[:, "productId"] = train_df.productId.map(row_lookup_products)

    train_df = train_df.iloc[:int(len(train_df) * 0.8)]
    val_df = train_df.iloc[int(len(train_df) * 0.8):]

    # bool_train_labels = train_df.purchased

    # NOTE: we have an imbalanced dataset, rather than use class-weights, I'll use oversampling, as this will be a smoother evolution (more positive samples in each batch rather than one heavily weighted sample)
    pos_train_dataset = tf.data.Dataset.from_tensor_slices((train_df[train_df.purchased][["productId", "customerId"]].values, train_df[train_df.purchased].purchased.values))
    neg_train_dataset = tf.data.Dataset.from_tensor_slices((train_df[~train_df.purchased][["productId", "customerId"]].values, train_df[~train_df.purchased].purchased.values))
    #train_dataset = tf.data.Dataset.from_tensor_slices((train_df[["productId", "customerId"]].values, train_df.purchased.values)) # TODO: make dataset for +ve and -ve, combine them with weightings
    pos_val_dataset = tf.data.Dataset.from_tensor_slices((val_df[val_df.purchased][["productId", "customerId"]].values, val_df[val_df.purchased].purchased.values))
    neg_val_dataset = tf.data.Dataset.from_tensor_slices((val_df[~val_df.purchased][["productId", "customerId"]].values, val_df[~val_df.purchased].purchased.values))
    #val_dataset = tf.data.Dataset.from_tensor_slices((val_df[["productId", "customerId"]].values, val_df.purchased.values))

    resampled_train_dataset = tf.data.experimental.sample_from_datasets([pos_train_dataset, neg_train_dataset], weights=[0.5, 0.5])
    resampled_val_dataset = tf.data.experimental.sample_from_datasets([pos_val_dataset, neg_val_dataset], weights=[0.5, 0.5])

    BATCH_SIZE = 100_000
    train_dataset = resampled_train_dataset.shuffle(len(train_df) // 10).batch(BATCH_SIZE).prefetch(2) # NOTE: didn't spend much time thinking about this, probably because of oversampling want this to be larger
    val_dataset = resampled_val_dataset.shuffle(len(val_df) // 10).batch(BATCH_SIZE).prefetch(2)

    #model = simple_NN(np.array(list(row_lookup_customers.items())), encoded_customers.todense(), np.array(list(row_lookup_products.items())), encoded_products.todense())
    model = simple_NN(encoded_customers.todense(), encoded_products.todense())
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
    

    if PLOT_LR:
        x_small = list(train_dataset.as_numpy_iterator())[0][0]
        y_small = list(train_dataset.as_numpy_iterator())[0][1]
        lr_plotter(model, x_small, y_small)
    learning_rate = 0.03 # NOTE: I found this from doing the lr_plotter above

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

    history = model.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=4,
        callbacks=[early_stopping],
        shuffle=True,
    )
    
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig("nn_loss.png")
    plt.close()

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='train')
    plt.legend()
    plt.savefig("nn_acc.png")

    breakpoint()
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df[["productId", "customerId"]].values, test_df.purchased.values))
    test_dataset = test_dataset.batch(BATCH_SIZE) # no shuffle!
    predictions = model.predict(test_dataset)
    import ipdb;ipdb.set_trace()
    test_df['purchased'] = predictions # NOTE: same name column as labels
    return test_df


################################################################################

def main():
    model_name = "content_nn"
    dataset_being_evaluated = "val"

    predictions = common_funcs.generate_and_cache_preds(model_name=model_name, model_fetching_func=get_content_nn_probs, dataset_being_evaluated=dataset_being_evaluated)
    labels = common_funcs.get_labels(dataset_to_fetch=dataset_being_evaluated)
    scores = common_funcs.get_scores(predictions, labels, model_name=model_name, dataset_being_evaluated=dataset_being_evaluated)
    
    if dataset_being_evaluated == "val":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.VAL_SCORES_DICT)
    elif dataset_being_evaluated == "test":
        common_funcs.add_scores_to_master_dict(scores, model_name=model_name, model_dict_path=constants.TEST_SCORES_DICT)

################################################################################

if __name__ == "__main__":
    main()
