from tensorflow import keras
import tensorflow as tf
import numpy as np


class simple_NN(keras.Model):

    def __init__(self, encoded_customers, encoded_products):#row_lookup_customers, encoded_customers, row_lookup_products, encoded_products) -> None:
        super().__init__()
        #self.row_lookup_customers = row_lookup_customers
        self.encoded_customers = encoded_customers
        #self.row_lookup_products = row_lookup_products
        self.encoded_products = encoded_products

        #self.row_lookup_customers = tf.nn.embedding_lookup(
        #    params=encoded_customers,
        #    ids=np.array(list(row_lookup_customers.keys())),
        #)

        #self.row_lookup_products = tf.nn.embedding_lookup(
        #    params=encoded_products,
        #    ids=np.array(list(row_lookup_products.keys())),
        #)

        self.prod_fc1 = keras.layers.Dense(units=10, activation='relu')

        self.cust_fc1 = keras.layers.Dense(units=10, activation='relu')

        self.fc2 = keras.layers.Dense(units=10, activation='relu')
        self.out = keras.layers.Dense(units=1, activation='sigmoid')


    def call(self, X):

        product_id = X[:, 0]
        #product_vec = self.row_lookup_products.lookup(product_id)

        #product_inds = tf.nn.embedding_lookup(
        #    params=self.row_lookup_customers,
        #    ids=product_id,
        #)
        product_vec = tf.nn.embedding_lookup(
            params=self.encoded_customers,
            ids=product_id,
        )

        
        customer_id = X[:, 1]
        #customer_vec = self.row_lookup_customers.lookup(customer_id)
        customer_vec = tf.nn.embedding_lookup(
            params=self.encoded_products,
            ids=customer_id,
        )


        breakpoint()
        p_hid = self.prod_fc1(product_vec)
        c_hid = self.cust_fc1(customer_vec)

        concatted = tf.concat([p_hid, c_hid])

        hid2 = self.fc2(concatted)
        out = self.out(hid2)

        return out


import numpy as np
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


def get_content_nn_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    row_lookup_customers, encoded_customers = customer_vector_similarity.encode_customer()
    row_lookup_products, encoded_products = product_vector_similarity.encode_products()

    train_df = train_df[(train_df.customerId.isin(row_lookup_customers.keys())) & (train_df.productId.isin(row_lookup_products.keys()))]
    test_df = test_df[(test_df.customerId.isin(row_lookup_customers.keys())) & (test_df.productId.isin(row_lookup_products.keys()))]
    
    breakpoint()
    train_df.loc[:, "customerId"] = train_df.customerId.map(row_lookup_customers)
    train_df.loc[:, "productId"] = train_df.productId.map(row_lookup_products)

    train_df = train_df.iloc[:int(len(train_df) * 0.8)]
    val_df = train_df.iloc[int(len(train_df) * 0.8):]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df[["productId", "customerId"]].values, train_df.purchased.values)) # TODO: make dataset for +ve and -ve, combine them with weightings
    val_dataset = tf.data.Dataset.from_tensor_slices((val_df[["productId", "customerId"]].values, val_df.purchased.values))

    BATCH_SIZE = 128 # TODO make larger
    train_dataset = train_dataset.shuffle(len(train_df) * 2).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(len(val_df) * 2).batch(BATCH_SIZE)

    breakpoint()
    #model = simple_NN(np.array(list(row_lookup_customers.items())), encoded_customers.todense(), np.array(list(row_lookup_products.items())), encoded_products.todense())
    model = simple_NN(encoded_customers.todense(), encoded_products.todense())
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
    learning_rate = 0.001 # TODO
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    _history = model.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=100,
        callbacks=[early_stopping],
        shuffle=True,
    )
    breakpoint()

    test_dataset = tf.data.Dataset.from_tensor_slices((train_df[["productId", "customerId"]].values, train_df.purchased.values))
    test_dataset = test_dataset.shuffle(len(val_df) * 2).batch(BATCH_SIZE)
    predictions = model.predict(test_dataset)

    breakpoint()
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
