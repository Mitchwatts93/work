from tensorflow import keras
import tensorflow as tf
import numpy as np
from typing import Optional

################################################################################
# without views

class simple_NN_no_views(keras.Model):

    def __init__(self, highest_customer_ind: int, highest_product_ind: int, encoded_customers: np.ndarray, encoded_products: np.ndarray, output_bias: Optional[float] = None) -> None:
        super().__init__()
        self.drop_rate = 0.01
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)

        self.encoded_customers = tf.convert_to_tensor(encoded_customers) #encoded_customers
        self.encoded_products = tf.convert_to_tensor(encoded_products)  #encoded_products

        self.prod_fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn_coll = keras.layers.BatchNormalization()
        self.prod_drop_coll = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn_coll = keras.layers.BatchNormalization()
        self.cust_drop_coll = keras.layers.Dropout(rate=self.drop_rate)

        self.fc2_coll = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_coll = keras.layers.BatchNormalization()
        self.fc2_drop_coll = keras.layers.Dropout(rate=self.drop_rate)


        ####
        embedding_dim = 20
        
        self.prod_emb = keras.layers.Embedding(highest_product_ind, embedding_dim)
        self.cust_emb = keras.layers.Embedding(highest_customer_ind, embedding_dim)

        self.prod_fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn_cont = keras.layers.BatchNormalization()
        self.prod_drop_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn_cont = keras.layers.BatchNormalization()
        self.cust_drop_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.fc2_cont = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_cont = keras.layers.BatchNormalization()
        self.fc2_drop_cont= keras.layers.Dropout(rate=self.drop_rate)

        ###

        self.fc1_tot = keras.layers.Dense(units=10, activation='relu')
        self.fc1_bn_drop = keras.layers.BatchNormalization()
        self.fc1_tot_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.out = keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)


    def call(self, X: tf.Tensor) -> tf.Tensor:
        product_id = X[:, 0]
        customer_id = X[:, 1]

        ### collaborative
        
        product_vec_coll = self.prod_emb(product_id)
        customer_vec_coll = self.cust_emb(customer_id)
        
        p_hid_coll = self.prod_fc1_coll(product_vec_coll)
        p_hid_coll = self.prod_bn_coll(p_hid_coll)
        p_hid_coll = self.prod_drop_coll(p_hid_coll)
        c_hid_coll = self.cust_fc1_coll(customer_vec_coll)
        c_hid_coll = self.cust_bn_coll(c_hid_coll)
        c_hid_coll = self.cust_drop_coll(c_hid_coll)

        concatted_coll = tf.concat([p_hid_coll, c_hid_coll], axis=-1)

        hid2_coll = self.fc2_coll(concatted_coll)
        hid2_coll = self.fc2_bn_coll(hid2_coll)
        hid2_coll = self.fc2_drop_coll(hid2_coll)


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

        concatted_cont = tf.concat([p_hid_cont, c_hid_cont], axis=-1)

        hid2_cont = self.fc2_cont(concatted_cont)
        hid2_cont = self.fc2_bn_cont(hid2_cont)
        hid2_cont = self.fc2_drop_cont(hid2_cont)


        ### concat both collab and content features
        concatted_all = tf.concat([hid2_coll, hid2_cont], axis=-1)
        hid_f1 = self.fc1_tot(concatted_all)
        hid_f1 = self.fc1_bn_drop(hid_f1)
        hid_f1 = self.fc1_tot_drop(hid_f1)
        out = self.out(hid_f1)

        return out


class deep_NN_no_views(keras.Model):

    def __init__(self, highest_customer_ind: int, highest_product_ind: int, encoded_customers: np.ndarray, encoded_products: np.ndarray, output_bias: Optional[float] = None) -> None:
        super().__init__()
        self.drop_rate = 0.01
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)

        self.encoded_customers = tf.convert_to_tensor(encoded_customers) #encoded_customers
        self.encoded_products = tf.convert_to_tensor(encoded_products)  #encoded_products

        self.prod_fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn1_coll = keras.layers.BatchNormalization()
        self.prod_drop1_coll = keras.layers.Dropout(rate=self.drop_rate)
        self.prod_fc2_coll = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn2_coll = keras.layers.BatchNormalization()
        self.prod_drop2_coll = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn1_coll = keras.layers.BatchNormalization()
        self.cust_drop1_coll = keras.layers.Dropout(rate=self.drop_rate)
        self.cust_fc2_coll = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn2_coll = keras.layers.BatchNormalization()
        self.cust_drop2_coll = keras.layers.Dropout(rate=self.drop_rate)

        self.fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.fc1_bn_coll = keras.layers.BatchNormalization()
        self.fc1_drop_coll = keras.layers.Dropout(rate=self.drop_rate)
        self.fc2_coll = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_coll = keras.layers.BatchNormalization()
        self.fc2_drop_coll = keras.layers.Dropout(rate=self.drop_rate)


        ####
        embedding_dim = 50
        
        self.prod_emb = keras.layers.Embedding(highest_product_ind, embedding_dim)
        self.cust_emb = keras.layers.Embedding(highest_customer_ind, embedding_dim)

        self.prod_fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn1_cont = keras.layers.BatchNormalization()
        self.prod_drop1_cont = keras.layers.Dropout(rate=self.drop_rate)
        self.prod_fc2_cont = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn2_cont = keras.layers.BatchNormalization()
        self.prod_drop2_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn1_cont = keras.layers.BatchNormalization()
        self.cust_drop1_cont = keras.layers.Dropout(rate=self.drop_rate)
        self.cust_fc2_cont = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn2_cont = keras.layers.BatchNormalization()
        self.cust_drop2_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.fc1_bn_cont = keras.layers.BatchNormalization()
        self.fc1_drop_cont= keras.layers.Dropout(rate=self.drop_rate)
        self.fc2_cont = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_cont = keras.layers.BatchNormalization()
        self.fc2_drop_cont= keras.layers.Dropout(rate=self.drop_rate)

        ###

        self.fc1_tot = keras.layers.Dense(units=10, activation='relu')
        self.fc1_bn_drop = keras.layers.BatchNormalization()
        self.fc1_tot_drop = keras.layers.Dropout(rate=self.drop_rate)
        self.fc2_tot = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_drop = keras.layers.BatchNormalization()
        self.fc2_tot_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.out = keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)


    def call(self, X: tf.Tensor) -> tf.Tensor:
        product_id = X[:, 0]
        customer_id = X[:, 1]

        ### collaborative
        product_vec_coll = self.prod_emb(product_id)
        customer_vec_coll = self.cust_emb(customer_id)
        
        p_hid_coll1 = self.prod_fc1_coll(product_vec_coll)
        p_hid_coll1 = self.prod_bn1_coll(p_hid_coll1)
        p_hid_coll1 = self.prod_drop1_coll(p_hid_coll1)
        p_hid_coll2 = self.prod_fc2_coll(p_hid_coll1)
        p_hid_coll2 = self.prod_bn2_coll(p_hid_coll2)
        p_hid_coll2 = self.prod_drop2_coll(p_hid_coll2)

        c_hid_coll1 = self.cust_fc1_coll(customer_vec_coll)
        c_hid_coll1 = self.cust_bn1_coll(c_hid_coll1)
        c_hid_coll1 = self.cust_drop1_coll(c_hid_coll1)
        c_hid_coll2 = self.cust_fc2_coll(c_hid_coll1)
        c_hid_coll2 = self.cust_bn2_coll(c_hid_coll2)
        c_hid_coll2 = self.cust_drop2_coll(c_hid_coll2)

        concatted_coll = tf.concat([p_hid_coll2, c_hid_coll2], axis=-1)

        hid2_coll1 = self.fc1_coll(concatted_coll)
        hid2_coll1 = self.fc1_bn_coll(hid2_coll1)
        hid2_coll1 = self.fc1_drop_coll(hid2_coll1)
        hid2_coll2 = self.fc2_coll(hid2_coll1)
        hid2_coll2 = self.fc2_bn_coll(hid2_coll2)
        hid2_coll2 = self.fc2_drop_coll(hid2_coll2)

        ### content

        product_vec_cont = tf.nn.embedding_lookup(
            params=self.encoded_products,
            ids=product_id,
        )
        customer_vec_cont = tf.nn.embedding_lookup(
            params=self.encoded_customers,
            ids=customer_id,
        )

        p_hid_cont1 = self.prod_fc1_cont(product_vec_cont)
        p_hid_cont1 = self.prod_bn1_cont(p_hid_cont1)
        p_hid_cont1 = self.prod_drop1_cont(p_hid_cont1)
        p_hid_cont2 = self.prod_fc2_cont(p_hid_cont1)
        p_hid_cont2 = self.prod_bn2_cont(p_hid_cont2)
        p_hid_cont2 = self.prod_drop2_cont(p_hid_cont2)

        c_hid_cont1 = self.cust_fc1_cont(customer_vec_cont)
        c_hid_cont1 = self.cust_bn1_cont(c_hid_cont1)
        c_hid_cont1 = self.cust_drop1_cont(c_hid_cont1)
        c_hid_cont2 = self.cust_fc2_cont(c_hid_cont1)
        c_hid_cont2 = self.cust_bn2_cont(c_hid_cont2)
        c_hid_cont2 = self.cust_drop2_cont(c_hid_cont2)

        concatted_cont = tf.concat([p_hid_cont2, c_hid_cont2], axis=-1)

        hid2_cont1 = self.fc1_cont(concatted_cont)
        hid2_cont1 = self.fc1_bn_cont(hid2_cont1)
        hid2_cont1 = self.fc1_drop_cont(hid2_cont1)
        hid2_cont2 = self.fc2_cont(hid2_cont1)
        hid2_cont2 = self.fc2_bn_cont(hid2_cont2)
        hid2_cont2 = self.fc2_drop_cont(hid2_cont2)

        ###

        concatted_all = tf.concat([hid2_cont2, hid2_coll2], axis=-1)

        hid_f1 = self.fc1_tot(concatted_all)
        hid_f1 = self.fc1_bn_drop(hid_f1)
        hid_f1 = self.fc1_tot_drop(hid_f1)
        hid_f2 = self.fc2_tot(hid_f1)
        hid_f2 = self.fc2_bn_drop(hid_f2)
        hid_f2 = self.fc2_tot_drop(hid_f2)

        out = self.out(hid_f2)

        return out

################################################################################
# with views

class simple_NN_views(keras.Model):

    def __init__(self, highest_customer_ind: int, highest_product_ind: int, encoded_customers: np.ndarray, encoded_products: np.ndarray, output_bias: Optional[float] = None) -> None:
        super().__init__()
        self.drop_rate = 0.01
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)

        # coll

        embedding_dim = 20
        
        self.prod_emb = keras.layers.Embedding(highest_product_ind, embedding_dim)
        self.cust_emb = keras.layers.Embedding(highest_customer_ind, embedding_dim)

        self.prod_fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn_coll = keras.layers.BatchNormalization()
        self.prod_drop_coll = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn_coll = keras.layers.BatchNormalization()
        self.cust_drop_coll = keras.layers.Dropout(rate=self.drop_rate)

        self.fc2_coll = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_coll = keras.layers.BatchNormalization()
        self.fc2_drop_coll = keras.layers.Dropout(rate=self.drop_rate)


        #### cont
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

        self.fc2_cont = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_cont = keras.layers.BatchNormalization()
        self.fc2_drop_cont= keras.layers.Dropout(rate=self.drop_rate)

        ###

        self.fc1_tot = keras.layers.Dense(units=10, activation='relu')
        self.fc1_bn_drop = keras.layers.BatchNormalization()
        self.fc1_tot_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.out = keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)


    def call(self, X: tf.Tensor) -> tf.Tensor:
        product_id = tf.cast(X[:, 0], tf.int32)
        customer_id =tf.cast( X[:, 1], tf.int32)
        views = X[:, 2:]

        ### collaborative
        
        product_vec_coll = self.prod_emb(product_id)
        customer_vec_coll = self.cust_emb(customer_id)
        
        p_hid_coll = self.prod_fc1_coll(product_vec_coll)
        p_hid_coll = self.prod_bn_coll(p_hid_coll)
        p_hid_coll = self.prod_drop_coll(p_hid_coll)
        c_hid_coll = self.cust_fc1_coll(customer_vec_coll)
        c_hid_coll = self.cust_bn_coll(c_hid_coll)
        c_hid_coll = self.cust_drop_coll(c_hid_coll)

        concatted_coll = tf.concat([p_hid_coll, c_hid_coll], axis=-1)

        hid2_coll = self.fc2_coll(concatted_coll)
        hid2_coll = self.fc2_bn_coll(hid2_coll)
        hid2_coll = self.fc2_drop_coll(hid2_coll)


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

        hid2_cont = self.fc2_cont(concatted_cont)
        hid2_cont = self.fc2_bn_cont(hid2_cont)
        hid2_cont = self.fc2_drop_cont(hid2_cont)


        ###

        concatted_all = tf.concat([hid2_coll, hid2_cont], axis=-1)
        hid_f1 = self.fc1_tot(concatted_all)
        hid_f1 = self.fc1_bn_drop(hid_f1)
        hid_f1 = self.fc1_tot_drop(hid_f1)
        out = self.out(hid_f1)

        return out


class deep_NN_views(keras.Model):

    def __init__(self, highest_customer_ind: int, highest_product_ind: int, encoded_customers: np.ndarray, encoded_products: np.ndarray, output_bias: Optional[float] = None) -> None:
        super().__init__()
        self.drop_rate = 0.01
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)

        self.encoded_customers = tf.convert_to_tensor(encoded_customers) #encoded_customers
        self.encoded_products = tf.convert_to_tensor(encoded_products)  #encoded_products

        self.prod_fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn1_coll = keras.layers.BatchNormalization()
        self.prod_drop1_coll = keras.layers.Dropout(rate=self.drop_rate)
        self.prod_fc2_coll = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn2_coll = keras.layers.BatchNormalization()
        self.prod_drop2_coll = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn1_coll = keras.layers.BatchNormalization()
        self.cust_drop1_coll = keras.layers.Dropout(rate=self.drop_rate)
        self.cust_fc2_coll = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn2_coll = keras.layers.BatchNormalization()
        self.cust_drop2_coll = keras.layers.Dropout(rate=self.drop_rate)

        self.fc1_coll = keras.layers.Dense(units=10, activation='relu')
        self.fc1_bn_coll = keras.layers.BatchNormalization()
        self.fc1_drop_coll = keras.layers.Dropout(rate=self.drop_rate)
        self.fc2_coll = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_coll = keras.layers.BatchNormalization()
        self.fc2_drop_coll = keras.layers.Dropout(rate=self.drop_rate)


        ####
        embedding_dim = 50
        
        self.prod_emb = keras.layers.Embedding(highest_product_ind, embedding_dim)
        self.cust_emb = keras.layers.Embedding(highest_customer_ind, embedding_dim)

        self.prod_fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn1_cont = keras.layers.BatchNormalization()
        self.prod_drop1_cont = keras.layers.Dropout(rate=self.drop_rate)
        self.prod_fc2_cont = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn2_cont = keras.layers.BatchNormalization()
        self.prod_drop2_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn1_cont = keras.layers.BatchNormalization()
        self.cust_drop1_cont = keras.layers.Dropout(rate=self.drop_rate)
        self.cust_fc2_cont = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn2_cont = keras.layers.BatchNormalization()
        self.cust_drop2_cont = keras.layers.Dropout(rate=self.drop_rate)

        self.fc1_cont = keras.layers.Dense(units=10, activation='relu')
        self.fc1_bn_cont = keras.layers.BatchNormalization()
        self.fc1_drop_cont= keras.layers.Dropout(rate=self.drop_rate)
        self.fc2_cont = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_cont = keras.layers.BatchNormalization()
        self.fc2_drop_cont= keras.layers.Dropout(rate=self.drop_rate)

        ###

        self.fc1_tot = keras.layers.Dense(units=10, activation='relu')
        self.fc1_bn_drop = keras.layers.BatchNormalization()
        self.fc1_tot_drop = keras.layers.Dropout(rate=self.drop_rate)
        self.fc2_tot = keras.layers.Dense(units=10, activation='relu')
        self.fc2_bn_drop = keras.layers.BatchNormalization()
        self.fc2_tot_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.out = keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)


    def call(self, X: tf.Tensor) -> tf.Tensor:
        product_id = tf.cast(X[:, 0], tf.int32)
        customer_id = tf.cast(X[:, 1], tf.int32)

        ### collaborative
        product_vec_coll = self.prod_emb(product_id)
        customer_vec_coll = self.cust_emb(customer_id)
        
        p_hid_coll1 = self.prod_fc1_coll(product_vec_coll)
        p_hid_coll1 = self.prod_bn1_coll(p_hid_coll1)
        p_hid_coll1 = self.prod_drop1_coll(p_hid_coll1)
        p_hid_coll2 = self.prod_fc2_coll(p_hid_coll1)
        p_hid_coll2 = self.prod_bn2_coll(p_hid_coll2)
        p_hid_coll2 = self.prod_drop2_coll(p_hid_coll2)

        c_hid_coll1 = self.cust_fc1_coll(customer_vec_coll)
        c_hid_coll1 = self.cust_bn1_coll(c_hid_coll1)
        c_hid_coll1 = self.cust_drop1_coll(c_hid_coll1)
        c_hid_coll2 = self.cust_fc2_coll(c_hid_coll1)
        c_hid_coll2 = self.cust_bn2_coll(c_hid_coll2)
        c_hid_coll2 = self.cust_drop2_coll(c_hid_coll2)

        concatted_coll = tf.concat([p_hid_coll2, c_hid_coll2], axis=-1)

        hid2_coll1 = self.fc1_coll(concatted_coll)
        hid2_coll1 = self.fc1_bn_coll(hid2_coll1)
        hid2_coll1 = self.fc1_drop_coll(hid2_coll1)
        hid2_coll2 = self.fc2_coll(hid2_coll1)
        hid2_coll2 = self.fc2_bn_coll(hid2_coll2)
        hid2_coll2 = self.fc2_drop_coll(hid2_coll2)

        ### content

        product_vec_cont = tf.nn.embedding_lookup(
            params=self.encoded_products,
            ids=product_id,
        )
        customer_vec_cont = tf.nn.embedding_lookup(
            params=self.encoded_customers,
            ids=customer_id,
        )

        p_hid_cont1 = self.prod_fc1_cont(product_vec_cont)
        p_hid_cont1 = self.prod_bn1_cont(p_hid_cont1)
        p_hid_cont1 = self.prod_drop1_cont(p_hid_cont1)
        p_hid_cont2 = self.prod_fc2_cont(p_hid_cont1)
        p_hid_cont2 = self.prod_bn2_cont(p_hid_cont2)
        p_hid_cont2 = self.prod_drop2_cont(p_hid_cont2)

        c_hid_cont1 = self.cust_fc1_cont(customer_vec_cont)
        c_hid_cont1 = self.cust_bn1_cont(c_hid_cont1)
        c_hid_cont1 = self.cust_drop1_cont(c_hid_cont1)
        c_hid_cont2 = self.cust_fc2_cont(c_hid_cont1)
        c_hid_cont2 = self.cust_bn2_cont(c_hid_cont2)
        c_hid_cont2 = self.cust_drop2_cont(c_hid_cont2)

        concatted_cont = tf.concat([p_hid_cont2, c_hid_cont2], axis=-1)

        hid2_cont1 = self.fc1_cont(concatted_cont)
        hid2_cont1 = self.fc1_bn_cont(hid2_cont1)
        hid2_cont1 = self.fc1_drop_cont(hid2_cont1)
        hid2_cont2 = self.fc2_cont(hid2_cont1)
        hid2_cont2 = self.fc2_bn_cont(hid2_cont2)
        hid2_cont2 = self.fc2_drop_cont(hid2_cont2)

        ###

        concatted_all = tf.concat([hid2_cont2, hid2_coll2], axis=-1)

        hid_f1 = self.fc1_tot(concatted_all)
        hid_f1 = self.fc1_bn_drop(hid_f1)
        hid_f1 = self.fc1_tot_drop(hid_f1)
        hid_f2 = self.fc2_tot(hid_f1)
        hid_f2 = self.fc2_bn_drop(hid_f2)
        hid_f2 = self.fc2_tot_drop(hid_f2)

        out = self.out(hid_f2)

        return out

################################################################################
# using views, and coclustering predictions directly for collaborative part

class simple_NN_views_coclustering(keras.Model):

    def __init__(self, encoded_customers: np.ndarray, encoded_products: np.ndarray, output_bias: Optional[float] = None) -> None:
        super().__init__()
        self.drop_rate = 0.01
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)

        self.encoded_customers = tf.convert_to_tensor(encoded_customers)
        self.encoded_products = tf.convert_to_tensor(encoded_products) 

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


    def call(self, X: tf.Tensor) -> tf.Tensor:
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

################################################################################
