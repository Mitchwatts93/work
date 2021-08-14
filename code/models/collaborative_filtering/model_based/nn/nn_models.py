"""nn models"""
from typing import Optional
from tensorflow import keras
import tensorflow as tf

################################################################################

class simple_NN(keras.Model):

    def __init__(self, 
        highest_customer_ind: int, highest_product_ind: int, 
        output_bias: Optional[float] = None
    ):
        super().__init__()
        self.drop_rate = 0.5
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        
        embedding_dim = 20
        
        self.prod_emb = keras.layers.Embedding(highest_product_ind, embedding_dim)

        self.prod_fc1 = keras.layers.Dense(units=10, activation='relu')
        self.prod_bn1 = keras.layers.BatchNormalization()
        self.prod_drop1 = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_emb = keras.layers.Embedding(highest_customer_ind, embedding_dim)

        self.cust_fc1 = keras.layers.Dense(units=10, activation='relu')
        self.cust_bn1 = keras.layers.BatchNormalization()
        self.cust_drop1 = keras.layers.Dropout(rate=self.drop_rate)

        self.fc1 = keras.layers.Dense(units=20, activation='relu')
        self.fc1_bn = keras.layers.BatchNormalization()
        self.fc1_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.out = keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)


    def call(self, X: tf.Tensor) -> tf.Tensor:

        product_id = X[:, 0]

        product_vec = self.prod_emb(product_id)

        customer_id = X[:, 1]
        customer_vec = self.cust_emb(customer_id)

        p_hid1 = self.prod_fc1(product_vec)
        p_hid1 = self.prod_bn1(p_hid1)
        p_hid1 = self.prod_drop1(p_hid1)

        c_hid1 = self.cust_fc1(customer_vec)
        c_hid1 = self.cust_bn1(c_hid1)
        c_hid1 = self.cust_drop1(c_hid1)

        concatted = tf.concat([p_hid1, c_hid1], axis=-1)

        hid1 = self.fc1(concatted)
        hid1 = self.fc1_bn(hid1)
        hid1 = self.fc1_drop(hid1)


        out = self.out(hid1)

        return out


class deep_NN(keras.Model):

    def __init__(self, 
        highest_customer_ind: int, highest_product_ind: int, 
        output_bias: Optional[float] = None
    ) -> None:
        super().__init__()
        self.drop_rate = 0.1
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        
        embedding_dim = 50
        
        self.prod_emb = keras.layers.Embedding(highest_product_ind, embedding_dim)

        self.prod_fc1 = keras.layers.Dense(units=20, activation='relu')
        self.prod_drop1 = keras.layers.Dropout(rate=self.drop_rate)
        self.prod_fc2 = keras.layers.Dense(units=10, activation='relu')
        self.prod_drop2 = keras.layers.Dropout(rate=self.drop_rate)

        self.cust_emb = keras.layers.Embedding(highest_customer_ind, embedding_dim)

        self.cust_fc1 = keras.layers.Dense(units=20, activation='relu')
        self.cust_drop1 = keras.layers.Dropout(rate=self.drop_rate)
        self.cust_fc2 = keras.layers.Dense(units=10, activation='relu')
        self.cust_drop2 = keras.layers.Dropout(rate=self.drop_rate)

        self.fc1 = keras.layers.Dense(units=20, activation='relu')
        self.fc1_drop = keras.layers.Dropout(rate=self.drop_rate)
        self.fc2 = keras.layers.Dense(units=10, activation='relu')
        self.fc2_drop = keras.layers.Dropout(rate=self.drop_rate)

        self.out = keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)


    def call(self, X: tf.Tensor) -> tf.Tensor:

        product_id = X[:, 0]

        product_vec = self.prod_emb(product_id)

        customer_id = X[:, 1]
        customer_vec = self.cust_emb(customer_id)

        p_hid1 = self.prod_fc1(product_vec)
        p_hid1 = self.prod_drop1(p_hid1)
        p_hid2 = self.prod_fc2(p_hid1)
        p_hid2 = self.prod_drop2(p_hid2)

        c_hid1 = self.cust_fc1(customer_vec)
        c_hid1 = self.cust_drop1(c_hid1)
        c_hid2 = self.cust_fc2(c_hid1)
        c_hid2 = self.cust_drop2(c_hid2)

        concatted = tf.concat([p_hid2, c_hid2], axis=-1)

        hid1 = self.fc1(concatted)
        hid1 = self.fc1_drop(hid1)
        hid2 = self.fc2(hid1)
        hid2 = self.fc2_drop(hid2)
        out = self.out(hid2)

        return out
