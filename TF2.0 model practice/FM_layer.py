import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd


class FM(Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FM, self).__init__()
        self.vector_dim = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name = 'w0', shape = (1,), initializer = tf.zeros_initializer(), trainable = True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1), initializer=tf.random_normal_initializer(), trainable=True, regularizer = tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.vector_dim), initializer=tf.random_normal_initializer(),
                                 trainable=True, regularizer = tf.keras.regularizers.l2(self.v_reg))



    def call(self, input):
        liner_part = tf.matmul(input, self.w) + self.w0
        inter_part1 = tf.pow(tf.matmul(input, self.v), 2)
        inter_part2 = tf.matmul(tf.pow(input, 2), tf.pow(self.v, 2))
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis = -1, keepdims=True)
        output = liner_part + inter_part

        return output

class model(Model):
    def __init__(self, k, w_reg, v_reg):
        super(model, self).__init__()
        self.fm = FM(k, w_reg, v_reg)

    def call(self, input):
        fm_out = self.fm(input)
        return tf.nn.sigmoid(fm_out)
