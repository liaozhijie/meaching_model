import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd


class Liner(Layer):
    def __init__(self):
        super(Liner, self).__init__()
        self.out_layer = Dense(1, activation = None, use_bias=True)

    def call(self, input):
        out = self.out_layer(input)
        return out

class model_L(Model):
    def __init__(self, activation = 'relu'):
        super(model_L, self).__init__()
        self.liner = Liner()

    def call(self, input):
        liner_out = self.liner(input)
        return tf.nn.sigmoid(liner_out)
