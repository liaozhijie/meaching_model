import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Concatenate, Flatten

class embedding_layer(Layer):
    def __init__(self, emb_type, feature_info):
        super(embedding_layer, self).__init__()
        self.emb_layer = Embedding(feature_info.vocabulary_size, feature_info.embedding_dim, emb_type + '_' + feature_info.name)

    def call(self, input):
        return self.emb_layer(input)

class dnn_layer(Layer):
    def __init__(self, unit, activation, output_dim):
        super(dnn_layer, self).__init__()
        self.unit = unit
        self.activation = activation
        self.output_dim = output_dim
        self.dense_list = [Dense(i, activation = self.activation) for i in self.unit]
        self.output_layer = Dense(self.output_dim, activation = None)
    def call(self, input):
        x = input
        for dense in self.dense_list:
            x = dense(x)
        output = self.output_layer(x)
        return output


class Liner(Layer):
    def __init__(self):
        super(Liner, self).__init__()
        self.out_layer = Dense(1, activation = None, use_bias=True)

    def call(self, input):
        out = self.out_layer(input)
        return out


class WDL(Layer):
    def __init__(self, spars_feature, units, dnn_out_dim):
        super(WDL, self).__init__()
        self.liner = Liner()
        self.emb = Dense(10, activation = None)
        self.dense1 = Dense(1, activation = None)
        self.dnn = dnn_layer(units, tf.nn.relu, output_dim=dnn_out_dim)

    def call(self, input):
        liner_output = self.liner(input)
        emb_output = self.emb(input)
        dnn_output = self.dnn(emb_output)
        dnn_output = tf.keras.layers.Flatten()(dnn_output)
        model = Concatenate(axis=1)([liner_output, dnn_output])
        return self.dense1(liner_output)


class model(Model):
    def __init__(self, spars_feature, units, dnn_out_dim):
        super(model, self).__init__()
        self.base_model = WDL(spars_feature, units, dnn_out_dim)

    def call(self, input):
        model_out = self.base_model(input)
        return tf.nn.sigmoid(model_out)
