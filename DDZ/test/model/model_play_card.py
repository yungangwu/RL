from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
import math
import tensorflow.keras as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Dense, Softmax, Conv1D, ZeroPadding1D
from tensorflow.keras.optimizers import Adam
from ddz.model.networks import ResdualLayer, Tanh
from tensorflow.keras.regularizers import l2


class PlayCardModel(K.Model):
    def __init__(self, filter=256, action_dim=310, resduals=10,
                 **kwargs):

        super(PlayCardModel, self).__init__(**kwargs)
        self.action_dim = action_dim
        kernel = 3
        regularizer = l2(1e-4)
        pad = math.floor((kernel - 1) / 2)
        sequence = [ZeroPadding1D(pad), Conv1D(
            filter, kernel_size=kernel, kernel_regularizer=regularizer), BatchNormalization(), ReLU()]

        for _ in range(resduals):
            sequence += [ResdualLayer(filter, kernel)]
        self.sequence = sequence

        policy_net = [ZeroPadding1D(pad), Conv1D(
            2, kernel_size=kernel, kernel_regularizer=regularizer), BatchNormalization(), ReLU()]
        policy_net += [Flatten(), Dense(self.action_dim, kernel_regularizer=regularizer), Softmax()]

        value_net = [ZeroPadding1D(pad), Conv1D(
            1, kernel_size=kernel, kernel_regularizer=regularizer), BatchNormalization(), ReLU()]
        value_net += [Flatten(), Dense(256, kernel_regularizer=regularizer), ReLU(), Dense(1, kernel_regularizer=regularizer), Tanh()]

        self.policy_net = policy_net
        self.value_net = value_net

    def call(self, inputs):
        x = inputs
        for d in self.sequence:
            x = d(x)
        po = x
        vo = x
        for p in self.policy_net:
            po = p(po)
        for v in self.value_net:
            vo = v(vo)
        return po, vo

    def plot(self, input_shape, save_file):
        input = K.layers.Input(shape=input_shape)
        plot_model = K.Model(inputs=input, outputs=self.call(input))
        tf.keras.utils.plot_model(
            plot_model, show_shapes=True, dpi=64, to_file=save_file)
        plot_model.summary()
