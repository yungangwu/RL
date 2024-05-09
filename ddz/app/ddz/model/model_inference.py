from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
from ddz.game.utils import sequence_cards_to_action_index
import math
import tensorflow.keras as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Dense, Softmax, Conv1D, ZeroPadding1D
from tensorflow.keras.optimizers import Adam
from ddz.model.networks import ResdualLayer, Tanh
from tensorflow.keras.regularizers import l2 

class InferenceModel(K.Model):
    def __init__(self, filter=256, action_dim=310, resduals=4,
                 **kwargs):

        super(InferenceModel, self).__init__(**kwargs)
        print("action dim:", action_dim)
        self.action_dim = action_dim
        kernel = 3
        pad = math.floor((kernel - 1) / 2)
        regularizer = l2(1e-4)
        sequence = [ZeroPadding1D(pad), Conv1D(
            filter, kernel_size=kernel, kernel_regularizer=regularizer), BatchNormalization(), ReLU()]

        for _ in range(resduals):
            sequence += [ResdualLayer(filter, kernel)]

        sequence += [ZeroPadding1D(pad), Conv1D(4, kernel_size=kernel, kernel_regularizer=regularizer), BatchNormalization(), ReLU()]
        sequence += [Flatten(), Dense(self.action_dim, kernel_regularizer=regularizer), Softmax()]

        self.sequence = sequence

    def call(self, inputs):
        x = inputs
        for d in self.sequence:
            x = d(x)
        return x

    def plot(self, input_shape, save_file):
        input = K.layers.Input(shape=input_shape)
        plot_model = K.Model(inputs=input, outputs=self.call(input))
        tf.keras.utils.plot_model(
            plot_model, show_shapes=True, dpi=64, to_file=save_file)
        plot_model.summary()
