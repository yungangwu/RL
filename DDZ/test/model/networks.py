import numpy as np
import tensorflow.keras as K
import tensorflow as tf
import math
from collections import Counter
from tensorflow.keras.layers import Dense, LayerNormalization, Conv2D, BatchNormalization, Activation, Layer, ReLU, Conv1D, InputSpec
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
from utils import *


def card_values_to_onehot_60(card_values):
    counts = Counter(card_values)
    onehot_code = np.zeros(60, dtype=np.float32)
    for card_value in card_values:
        card_index = card_value-1
        subvec = np.zeros(4, dtype=np.float32)
        subvec[:counts[card_value]] = 1
        onehot_code[card_index*4:card_index*4+4] = subvec
    return onehot_code


def cards_to_onehot_60(cards):
    card_values = [x[1] for x in cards]
    return card_values_to_onehot_60(card_values)


def legal_actions_mask(agent):
    mask = np.zeros(310, dtype=np.bool)
    actions = agent.get_legal_actions()
    for action in actions:
        mask[action] = True
    return mask

class ResdualDense(Layer):
    def __init__(self, units, stack=4, activation='relu', **kwargs):
        super(ResdualDense, self).__init__(**kwargs)

        sequence = []
        for i in range(stack):
            sequence += [Dense(units, activation=activation,
                              name='res_dense_{}'.format(i+1))]
        self.sequence = sequence
        self.normal_layer = LayerNormalization(scale=False)

    def call(self, inputs):
        x = inputs
        for layer in self.sequence:
            x = layer(x)
        return self.normal_layer(inputs + x)


class ResdualLayer(Layer):
    def __init__(self, filter, kernel_size=4, activation='relu', norm_layer=BatchNormalization, **kwargs):
        super(ResdualLayer, self).__init__(**kwargs)
        padding = 'same'
        initializer = tf.random_normal_initializer(0.0, 0.02)
        regularizer = l2(1e-4)
        pad = math.floor((kernel_size - 1) / 2)
        sequence =  [ZeroPadding1D(pad), Conv1D(filter, kernel_size=kernel_size,
                            kernel_initializer=initializer, kernel_regularizer=regularizer), norm_layer(), Activation(activation)]

        sequence += [ZeroPadding1D(pad), Conv1D(filter, kernel_size=kernel_size,
                            kernel_initializer=initializer, kernel_regularizer=regularizer), norm_layer()]
        self.sequence = sequence
        self.relu = ReLU()

    def call(self, inputs):
        x = inputs
        for layer in self.sequence:
            x = layer(x)
        return self.relu(inputs + x)

class ConvlBlock(Layer):
    def __init__(self, filter, kernel_size=4, activation='relu', norm_layer=BatchNormalization, **kwargs):
        super(ConvlBlock, self).__init__(**kwargs)
        padding = 'same'
        initializer = tf.random_normal_initializer(0.0, 0.02)
        sequence =  [Conv2D(filter, kernel_size=kernel_size,
                            padding=padding, kernel_initializer=initializer), norm_layer(), Activation(activation)]

        self.sequence = sequence

    def call(self, inputs):
        x = inputs
        for layer in self.sequence:
            x = layer(x)
        return x

class MaskSoftMask(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(MaskSoftMask, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, mask):
        return K.softmax(inputs*mask, axis=self.axis)

class Pad(Layer):
    def __init__(self, paddings=1, type='ZERO', **kwargs):
        self.paddings = paddings
        self.type = type
        self.input_spec = [InputSpec(ndim=3)]
        super(Pad, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1]+self.paddings[0], s[2]+self.paddings[1], s[3])

    def call(self, x):
        w_pad, h_pad = self.paddings
        x = tf.pad(x, [[0, 0], [h_pad, h_pad], [
                   w_pad, w_pad], [0, 0]], 'REFLECT')
        return x

class Tanh(Layer):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def call(self, x):
        return tf.keras.activations.tanh(x)