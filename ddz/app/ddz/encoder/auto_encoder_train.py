from numpy.lib.npyio import save
from ddz.game.utils import *
import tensorflow.keras as K
import tensorflow as tf
import datetime
import numpy as np
import time
import random
from tensorflow.keras.layers import Conv2D, LayerNormalization
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers.convolutional import Conv, Conv2DTranspose
from ddz.model.networks import ResdualLayer, card_values_to_onehot_60, cards_to_onehot_60


BATCH_SIZE = 256
EPOCHS = 400

class DownsampleBlock(K.layers.Layer):
    def __init__(self, filter=3, kernel_size=3, **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)

        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.sequence = [LayerNormalization(scale=True)]
        self.sequence +=[K.layers.ReLU()]

        self.sequence +=[Conv2D(filter, kernel_size=[1, kernel_size],
                    strides=[1, 2], padding='same', bias_initializer=None, kernel_initializer=initializer)]
        self.sequence +=[LayerNormalization(scale=True)]
        self.sequence +=[K.layers.ReLU()]

        self.sequence +=[Conv2D(filter, kernel_size=[1, kernel_size],
                    strides=[1, 1], padding='same', bias_initializer=None, kernel_initializer=initializer)]

        self.shortcut = Conv2D(filter, kernel_size=[1, 1],
                    strides=[1, 2], padding='same', bias_initializer=None, kernel_initializer=initializer)

    def call(self, inputs):
        x = inputs
        x = self.sequence[0](x)
        x = self.sequence[1](x)
        r = x
        s = self.shortcut(x)
        for i in range(2, len(self.sequence)):
            r = self.sequence[i](r)
        return s + r


class UpsampleBlock(K.layers.Layer):
    def __init__(self, filter=3, kernel_size=3, **kwargs):
        super(UpsampleBlock, self).__init__(**kwargs)

        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.sequence = [LayerNormalization(scale=True)]
        self.sequence +=[K.layers.ReLU()]

        self.sequence +=[Conv2DTranspose(filter, kernel_size=[1, kernel_size],
                    strides=[1, kernel_size], padding='same', bias_initializer=None, kernel_initializer=initializer)]
        self.sequence +=[LayerNormalization(scale=True)]
        self.sequence +=[K.layers.ReLU()]

        self.sequence +=[Conv2DTranspose(filter, kernel_size=[1, kernel_size],
                    strides=[1, 1], padding='same', bias_initializer=None, kernel_initializer=initializer)]

        self.shortcut = Conv2DTranspose(filter, kernel_size=[1, 1],
                    strides=[1, kernel_size], padding='same', bias_initializer=None, kernel_initializer=initializer)

    def call(self, inputs):
        x = inputs
        x = self.sequence[0](x)
        x = self.sequence[1](x)

        s = self.shortcut(x)
        r = x
        for i in range(2, len(self.sequence)):
            r = self.sequence[i](r)
        return s + r

class Encoder(K.Model):
    def __init__(self, input_filter=32,
                 **kwargs):

        super(Encoder, self).__init__(**kwargs)

        inp_conv = [Conv2D(input_filter, kernel_size=(1,1), strides=(1,4), padding='same')]
        inp_conv +=[Conv2D(input_filter, kernel_size=(1,2), strides=(1,4), padding='same')]
        inp_conv +=[Conv2D(input_filter, kernel_size=(1,3), strides=(1,4), padding='same')]
        inp_conv +=[Conv2D(input_filter, kernel_size=(1,4), strides=(1,4), padding='same')]

        encoding_params =  [[128, 3, 'resdual'],            ## (1, 15)
                            [128, 3, 'resdual'],            ## (1, 15)
                            [128, 3, 'downsampling'],       ## (1, 8)
                            [128, 3, 'resdual'],            ## (1, 8)
                            [128, 3, 'resdual'],            ## (1, 8)
                            [256, 3, 'downsampling'],       ## (1, 4)
                            [256, 3, 'resdual'],            ## (1, 4)
                            [256, 3, 'resdual']             ## (1, 4, 256)
                        ]

        downsamples= []
        for param in encoding_params:
            if param[-1] == 'resdual':
                downsamples += [ResdualLayer(param[0], param[1], norm_layer=LayerNormalization)]
            elif param[-1] == 'downsampling':
                downsamples += [DownsampleBlock(param[0], param[1])]

        decoding_params =  [[256, 4, 'upsampling'],         ## (1, 8)
                            [256, 3, 'resdual'],            ## (1, 8)
                            [256, 3, 'resdual'],            ## (1, 8)
                            [256, 4, 'upsampling'],         ## (1, 16)
                            [128, 3, 'conv'],               ## (1, 16)
                            [128, 3, 'resdual'],            ## (1, 16)
                            [128, 3, 'resdual'],            ## (1, 16)
                            [128, 4, 'upsampling'],         ## (1, 32)
                            [128, 3, 'resdual'],            ## (1, 32)
                            [1, 3, 'conv'],                 ## (1, 32)
                            [1, 3, 'resdual']               ## (1, 32, 1)
                        ]
        upsamples = []
        for param in decoding_params:
            if param[-1] == 'resdual':
                upsamples += [ResdualLayer(param[0], param[1], norm_layer=LayerNormalization)]
            elif param[-1] == 'upsampling':
                upsamples += [UpsampleBlock(param[0], param[1])]
            elif param[-1] == 'conv':
                upsamples += [Conv2D(param[0], param[1], padding='same', kernel_initializer=tf.random_normal_initializer(0.0, 0.02))]

        self.inp_conv = inp_conv
        self.downsamples = downsamples
        self.upsamples = upsamples

    def encode(self, inputs):
        x = []
        for conv in self.inp_conv:
            x += [conv(inputs)]
        x = tf.concat(x, axis=-1) ## (#, 1, 60)
        for d in self.downsamples:
            x = d(x)
        ## (#, 1, 60)
        x = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        return x

    def call(self, inputs):
        x = []
        for conv in self.inp_conv:
            x += [conv(inputs)]
        x = tf.concat(x, axis=-1) ## (#, 1, 60)
        for d in self.downsamples:
            x = d(x)
        ## (#, 1, 60)
        x = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        ## (#, 1, 15, 256)
        encoding = tf.identity(x, name='encoding')

        for u in self.upsamples:
            x = u(x)

        output = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
        return output

    def plot(self, input_shape, save_file):
        input = K.layers.Input(shape=input_shape)
        plot_model = K.Model(inputs=input, outputs=self.call(input))
        tf.keras.utils.plot_model(
            plot_model, show_shapes=True, dpi=64, to_file=save_file)
        plot_model.summary()

class EncoderEvaluator(object):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(32)
        self.optimizer = K.optimizers.Adam(learning_rate=1e-4)
        self.summary_writer = tf.summary.create_file_writer("./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder)
        self.encoder.build((BATCH_SIZE, 1, 60, 1))

    @tf.function
    def loss(self, inputs):
        outputs = self.encoder(inputs, training=True)
        x = tf.reshape(inputs, [-1, inputs.shape[2]])
        x = tf.pad(x, [[0,0], [0,4]])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=outputs)
        loss = tf.reduce_mean(tf.reduce_sum(loss, -1))
        trainable_variables = self.encoder.trainable_variables
        if trainable_variables:
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in trainable_variables if 'bias' not in v.name]) * 0.001
        else:
            lossL2 = 0
        return loss, lossL2

    def step(self, inputs, epoch):
        with tf.GradientTape() as tape:
            loss_bc, loss_l2 = self.loss(inputs)
            loss = loss_bc + loss_l2
        gradients = tape.gradient(loss,self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        with self.summary_writer.as_default():
            tf.summary.scalar('loss_total', loss, step=epoch)
            tf.summary.scalar('loss_bc', loss_bc, step=epoch)
            tf.summary.scalar('loss_l2', loss_l2, step=epoch)

    def save(self):
        self.checkpoint.save(file_prefix='./checkpoints/ckpt')

    def load(self, file, trainable=True):
        self.checkpoint.restore(file).assert_existing_objects_matched()
        if not trainable:
            self.encoder.trainable = False

    def evaluate(self, inputs):
        x = self.encoder.encode(inputs)
        return tf.squeeze(x, axis=[1, 2])

    def run(self, train_dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            print("epoch: ", epoch)

            for n, inputs in train_dataset.enumerate():
                print('.', end='')
                if (n+1) % 100 == 0:
                    print()
                self.step(inputs, epoch)
            print()
            print('time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time()-start))
        self.save()


def get_all_cards():
    all_cards = []
    for i in range(1, 310):
        action_type, action_index = decode_action(i)
        cards = action_to_card(action_type, action_index)
        all_cards.append(cards)
    # print("num cards:", len(all_cards))
    return all_cards

ACTION_SPACE = get_all_cards()

def card_data_generator():
    action_space_onehot = [card_values_to_onehot_60(a) for a in ACTION_SPACE]
    for sample in action_space_onehot:
        sample = np.expand_dims(sample, axis=(0, -1))
        yield sample

def create_train_data():

    train_dataset = tf.data.Dataset.from_generator(card_data_generator,
                                                   output_types=tf.float32,
                                                   output_shapes=tf.TensorShape([1,60,1]))
    train_dataset = train_dataset.shuffle(len(ACTION_SPACE))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    return train_dataset

def random_generator():
    for i in range(2000):
        cards = [(x + 1, y + 1) for x in range(NUM_CARD_TYPE)
                        for y in range(NUM_REGULAR_CARD_VALUE)]
        cards.extend([(1, SMALL_JOKER), (1, BIG_JOKER)])
        random.shuffle(cards)
        player_cards = cards[3:]
        for i in range(NUM_AGENT):
            sample = player_cards[17 * i:17 * (i + 1)]
            sample = cards_to_onehot_60(sample)
            sample = np.expand_dims(sample, axis=(0, -1))
            yield sample

def create_random_data():

    dataset = tf.data.Dataset.from_generator(random_generator,
                                             output_types=tf.float32,
                                             output_shapes=tf.TensorShape([1,60,1]))
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def train():
    train_dataset = create_random_data()

    evaluator = EncoderEvaluator()
    evaluator.run(train_dataset, EPOCHS)



def test():
    train_dataset = create_random_data()
    evaluator = EncoderEvaluator()
    evaluator.load("./checkpoints/ckpt-1", False)
    for n, inputs in train_dataset.enumerate():
        print("inputs:", inputs[0])
        encoded_inputs = evaluator.encoder(inputs)
        print("encoded_inputs:", encoded_inputs[0])
        print("loss:", evaluator.loss(inputs))
        if n > 1:
            break
