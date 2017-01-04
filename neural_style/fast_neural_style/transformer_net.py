import sys

import numpy as np
import theano.tensor as T
from keras.layers import Input, Conv2D, Activation, Lambda, UpSampling2D, merge
from keras.models import Model
from keras.engine.topology import Layer

from neural_style.utils import floatX


class InstanceNormalization(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[1],), initializer="uniform", trainable=True)
        self.shift = self.add_weight(shape=(input_shape[1],), initializer="zero", trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        hw = T.cast(x.shape[2] * x.shape[3], floatX)
        mu = x.sum(axis=-1).sum(axis=-1) / hw
        mu_vec = mu.dimshuffle(0, 1, "x", "x")
        sig2 = T.square(x - mu_vec).sum(axis=-1).sum(axis=-1) / hw
        y = (x - mu_vec) / T.sqrt(sig2.dimshuffle(0, 1, "x", "x") + 1e-5)
        return self.scale.dimshuffle("x", 0, "x", "x") * y + self.shift.dimshuffle("x", 0, "x", "x")


class ReflectPadding2D(Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = padding
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, mask=None):
        p0, p1 = self.padding[0], self.padding[1]
        y = T.zeros((x.shape[0], x.shape[1], x.shape[2]+(2*p0), x.shape[3]+(2*p1)), dtype=floatX)
        y = T.set_subtensor(y[:, :, p0:-p0, p1:-p1], x)
        y = T.set_subtensor(y[:, :, :p0, p1:-p1], x[:, :, p0:0:-1, :])
        y = T.set_subtensor(y[:, :, -p0:, p1:-p1], x[:, :, -2:-2-p0:-1])
        y = T.set_subtensor(y[:, :, p0:-p0, :p1], x[:, :, :, p1:0:-1])
        y = T.set_subtensor(y[:, :, p0:-p0, -p1:], x[:, :, :, -2:-2-p1:-1])
        y = T.set_subtensor(y[:, :, :p0, :p1], x[:, :, p0:0:-1, p1:0:-1])
        y = T.set_subtensor(y[:, :, -p0:, :p1], x[:, :, -2:-2-p0:-1, p1:0:-1])
        y = T.set_subtensor(y[:, :, :p0, -p1:], x[:, :, p0:0:-1, -2:-2-p1:-1])
        y = T.set_subtensor(y[:, :, -p0:, -p1:], x[:, :, -2:-2-p0:-1, -2:-2-p1:-1])
        return y

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]+(2*self.padding[0]), input_shape[3]+(2*self.padding[1]))


def conv_layer(in_, nb_filter, filter_length, subsample=1, upsample=1, only_conv=False):
    if upsample != 1:
        out = UpSampling2D(size=(upsample, upsample))(in_)
    else:
        out = in_
    padding = int(np.floor(filter_length / 2))
    out = ReflectPadding2D((padding, padding))(out)
    out = Conv2D(nb_filter, filter_length, filter_length, subsample=(subsample, subsample), border_mode="valid")(out)
    if not only_conv:
        out = InstanceNormalization()(out)
        out = Activation("relu")(out)
    return out


def residual_block(in_):
    out = conv_layer(in_, 128, 3)
    out = conv_layer(out, 128, 3, only_conv=True)
    return merge([out, in_], mode="sum")


def get_transformer_net(X, weights=None):
    input_ = Input(tensor=X, shape=(3, 256, 256))
    y = conv_layer(input_, 32, 9)
    y = conv_layer(y, 64, 3, subsample=2)
    y = conv_layer(y, 128, 3, subsample=2)
    y = residual_block(y)
    y = residual_block(y)
    y = residual_block(y)
    y = residual_block(y)
    y = residual_block(y)
    y = conv_layer(y, 64, 3, upsample=2)
    y = conv_layer(y, 32, 3, upsample=2)
    y = conv_layer(y, 3, 9, only_conv=True)
    y = Activation("tanh")(y)
    y = Lambda(lambda x: (x * 150) + 127.5, output_shape=(3, None, None))(y)
    y = Lambda(lambda x: x - np.array([103.939, 116.779, 123.68], dtype=floatX).reshape(1, 3, 1, 1), output_shape=(3, None, None))(y)

    net = Model(input=input_, output=y)
    if weights is not None:
        try:
            net.load_weights(weights)
        except OSError as e:
            print(e)
            sys.exit(1)
    return net

