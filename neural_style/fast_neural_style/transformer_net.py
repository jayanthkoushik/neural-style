import sys

import numpy as np
import theano.tensor as T
from keras.layers import Input, Conv2D, Activation, Lambda, merge
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


class TransConv2D(Layer):

    def __init__(self, nb_filter, filter_length, subsample, **kwargs):
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.subsample = subsample
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.filters = self.add_weight(shape=(input_shape[1], self.nb_filter, self.filter_length, self.filter_length), initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(shape=(self.nb_filter,), initializer="zero", trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        output_shape = (None, self.nb_filter, x.shape[2]*self.subsample, x.shape[3]*self.subsample)
        out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(x, self.filters, output_shape, border_mode="half", subsample=(self.subsample, self.subsample))
        out = out + self.bias.dimshuffle("x", 0, "x", "x")
        return out

    def get_output_shape_for(self, input_shape):
        outh = None if input_shape[2] is None else input_shape[2]*self.subsample
        outw = None if input_shape[3] is None else input_shape[3]*self.subsample
        return (input_shape[0], self.nb_filter, outh, outw)


def conv_layer(in_, nb_filter, filter_length, subsample, only_conv=False):
    out = Conv2D(nb_filter, filter_length, filter_length, subsample=(subsample, subsample), border_mode="same")(in_)
    if not only_conv:
        out = InstanceNormalization()(out)
        out = Activation("relu")(out)
    return out


def trans_conv_layer(in_, nb_filter, filter_length, subsample):
    out = TransConv2D(nb_filter, filter_length, subsample)(in_)
    out = InstanceNormalization()(out)
    return Activation("relu")(out)


def residual_block(in_):
    out = conv_layer(in_, 128, 3, 1)
    out = conv_layer(out, 128, 3, 1, True)
    return merge([out, in_], mode="sum")


def get_transformer_net(X, weights=None):
    input_ = Input(tensor=X, shape=(3, None, None))
    y = conv_layer(input_, 32, 9, 1)
    y = conv_layer(y, 64, 3, 2)
    y = conv_layer(y, 128, 3, 2)
    y = residual_block(y)
    y = residual_block(y)
    y = residual_block(y)
    y = residual_block(y)
    y = residual_block(y)
    y = trans_conv_layer(y, 64, 3, 2)
    y = trans_conv_layer(y, 32, 3, 2)
    y = conv_layer(y, 3, 9, 1, True)
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

