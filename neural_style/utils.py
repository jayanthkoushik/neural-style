import sys

import theano
import theano.tensor as T
import numpy as np
from scipy.misc import imread, imsave, imresize
from keras.applications import VGG16, VGG19, ResNet50

floatX = theano.config.floatX
models_table = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet50": ResNet50,
}


def subtract_imagenet_mean(img):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    img[0, :, :] -= 103.939
    img[1, :, :] -= 116.779
    img[2, :, :] -= 123.68


def add_imagenet_mean(img):
    """Add ImageNet mean pixel-wise to a BGR image."""
    img[0, :, :] += 103.939
    img[1, :, :] += 116.779
    img[2, :, :] += 123.68


def load_and_preprocess_img(filename, size=None, center_crop=False):
    """Load an image, and pre-process it as needed by models."""
    try:
        img = imread(filename, mode="RGB")
    except OSError as e:
        print(e)
        sys.exit(1)

    if center_crop:
        # Extract a square crop from the center of the image.
        cur_shape = img.shape[:2]
        shorter_side = min(cur_shape)
        longer_side_xs = max(cur_shape) - shorter_side
        longer_side_start = int(longer_side_xs / 2.)
        longer_side_slice = slice(longer_side_start, longer_side_start + shorter_side)
        if shorter_side == cur_shape[0]:
            img = img[:, longer_side_slice, :]
        else:
            img = img[longer_side_slice, :, :]

    if size is not None:
        # Resize the image.
        cur_shape = img.shape[:2]
        shorter_side = min(cur_shape)
        aspect = max(cur_shape) / float(shorter_side)
        new_shorter_side = int(size / aspect)
        if shorter_side == cur_shape[0]:
            new_shape = (new_shorter_side, size)
        else:
            new_shape = (size, new_shorter_side)
        img = imresize(img, new_shape)

    # Bring the color dimension to the front, convert to BGR.
    img = img.transpose((2, 0, 1))[::-1].astype(floatX)

    subtract_imagenet_mean(img)
    return img[np.newaxis, :]


def deprocess_img_and_save(img, filename):
    """Undo pre-processing on an image, and save it."""
    img = img[0, :, :, :]
    add_imagenet_mean(img)
    img = img[::-1].transpose((1, 2, 0))
    img = np.clip(img, 0, 255).astype(np.uint8)
    try:
        imsave(filename, img)
    except OSError as e:
        print(e)
        sys.exit(1)


def get_adam_updates(f, params, lr=10., b1=0.9, b2=0.999, e=1e-8, dec=5e-3, norm_grads=False):
    """Generate updates to optimize using the Adam optimizer with linear learning rate decay."""
    t = theano.shared(0)
    ms = [theano.shared(np.zeros(param.shape.eval(), dtype=floatX), borrow=True) for param in params]
    vs = [theano.shared(np.zeros(param.shape.eval(), dtype=floatX), borrow=True) for param in params]

    gs = T.grad(f, params)
    if norm_grads:
        gs = [g / (T.sum(T.abs_(g)) + 1e-8) for g in gs]
    t_u = (t, t + 1)
    m_us = [(m, b1 * m + (1. - b1) * g) for m, g in zip(ms, gs)]
    v_us = [(v, b2 * v + (1. - b2) * T.sqr(g)) for v, g in zip(vs, gs)]
    lr_hat =  (lr / (1. + t_u[1] * dec)) * T.sqrt(1. - T.pow(b2, t_u[1])) / (1. - T.pow(b1, t_u[1]))
    param_us = [(param,  T.cast(param - lr_hat * m_u[1] / (T.sqrt(v_u[1]) + e), floatX)) for m_u, v_u, param in zip(m_us, v_us, params)]
    return m_us + v_us + param_us + [t_u]

