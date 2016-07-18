import sys
import argparse
import os
os.environ["GLOG_minloglevel"] = "3"  # Disable low level Caffe logs

import caffe
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal.pool import pool_2d
from scipy.misc import imread, imsave
from scipy.optimize import fmin_l_bfgs_b
from tqdm import tqdm


def vgg_conv_layer(input_, vgg_caffe, layer_name, layer_dict):
    """Construct a VGG convolutional layer from a Caffe net."""
    W_weights = vgg_caffe.params[layer_name][0].data
    b_weights = vgg_caffe.params[layer_name][1].data
    W = theano.shared(W_weights, borrow=True)
    b = theano.shared(b_weights, borrow=True)
    layer = T.nnet.relu(T.nnet.conv2d(input_, W, border_mode="half") + b.dimshuffle("x", 0, "x", "x"))
    layer_dict[layer_name] = layer
    return layer


def load_img(filename):
    """Load an image, and process it as needed by VGGNet."""
    try:
        # Bring the color dimension to the front, convert to BGR.
        img = imread(filename).transpose((2, 0, 1))[::-1].astype(theano.config.floatX)
    except OSError as e:
        print(e)
        sys.exit(1)
    if img.shape[1] > 1024 or img.shape[2] > 1024:
        print("Error: {}: due to Theano limitations, images with any side greater than 1024 pixels are not supported".format(filename))
        sys.exit(1)
    # Subtract mean from each color.
    img[0, :, :] -= 103.939
    img[1, :, :] -= 116.779
    img[2, :, :] -= 123.68
    return img[np.newaxis, :]


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--content-image", type=str, required=True)
arg_parser.add_argument("--style-image", type=str, required=True)
arg_parser.add_argument("--output-image", type=str, required=True)
arg_parser.add_argument("--content-layer", type=str, default="conv4_2")
arg_parser.add_argument("--style-layers", type=str, nargs="+", default=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"])
arg_parser.add_argument("--content-weight", type=float, default=0.001)
arg_parser.add_argument("--style-weight", type=float, default=2e5)
arg_parser.add_argument("--iterations", type=int, default=500)
args = arg_parser.parse_args()

content_image = load_img(args.content_image)
style_image = load_img(args.style_image)

# Construct VGGNet.
if not os.path.exists("data/vgg19"):
    print("Error: VGG weights not found: run ./download_vgg.sh")
    sys.exit(1)
vgg_caffe = caffe.Net("data/vgg19/vgg19.prototxt", "data/vgg19/vgg19_normalized.caffemodel", caffe.TEST)
layer_dict = {}

X = theano.shared(np.array([[[[]]]], dtype=theano.config.floatX), borrow=True)
conv1_1 = vgg_conv_layer(X, vgg_caffe, "conv1_1", layer_dict)
conv1_2 = vgg_conv_layer(conv1_1, vgg_caffe, "conv1_2", layer_dict)
pool1 = pool_2d(conv1_2, (2, 2), ignore_border=True, mode="average_exc_pad")

conv2_1 = vgg_conv_layer(pool1, vgg_caffe, "conv2_1", layer_dict)
conv2_2 = vgg_conv_layer(conv2_1, vgg_caffe, "conv2_2", layer_dict)
pool2 = pool_2d(conv2_2, (2, 2), ignore_border=True, mode="average_exc_pad")

conv3_1 = vgg_conv_layer(pool2, vgg_caffe, "conv3_1", layer_dict)
conv3_2 = vgg_conv_layer(conv3_1, vgg_caffe, "conv3_2", layer_dict)
conv3_3 = vgg_conv_layer(conv3_2, vgg_caffe, "conv3_3", layer_dict)
conv3_4 = vgg_conv_layer(conv3_3, vgg_caffe, "conv3_4", layer_dict)
pool3 = pool_2d(conv3_4, (2, 2), ignore_border=True, mode="average_exc_pad")

conv4_1 = vgg_conv_layer(pool3, vgg_caffe, "conv4_1", layer_dict)
conv4_2 = vgg_conv_layer(conv4_1, vgg_caffe, "conv4_2", layer_dict)
conv4_3 = vgg_conv_layer(conv4_2, vgg_caffe, "conv4_3", layer_dict)
conv4_4 = vgg_conv_layer(conv4_3, vgg_caffe, "conv4_4", layer_dict)
pool4 = pool_2d(conv4_4, (2, 2), ignore_border=True, mode="average_exc_pad")

conv5_1 = vgg_conv_layer(pool4, vgg_caffe, "conv5_1", layer_dict)
conv5_2 = vgg_conv_layer(conv5_1, vgg_caffe, "conv5_2", layer_dict)
conv5_3 = vgg_conv_layer(conv5_2, vgg_caffe, "conv5_3", layer_dict)
conv5_4 = vgg_conv_layer(conv5_3, vgg_caffe, "conv5_4", layer_dict)
pool5 = pool_2d(conv5_4, (2, 2), ignore_border=True, mode="average_exc_pad")

print("Compiling functions...")

# Build the content loss.
X.set_value(content_image, borrow=True)
try:
    get_content_target = theano.function([], layer_dict[args.content_layer])
except KeyError:
    print("Error: {}: unrecognized layer".format(args.content_layer))
    sys.exit(1)
content_target = theano.shared(get_content_target(), borrow=True)
content_loss = T.sum(T.sqr(layer_dict[args.content_layer] - content_target))

# Build the style loss.
style_loss = 0.
X.set_value(style_image, borrow=True)
for layer_name in args.style_layers:
    try:
        target = layer_dict[layer_name]
    except KeyError:
        print("Error: {}: unrecognized layer".format(layer_name))
        sys.exit(1)
    target_flat = T.reshape(target, (target.shape[1], -1))
    gram_target = T.dot(target_flat, target_flat.T)
    get_gram_target = theano.function([], gram_target)
    style_gram_target = theano.shared(get_gram_target(), borrow=True)
    gram_loss_normalizer = T.cast(T.square(target.shape[1] * target.shape[2] * target.shape[3]), theano.config.floatX)
    style_loss = style_loss + T.sum(T.sqr(gram_target - style_gram_target)) / gram_loss_normalizer

# Build the overall loss and gradient. Wrapper functions are needed to interface with fmin_l_bfgs_b.
loss = args.content_weight * content_loss + args.style_weight * style_loss
loss_function_theano = theano.function([], loss)


def loss_function(X_flat):
    X.set_value(X_flat.reshape(content_image.shape).astype(theano.config.floatX))
    return loss_function_theano().astype(np.float64)


grad = T.grad(loss, X)
grad_function_theano = theano.function([], grad)


def grad_function(X_flat):
    X.set_value(X_flat.reshape(content_image.shape).astype(theano.config.floatX))
    return np.array(grad_function_theano()).flatten().astype(np.float64)


# Optimize with fmin_l_bfgs_b. The function only works on vectors; so image needs to be flattened.
output_image_flat = np.random.normal(size=content_image.shape, scale=10).flatten()
with tqdm(desc="Generating image", total=args.iterations, file=sys.stdout, ncols=80, ascii=False, bar_format="{l_bar}{bar}|") as pbar:
    output_image_flat = fmin_l_bfgs_b(loss_function, output_image_flat, grad_function, maxiter=args.iterations, callback=lambda xk: pbar.update(1))[0]

# "Deprocess" the output image.
output_image = output_image_flat.reshape(content_image.shape)[0, :, :, :]
output_image[0, :, :] += 103.939
output_image[1, :, :] += 116.779
output_image[2, :, :] += 123.68
output_image = np.clip(output_image, 0, 255).astype(np.uint8)
output_image = output_image[::-1].transpose((1, 2, 0))

try:
    imsave(args.output_image, output_image)
except OSError as e:
    print(e)
    sys.exit(1)
