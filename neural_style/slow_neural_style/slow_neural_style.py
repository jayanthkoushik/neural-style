import sys
import argparse

import theano
import theano.tensor as T
import numpy as np
from tqdm import tqdm

from neural_style.utils import *

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--content-image", type=str, required=True)
arg_parser.add_argument("--content-size", type=int, default=None)
arg_parser.add_argument("--style-image", type=str, required=True)
arg_parser.add_argument("--style-size", type=int, default=None)
arg_parser.add_argument("--output-image", type=str, required=True)
arg_parser.add_argument("--model", type=str, choices=models_table.keys(), default="vgg16")
arg_parser.add_argument("--content-layer", type=str, default="block2_conv2")
arg_parser.add_argument("--style-layers", type=str, nargs="+", default=["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"])
arg_parser.add_argument("--content-weight", type=float, default=4.)
arg_parser.add_argument("--style-weight", type=float, default=5e-4)
arg_parser.add_argument("--tv-weight", type=float, default=1e-4)
arg_parser.add_argument("--iterations", type=int, default=500)
arg_parser.add_argument("--lr", type=float, default=10)
arg_parser.add_argument("--lr-decay", type=float, default=5e-3)
arg_parser.add_argument("--normalize-gradient", action="store_true")
args = arg_parser.parse_args()

# Load images.
content_image = load_and_preprocess_img(args.content_image, args.content_size)
style_image = load_and_preprocess_img(args.style_image, args.style_size)

# Build model.
X = theano.shared(np.array([[[[]]]], dtype=floatX))
try:
    base_model = models_table[args.model](input_tensor=X, include_top=False, weights="imagenet")
except KeyError:
    print("Error: Unrecognized model: {}".format(args.model))
    sys.exit(1)

# Build the content loss.
X.set_value(content_image, borrow=True)
try:
    content_layer = base_model.get_layer(args.content_layer).output
except AttributeError:
    print("Error: unrecognized content layer: {}".format(args.content_layer))
    sys.exit(1)
get_content_target = theano.function([], content_layer)
content_target_np = get_content_target()
content_target = theano.shared(content_target_np, borrow=True)
content_loss = T.sum(T.sqr(content_layer - content_target)) / T.cast(content_target_np.size, floatX)

# Build the style loss.
style_loss = 0.
X.set_value(style_image, borrow=True)
for layer_name in args.style_layers:
    try:
        style_layer = base_model.get_layer(layer_name).output
    except AttributeError:
        print("Error: unrecognized style layer: {}".format(layer_name))
        sys.exit(1)
    style_layer_flat = T.reshape(style_layer, (style_layer.shape[1], -1))
    gram_target = T.dot(style_layer_flat, style_layer_flat.T) / T.cast(style_layer_flat.size, floatX)
    get_gram_target = theano.function([], gram_target)
    style_gram_target = theano.shared(get_gram_target(), borrow=True)
    style_loss += T.sum(T.sqr(gram_target - style_gram_target))

# Build the TV loss.
tv_loss = T.sum(T.abs_(X[:, :, 1:, :] - X[:, :, :-1, :])) + T.sum(T.abs_(X[:, :, :, 1:] - X[:, :, :, :-1]))

# Build the total loss, and optimization funciton.
loss = (args.content_weight * content_loss) + (args.style_weight * style_loss) + (args.tv_weight * tv_loss)
X.set_value(np.random.normal(size=content_image.shape, scale=10).astype(floatX), borrow=True)
optim_step = theano.function([], loss, updates=get_adam_updates(loss, [X], lr=args.lr, dec=args.lr_decay, norm_grads=args.normalize_gradient))

# Run the optimization loop.
with tqdm(desc="Generating image", file=sys.stdout, ncols=100, total=args.iterations, ascii=False, unit="iteration") as bar:
    for _ in range(args.iterations):
        loss = optim_step().item()
        bar.set_description("Generating image (loss {:.3g})".format(loss))
        bar.update(1)

# Save the result.
deprocess_img_and_save(X.get_value(borrow=True), args.output_image)

