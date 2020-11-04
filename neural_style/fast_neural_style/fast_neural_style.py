import sys
import os
import pickle
import argparse

import theano
import theano.tensor as T
import numpy as np
from tqdm import tqdm

from neural_style.utils import *
from neural_style.fast_neural_style.batch_generator import BatchGenerator
from neural_style.fast_neural_style.transformer_net import get_transformer_net

main_arg_parser = argparse.ArgumentParser()
subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

train_arg_parser = subparsers.add_parser("train")
train_arg_parser.add_argument("--train-dir", type=str, required=True)
train_arg_parser.add_argument("--val-dir", type=str, required=True)
train_arg_parser.add_argument("--train-iterations", type=int, default=40000)
train_arg_parser.add_argument("--val-iterations", type=int, default=10)
train_arg_parser.add_argument("--val-every", type=int, default=1000)
train_arg_parser.add_argument("--batch-size", type=int, default=4)
train_arg_parser.add_argument("--content-size", type=int, default=256)
train_arg_parser.add_argument("--perceptual-model", type=str, choices=models_table.keys(), default="vgg19")
train_arg_parser.add_argument("--content-layer", type=str, default="block4_conv2")
train_arg_parser.add_argument("--style-layers", type=str, nargs="+", default=["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"])
train_arg_parser.add_argument("--content-weight", type=float, default=8.)
train_arg_parser.add_argument("--style-weight", type=float, default=5e-4)
train_arg_parser.add_argument("--tv-weight", type=float, default=1e-4)
train_arg_parser.add_argument("--style-image", type=str, required=True)
train_arg_parser.add_argument("--style-size", type=int, default=None)
train_arg_parser.add_argument("--lr", type=float, default=1e-3)
train_arg_parser.add_argument("--lr-decay", type=float, default=2e-4)
train_arg_parser.add_argument("--output-dir", type=str, required=True)
train_arg_parser.add_argument("--test-image", type=str, default=None)
train_arg_parser.add_argument("--test-size", type=int, default=None)
train_arg_parser.add_argument("--checkpoint", action="store_true")

eval_arg_parser = subparsers.add_parser("eval")
eval_arg_parser.add_argument("--content-image", type=str, required=True)
eval_arg_parser.add_argument("--content-size", type=int, default=None)
eval_arg_parser.add_argument("--output-image", type=str, required=True)
eval_arg_parser.add_argument("--model", type=str, required=True)

args = main_arg_parser.parse_args()

if args.subcommand is None:
    print("Error: specify either train or eval")
    sys.exit(1)

# Build transformer model.
X = theano.shared(np.array([[[[]]]], dtype=floatX))
weights = None if args.subcommand == "train" else args.model
transformer_net = get_transformer_net(X, weights)
Xtr = transformer_net.output
get_Xtr = theano.function([], Xtr)

if args.subcommand == "train":
    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

    # Prepare batch generators.
    train_batch_generator = BatchGenerator(args.train_dir, args.train_iterations, args.batch_size, args.content_size)
    num_validations = int(np.ceil(args.train_iterations / args.val_every))
    val_batch_generator = BatchGenerator(args.val_dir, args.val_iterations * num_validations, args.batch_size, args.content_size, args.val_iterations)

    # Load the style, and (optionally) test image(s).
    style_image = load_and_preprocess_img(args.style_image, args.style_size)
    if args.test_image is not None:
        test_image = load_and_preprocess_img(args.test_image, args.test_size)

    # Build perceptual model.
    try:
        perceptual_net_X = models_table[args.perceptual_model](input_tensor=X, include_top=False, weights="imagenet")
        perceptual_net_Xtr = models_table[args.perceptual_model](input_tensor=Xtr, include_top=False, weights="imagenet")
    except KeyError:
        print("Error: Unrecognized model: {}".format(args.perceptual_model))
        sys.exit(1)

    # Build the content loss.
    try:
        cl_X = perceptual_net_X.get_layer(args.content_layer).output
        cl_Xtr = perceptual_net_Xtr.get_layer(args.content_layer).output
    except AttributeError:
        print("Error: unrecognized content layer: {}".format(args.content_layer))
        sys.exit(1)
    content_loss = T.sum(T.sqr(cl_X - cl_Xtr)) / T.cast(cl_X.size, floatX)

    # Build the style loss.
    style_loss = 0.
    X.set_value(style_image)
    for layer_name in args.style_layers:
        try:
            sl_X = perceptual_net_X.get_layer(layer_name).output
            sl_Xtr = perceptual_net_Xtr.get_layer(layer_name).output
        except AttributeError:
            print("Error: unrecognized style layer: {}".format(layer_name))
            sys.exit(1)
        slf_X = T.reshape(sl_X, (sl_X.shape[0], sl_X.shape[1], -1))
        gram_X = (T.batched_tensordot(slf_X, slf_X.dimshuffle(0, 2, 1), axes=1) / T.cast(slf_X.size, floatX)) * T.cast(slf_X.shape[0], floatX)
        slf_Xtr = T.reshape(sl_Xtr, (sl_Xtr.shape[0], sl_Xtr.shape[1], -1))
        gram_Xtr = (T.batched_tensordot(slf_Xtr, slf_Xtr.dimshuffle(0, 2, 1), axes=1) / T.cast(slf_Xtr.size, floatX)) * T.cast(slf_Xtr.shape[0], floatX)

        get_gram_X = theano.function([], gram_X)
        style_gram = theano.shared(get_gram_X()[0, :, :])
        style_loss += T.sum(
            T.sqr(style_gram.dimshuffle("x", 0, 1) - gram_Xtr)
        ) / T.cast(Xtr.shape[0], floatX)


    # Build the TV loss.
    tv_loss = (T.sum(T.abs_(Xtr[:, :, 1:, :] - Xtr[:, :, :-1, :])) + T.sum(T.abs_(Xtr[:, :, :, 1:] - Xtr[:, :, :, :-1]))) / T.cast(Xtr.shape[0], floatX)

    # Build the total loss, and optimization, validation funciton.
    loss = (args.content_weight * content_loss) + (args.style_weight * style_loss) + (args.tv_weight * tv_loss)
    optim_step = theano.function([], loss, updates=get_adam_updates(loss, transformer_net.trainable_weights, lr=args.lr, dec=args.lr_decay))
    get_loss = theano.function([], loss)

    # Run the optimization loop.
    train_losses, val_losses = [], []
    with tqdm(desc="Training", file=sys.stdout, ncols=100, total=args.train_iterations, ascii=False, unit="iteration") as trbar:
        for tri in range(args.train_iterations):
            X.set_value(train_batch_generator.get_batch(), borrow=True)
            loss = optim_step().item()
            train_losses.append(loss)
            trbar.set_description("Training (loss {:.3g})".format(loss))
            trbar.update(1)

            if (tri + 1) % args.val_every == 0 or (tri + 1) == args.train_iterations:
                batch_val_losses = []
                n_val = 0
                with tqdm(desc="Validating", file=sys.stdout, ncols=100, total=args.val_iterations, ascii=False, unit="iteration", leave=False) as valbar:
                    for vali in range(args.val_iterations):
                        X.set_value(val_batch_generator.get_batch(), borrow=True)
                        loss = get_loss().item()
                        batch_size = X.shape[0].eval()
                        n_val += batch_size
                        batch_val_losses.append(loss*batch_size)
                        valbar.update(1)
                mean_val_loss = np.sum(batch_val_losses) / n_val
                val_losses.append(mean_val_loss)

                if args.checkpoint:
                    transformer_net.save_weights(os.path.join(args.output_dir, "model_checkpoint_{}.h5".format(tri + 1)), overwrite=True)

                if args.test_image is not None:
                    X.set_value(test_image, borrow=True)
                    test_tr = get_Xtr()
                    deprocess_img_and_save(test_tr, os.path.join(args.output_dir, "test_iter_{}.jpg".format(tri + 1)))

    # Save weights and losses.
    try:
        with open(os.path.join(args.output_dir, "train_losses.pkl"), "wb") as f:
            pickle.dump(train_losses, f)
        with open(os.path.join(args.output_dir, "val_losses.pkl"), "wb") as f:
            pickle.dump(val_losses, f)
        transformer_net.save_weights(os.path.join(args.output_dir, "model.h5"), overwrite=True)
    except OSError as e:
        print(e)
        sys.exit(1)

else:
    content_image = load_and_preprocess_img(args.content_image, args.content_size)
    X.set_value(content_image)
    output_image = get_Xtr()
    deprocess_img_and_save(output_image, args.output_image)

