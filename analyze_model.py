#!/usr/bin/env python

"""Analyze a Keras model."""

import logging
import sys
import keras.layers.convolutional
import keras.layers.normalization
from keras import backend as K
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import operator
from functools import reduce
import numpy as np
import yaml
import imp
from run_training import make_paths_absolute
import os
import pprint
import scipy.misc
import scipy.stats

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


from msthesis_utils import make_mosaic


def get_activations(model, layer_index, X_batch):
    """Get activation of one layer for one input."""
    get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[layer_index].output, ])
    activations = get_activations([X_batch, 0])
    return activations


def show_conv_act_distrib(model, X, show_feature_maps=False):
    """Show the distribution of convolutional layers for one input."""
    X_train = np.array([X])
    f, (ax1, ax2) = plt.subplots(1, 2)

    # Get layer indices
    layer_index = 0
    first_layer_index = None
    last_layer_index = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            if first_layer_index is None:
                first_layer_index = layer_index
            last_layer_index = layer_index
        layer_index += 1

    layer_index = 0
    for layer_index in range(len(model.layers)):
        act = get_activations(model, layer_index, X_train)[0]
        if show_feature_maps:
            scipy.misc.imshow(X_train[0])
            mosaik = make_mosaic(act[0], 8, 4)
            scipy.misc.imshow(mosaik)
        data = act[0].flatten()
        if isinstance(model.layers[layer_index],
                      keras.layers.convolutional.Conv2D):
            print("\tlayer {}: len(data)={}".format(layer_index, len(data)))
            if layer_index == first_layer_index:
                color = 'red'
                ax = ax1
            elif layer_index == last_layer_index:
                color = 'lime'
                ax = ax1
            else:
                color = None
                ax = ax2
            if len(data) > 3:
                _, p = scipy.stats.normaltest(data)
                print("\tData is normal distributed (p={})".format(p))
            sns.distplot(data, hist=False, norm_hist=False, kde=True, ax=ax,
                         label=str(layer_index),
                         color=color)
        layer_index += 1
    ax1.set_title('convolution activation of first layer')
    ax1.legend()
    ax2.set_title('convolution activations')
    ax2.legend()
    sns.plt.show()


def show_conv_weight_dist(model, small_thres=10**-6):
    """Show the distribution of conv weights of model."""
    f, (ax1, ax2) = plt.subplots(1, 2)

    # Get layer indices
    layer_index = 0
    first_layer_index = None
    last_layer_index = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            if first_layer_index is None:
                first_layer_index = layer_index
            last_layer_index = layer_index
        layer_index += 1

    layer_index = 0
    filter_weights = []
    for layer in model.layers:
        if not isinstance(layer, keras.layers.convolutional.Conv2D):
            layer_index += 1
            continue
        weights = layer.get_weights()

        # Color
        if layer_index == first_layer_index:
            color = 'red'
        elif layer_index == last_layer_index:
            color = 'lime'
        else:
            color = None

        # Filter
        print("{}: {} filter weights in {}th layer"
              .format(weights[0].shape,
                      reduce(operator.mul, weights[0].shape, 1),
                      layer_index))
        data = weights[0].flatten()
        filter_weights.append(data)
        data_small = np.array([el for el in data if abs(el) < small_thres])
        print("< {}: {}".format(small_thres, len(data_small)))
        sns.distplot(data, hist=False, norm_hist=True, kde=True, ax=ax1,
                     label=str(layer_index), color=color)

        # Bias
        print("{}: {} bias weights in {}th layer"
              .format(weights[1].shape,
                      reduce(operator.mul, weights[1].shape, 1),
                      layer_index))
        data = weights[1].flatten()
        data_small = np.array([el for el in data if abs(el) < small_thres])
        print("< {}: {}".format(small_thres, len(data_small)))
        sns.distplot(data, hist=False, norm_hist=False, kde=True, ax=ax2,
                     label=str(layer_index), color=color)
        layer_index += 1
    ax1.set_title('Filter weights')
    ax2.set_title('Bias weights')
    ax1.legend()
    ax2.legend()
    sns.plt.show()

    ax = sns.violinplot(data=filter_weights, orient="h", palette="Set2")
    ax.set_title('Filter weight distribution by layer')
    sns.plt.show()


def show_batchnorm_weight_dist(model):
    """Show the distribution of batch norm weighs for one model."""
    f, (ax1, ax2) = plt.subplots(1, 2)
    layer_index = 0

    first_layer_index = None
    last_layer_index = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            if first_layer_index is None:
                first_layer_index = layer_index
            last_layer_index = layer_index
        layer_index += 1

    layer_index = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            # Color
            if layer_index == first_layer_index:
                color = 'red'
            elif layer_index == last_layer_index:
                color = 'lime'
            else:
                color = None

            weights = layer.get_weights()
            data = weights[0].flatten()
            sns.distplot(data, hist=False, norm_hist=False, kde=True, ax=ax1,
                         label=str(layer_index), color=color)
            data = weights[1].flatten()
            sns.distplot(data, hist=False, norm_hist=False, kde=True, ax=ax2,
                         label=str(layer_index), color=color)
        layer_index += 1
    ax1.set_title('gamma weights')
    ax2.set_title('beta weights')
    ax1.legend()
    ax2.legend()
    sns.plt.show()


def main(data_module, model_path, image_fname):
    """Run analysis."""
    model = load_model(model_path)

    print("## Activation analyzation")
    if image_fname is not None:
        image = scipy.misc.imread(image_fname)
    else:
        data = data_module.load_data()
        X_train = data['x_train']
        X_test = data['x_test']
        # y_train = data['y_train']
        # y_test = data['y_test']

        X_train = data_module.preprocess(X_train)
        X_test = data_module.preprocess(X_test)
        image = X_train[0]

    show_conv_act_distrib(model, image)

    print("## Weight analyzation")
    show_conv_weight_dist(model)
    show_batchnorm_weight_dist(model)


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="Experiment yaml file",
                        required=True,
                        metavar="FILE")
    parser.add_argument("--model",
                        dest="model_path",
                        help="Model h5 file",
                        metavar="FILE")
    parser.add_argument("--image",
                        dest="image_fname",
                        help="A single image",
                        metavar="FILE")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Read YAML experiment definition file
    with open(args.filename, 'r') as stream:
        config = yaml.load(stream)
    # Make paths absolute
    config = make_paths_absolute(os.path.dirname(args.filename),
                                 config)

    # Print experiment file
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    # Load data module
    dpath = config['dataset']['script_path']
    sys.path.insert(1, os.path.dirname(dpath))
    data = imp.load_source('data', config['dataset']['script_path'])

    # Load model
    if args.model_path is not None:
        model_path = args.model_path
    else:
        artifacts_path = config['train']['artifacts_path']
        model_path = os.path.basename(config['train']['artifacts_path'])
        model_path = os.path.join(artifacts_path,
                                  "{}.h5".format(model_path))
    main(data, model_path, args.image_fname)
