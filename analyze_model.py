#!/usr/bin/env python

"""Analyze a Keras model."""

import logging
import sys
import keras.layers.convolutional
import keras.layers.normalization
from keras import backend as K
from keras.models import load_model
import seaborn as sns
sns.set_style("whitegrid")
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
import glob

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

    layer_index = 0
    activations_by_layer = []
    labels = []
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
            activations_by_layer.append(data)
            labels.append(layer_index)
        layer_index += 1

    # Activations
    for label, fw in enumerate(activations_by_layer):
        print("99% filter weight interval of layer {}: [{:.2f}, {:.2f}]"
              .format(label, np.percentile(fw, 0.5), np.percentile(fw, 99.5)))

    f, ax1 = plt.subplots(1, 1)
    p = sns.violinplot(data=activations_by_layer, orient="v",
                       palette=sns.color_palette(palette="RdBu", n_colors=1),
                       ax=ax1)
    p.tick_params(labelsize=16)
    ax1.set_xticklabels(labels)
    ax1.set_title('Convolution activations by layer')
    sns.plt.show()


def show_conv_weight_dist(model, small_thres=10**-6):
    """Show the distribution of conv weights of model."""
    layer_index = 0
    filter_weights = []
    bias_weights = []
    filter_weight_ranges = []
    labels = []
    for layer in model.layers:
        if not isinstance(layer, keras.layers.convolutional.Conv2D):
            layer_index += 1
            continue
        weights = layer.get_weights()

        # Filter
        print("{}: {} filter weights in {}th layer"
              .format(weights[0].shape,
                      reduce(operator.mul, weights[0].shape, 1),
                      layer_index))

        # Measure distribution (replacement by 1x1 filters)
        ranges = []
        w, h, range_i, range_j = weights[0].shape
        if w > 1 or h > 1:
            for i in range(range_i):  # one filter, but different channel
                for j in range(range_j):  # different filters
                    elements = weights[0][:, :, i, j].flatten()
                    ranges.append(elements.max() - elements.min())
            ranges = np.array(ranges)
            filter_weight_ranges.append(ranges)

        # measure distribution
        data = weights[0].flatten()
        labels.append(layer_index)
        filter_weights.append(data)
        data_small = np.array([el for el in data if abs(el) < small_thres])
        print("< {}: {}".format(small_thres, len(data_small)))

        # Bias
        if len(weights) > 1:
            print("{}: {} bias weights in {}th layer"
                  .format(weights[1].shape,
                          reduce(operator.mul, weights[1].shape, 1),
                          layer_index))
            data = weights[1].flatten()
            bias_weights.append(data)
            data_small = np.array([el for el in data if abs(el) < small_thres])
            print("< {}: {}".format(small_thres, len(data_small)))
        else:
            print("No bias in layer {}".format(layer_index))
        layer_index += 1

    labels = [1, 3, 5, 7, 9, 11, 13, 15]  # baseline

    # Filter weight ranges
    f, ax1 = plt.subplots(1, 1)
    p = sns.violinplot(data=filter_weight_ranges, orient="v",
                       palette=sns.color_palette(palette="RdBu", n_colors=1),
                       ax=ax1)
    p.tick_params(labelsize=16)
    p.set_xlabel('Layer', fontsize=20)
    p.set_ylabel('Weight range', fontsize=20)
    ax1.set_xticklabels(labels)
    ax1.set_title('Filter weight ranges by layer')
    sns.plt.show()

    # Filter weights
    for label, fw in enumerate(filter_weights):
        print("99% filter weight interval of layer {}: [{:.2f}, {:.2f}]"
              .format(label, np.percentile(fw, 0.5), np.percentile(fw, 99.5)))

    f, ax1 = plt.subplots(1, 1)
    p = sns.violinplot(data=filter_weights, orient="v",
                       palette=sns.color_palette(palette="RdBu", n_colors=1),
                       ax=ax1)
    p.tick_params(labelsize=16)
    p.set_xlabel('Layer', fontsize=20)
    ax1.set_xticklabels(labels)
    ax1.set_title('Filter weight distribution by layer')
    sns.plt.show()

    # Bias weights
    for label, fw in enumerate(bias_weights):
        print("99% bias weight interval of layer {}: [{:.2f}, {:.2f}]"
              .format(label, np.percentile(fw, 0.5), np.percentile(fw, 99.5)))

    f, ax1 = plt.subplots(1, 1)
    p = sns.violinplot(data=bias_weights[:], orient="v",
                       palette=sns.color_palette(palette="RdBu", n_colors=1),
                       ax=ax1)
    p.tick_params(labelsize=16)
    p.set_xlabel('Layer', fontsize=20)
    ax1.set_xticklabels(labels[:])
    ax1.set_title('Bias weight distribution by layer')
    sns.plt.show()


def show_batchnorm_weight_dist(model):
    """Show the distribution of batch norm weighs for one model."""
    # analyze
    gamma_weights = []
    beta_weights = []
    layer_index = 0
    labels = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            labels.append(layer_index)
            weights = layer.get_weights()
            data = weights[0].flatten()
            gamma_weights.append(data)
            data = weights[1].flatten()
            beta_weights.append(data)
        layer_index += 1

    labels = [2, 4, 6, 8, 10, 12, 14, 16]  # baseline

    # Gamma weights
    if len(gamma_weights) > 0:
        for label, fw in zip(labels, gamma_weights):
            print("99% gamma interval of layer {}: [{:.2f}, {:.2f}]"
                  .format(label,
                          np.percentile(fw, 0.5),
                          np.percentile(fw, 99.5)))

        f, ax1 = plt.subplots(1, 1)
        p = sns.violinplot(data=gamma_weights, orient="v",
                           palette=sns.color_palette(palette="RdBu",
                                                     n_colors=1),
                           ax=ax1)
        p.tick_params(labelsize=16)
        p.set_xlabel('Layer', fontsize=20)
        ax1.set_xticklabels(labels)
        ax1.set_title('Gamma distribution by layer')
        sns.plt.show()

    # beta weights
    if len(beta_weights) > 0:
        for label, fw in zip(labels, beta_weights):
            print("99% beta interval of layer {}: [{:.2f}, {:.2f}]"
                  .format(label,
                          np.percentile(fw, 0.5),
                          np.percentile(fw, 99.5)))

        f, ax1 = plt.subplots(1, 1)
        p = sns.violinplot(data=beta_weights, orient="v",
                           palette=sns.color_palette(palette="RdBu",
                                                     n_colors=1),
                           ax=ax1)
        p.tick_params(labelsize=16)
        p.set_xlabel('Layer', fontsize=20)
        ax1.set_xticklabels(labels)
        ax1.set_title('Beta distribution by layer')
        sns.plt.show()


def main(config, data_module, model_path, image_fname):
    """Run analysis."""
    model = load_model(model_path)

    print("## Activation analyzation")
    if image_fname is not None:
        image = scipy.misc.imread(image_fname)
    else:
        data = data_module.load_data(config)
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
    print("## BN analyzation")
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
        model_paths = glob.glob("{}/*.h5".format(artifacts_path))
        model_paths = [m for m in model_paths if "_chk.h5" not in m]
        model_path = model_paths[0]
    logging.info("Take {}".format(model_path))
    main(config, data, model_path, args.image_fname)
