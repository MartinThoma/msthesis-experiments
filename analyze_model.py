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
import numpy.ma as ma
import yaml
import matplotlib.cm as cm
import imp
from run_training import make_paths_absolute
import os
import pprint
import scipy.misc
import scipy.stats

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


from mpl_toolkits.axes_grid1 import make_axes_locatable
from msthesis_utils import make_mosaic


def get_activations(model, layer_index, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[layer_index].output, ])
    activations = get_activations([X_batch, 0])
    return activations


def show_conv_act_distrib(model, X, show_feature_maps=False):
    X_train = np.array([X])
    ax = plt.axes()
    layer_count = 0
    for layer_index in range(len(model.layers)):
        act = get_activations(model, layer_index, X_train)[0]
        if show_feature_maps:
            scipy.misc.imshow(X_train[0])
            mosaik = make_mosaic(act[0], 8, 4)
            scipy.misc.imshow(mosaik)
        data = act[0].flatten()
        if isinstance(model.layers[layer_index],
                      keras.layers.convolutional.Conv2D):
            print("\tlen(data)={}".format(len(data)))
            if len(data) > 3:
                _, p = scipy.stats.normaltest(data)
                print("\tData is normal distributed (p={})".format(p))
            sns.distplot(data, hist=False, norm_hist=False, kde=True, ax=ax,
                         label=str(layer_count))
        layer_count += 1
    ax.set_title('convolution activations')
    ax.legend()
    sns.plt.show()


def show_conv_weight_dist(model, small_thres=10**-6):
    f, (ax1, ax2) = plt.subplots(1, 2)
    layer_count = 0
    for layer in model.layers:
        if not isinstance(layer, keras.layers.convolutional.Conv2D):
            continue
        weights = layer.get_weights()

        # Filter
        print("{}: {} filter weights in {}th layer"
              .format(weights[0].shape,
                      reduce(operator.mul, weights[0].shape, 1),
                      layer_count))
        data = weights[0].flatten()
        data_small = np.array([el for el in data if abs(el) < small_thres])
        print("< {}: {}".format(small_thres, len(data_small)))
        sns.distplot(data, hist=False, norm_hist=True, kde=True, ax=ax1,
                     label=str(layer_count))

        # Bias
        print("{}: {} bias weights in {}th layer"
              .format(weights[1].shape,
                      reduce(operator.mul, weights[1].shape, 1),
                      layer_count))
        data = weights[1].flatten()
        data_small = np.array([el for el in data if abs(el) < small_thres])
        print("< {}: {}".format(small_thres, len(data_small)))
        sns.distplot(data, hist=False, norm_hist=False, kde=True, ax=ax2,
                     label=str(layer_count))
        layer_count += 1
    ax1.set_title('Filter weights')
    ax2.set_title('Bias weights')
    ax1.legend()
    ax2.legend()
    sns.plt.show()


def show_batchnorm_weight_dist(model):
    layer_count = 1
    f, (ax1, ax2) = plt.subplots(1, 2)
    layer_count = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            weights = layer.get_weights()
            data = weights[0].flatten()
            sns.distplot(data, hist=False, norm_hist=False, kde=True, ax=ax1,
                         label=str(layer_count))
            data = weights[1].flatten()
            sns.distplot(data, hist=False, norm_hist=False, kde=True, ax=ax2,
                         label=str(layer_count))
        layer_count += 1
    ax1.set_title('gamma weights')
    ax2.set_title('beta weights')
    ax1.legend()
    ax2.legend()
    sns.plt.show()


def main(data_module, model_path):
    model = load_model(model_path)

    print("## Activation analyzation")
    data = data_module.load_data()
    X_train = data['x_train']
    X_test = data['x_test']
    # y_train = data['y_train']
    # y_test = data['y_test']

    X_train = data_module.preprocess(X_train)
    X_test = data_module.preprocess(X_test)

    show_conv_act_distrib(model, X_train[0])

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
                        metavar="FILE")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Read YAML experiment definition file
    with open(args.filename, 'r') as stream:
        experiment_meta = yaml.load(stream)
    # Make paths absolute
    experiment_meta = make_paths_absolute(os.path.dirname(args.filename),
                                          experiment_meta)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(experiment_meta)
    dpath = experiment_meta['dataset']['script_path']
    sys.path.insert(1, os.path.dirname(dpath))
    data = imp.load_source('data', experiment_meta['dataset']['script_path'])
    artifacts_path = experiment_meta['train']['artifacts_path']
    model_path = os.path.join(artifacts_path,
                              experiment_meta['train']['model_output_fname'])
    main(data, model_path)
