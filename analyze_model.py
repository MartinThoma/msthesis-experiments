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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Make mosaik.

    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    imshape = imgs.shape[1:]

    mosaic = np.zeros((nrows * imshape[0] + (nrows - 1) * border,
                       ncols * imshape[1] + (ncols - 1) * border,
                       imgs.shape[-1]),
                      dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nrows * ncols):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def get_activations(model, layer_index, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[layer_index].output, ])
    activations = get_activations([X_batch, 0])
    return activations


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

    for layer_index in range(len(model.layers)):
        # layer_index = 3
        print(model.layers[layer_index])
        act = get_activations(model, layer_index, X_train[:32])[0]
        for i in [0, 10, 30]:
            scipy.misc.imshow(X_train[i])
            plt.figure(figsize=(15, 15))
            plt.title('conv1 weights')
            mosaik = make_mosaic(act[i], 8, 4)
            scipy.misc.imshow(mosaik)
            data = act[i].flatten()
            ax = plt.axes()
            sns.distplot(data, hist=True, norm_hist=False, kde=False, ax=ax)
            ax.set_title('activation for input {}'.format(i))
            sns.plt.show()

    print("## Weight analyzation")
    layer_count = 1
    for layer in model.layers:
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            weights = layer.get_weights()
            print("--BatchNorm:")
            data = weights[0].flatten()
            ax = plt.axes()
            sns.distplot(data, hist=True, norm_hist=False, kde=False, ax=ax)
            ax.set_title('gamma weights')
            sns.plt.show()
            #
            data = weights[1].flatten()
            ax = plt.axes()
            sns.distplot(data, hist=True, norm_hist=False, kde=False, ax=ax)
            ax.set_title('beta weights')
            sns.plt.show()
            print(weights[2].shape)
            print(weights[3].shape)
            print("--")
        if not isinstance(layer, keras.layers.convolutional.Conv2D):
            continue
        weights = layer.get_weights()

        # Filter
        print("{}: {} filter weights in {}th layer"
              .format(weights[0].shape,
                      reduce(operator.mul, weights[0].shape, 1),
                      layer_count))
        data = weights[0].flatten()
        data = np.array([el for el in data if abs(el) < 10**-5])
        print("small: {}".format(len(data)))
        if len(data) < 2:
            continue
        ax = plt.axes()
        sns.distplot(data, hist=True, norm_hist=False, kde=False, ax=ax)
        ax.set_title('Filter weights')
        sns.plt.show()

        # Bias
        continue
        print("{}: {} bias weights in {}th layer"
              .format(weights[1].shape,
                      reduce(operator.mul, weights[1].shape, 1),
                      layer_count))
        data = weights[1].flatten()
        data = np.array([el for el in data if abs(el) < 10**-3])
        ax = plt.axes()
        sns.distplot(data, hist=True, norm_hist=False, kde=False, ax=ax)
        ax.set_title('Bias weights')
        sns.plt.show()
        layer_count += 1
        # sys.exit()


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="Keras model file to be analyzed",
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
