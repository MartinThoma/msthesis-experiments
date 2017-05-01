#!/usr/bin/env python

"""Compare the weights of layer i."""

import glob
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import natsort
import os
import numpy as np
import cPickle as pickle
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Greens_d", 8))


def main(directory, ignore_first, max_epochs, agg_func):
    """Orchestrate."""
    h5_fnames = glob.glob("{}/*.chk.*.h5".format(directory))
    h5_fnames = natsort.natsorted(h5_fnames)
    print("{} models could be loaded.".format(len(h5_fnames)))

    aggregation_functions = [('mean', np.mean),
                             ('max', np.max),
                             ('min', np.min),
                             ('sum', np.sum)]
    changes_small = {}
    for name, _ in aggregation_functions:
        changes_small[name] = {}

    pickle_fname = "{}/weight-updates-{}-epochs.pickle".format(directory,
                                                               max_epochs)

    if not os.path.isfile(pickle_fname):
        # Print first models layers
        model = load_model(h5_fnames[0])
        for i, layer in enumerate(model.layers):
            print("{}: {}".format(i, layer))

        # Start analysing weights
        last_weights = {}
        recorded_epochs = 0
        for epoch, h5_fname in enumerate(h5_fnames):
            print("Load {}".format(h5_fname))
            model = load_model(h5_fname)
            recorded_epochs += 1
            for layer_index, layer in enumerate(model.layers):
                if not isinstance(layer, keras.layers.convolutional.Conv2D):
                    continue
                weights = model.layers[layer_index].get_weights()[0].flatten()
                if layer_index not in last_weights:
                    last_weights[layer_index] = weights
                    for name, _ in aggregation_functions:
                        changes_small[name][layer_index] = []
                    continue
                else:
                    change_epoch = []
                    for w1, w2 in zip(last_weights[layer_index], weights):
                        change_epoch.append(abs(w1 - w2))
                    for name, func in aggregation_functions:
                        tmp = func(change_epoch)
                        changes_small[name][layer_index].append(tmp)
                    last_weights[layer_index] = weights
            if recorded_epochs > max_epochs:
                break
        # serialize
        with open(pickle_fname, 'wb') as handle:
            pickle.dump(changes_small, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_fname, 'rb') as handle:
            changes_small = pickle.load(handle)
    recorded_epochs = len(changes_small['mean'].items()[0][1])
    print("Recorded epochs={}".format(recorded_epochs))
    print("Done. plot stuff ({} lines)".format(len(changes_small[agg_func])))
    f, ax1 = plt.subplots(1, 1)
    labels = [1, 3, 5, 7, 9, 11, 13, 15]  # Adjust
    i = 0
    for layer_index, layer_changes in sorted(changes_small[agg_func].items()):
        y = [np.array(epoch).mean() for epoch in layer_changes]  # Adjust

        if ignore_first:
            y[0] = y[1]
        x = list(range(recorded_epochs))
        p = ax1.plot(x, y, label="Layer {}".format(labels[i]))  # layer_index
        color = p[0].get_color()
        if len(y) > 100:
            for x in [40, 80, 100]:
                y2 = y[x]
                sns.plt.plot(x, y2, 'o', color='white', markersize=9)
                sns.plt.plot(x, y2, 'k', marker="${}$".format(labels[i]),
                             color=color,
                             markersize=7)
        i += 1
    sns.plt.legend()
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Absolute weight change ({})'.format(agg_func),
                   fontsize=20,  # Adjust
                   rotation=90, labelpad=20)
    sns.plt.show()


def get_parser():
    """Get parser object for compare_weights.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d",
                        dest="directory",
                        help="artifacts directory with .chk.h5 files",
                        metavar="DIR",
                        required=True)
    parser.add_argument("-i",
                        action="store_true",
                        dest="ignore_first",
                        default=False,
                        help="ignore first")
    parser.add_argument("--max",
                        dest="max_epochs",
                        default=1000,
                        type=int,
                        help="maximum number of epochs to be handled")
    parser.add_argument("--agg",
                        dest="aggregation_function",
                        help="Aggregation function to apply",
                        choices=['mean', 'max', 'min', 'sum'],
                        default='mean')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.directory, args.ignore_first, args.max_epochs,
         args.aggregation_function)
