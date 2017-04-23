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


def main(directory):
    """Orchestrate."""
    h5_fnames = glob.glob("{}/*.chk.*.h5".format(directory))
    h5_fnames = natsort.natsorted(h5_fnames)
    print("{} models to be loaded.".format(len(h5_fnames)))

    changes = {}

    pickle_fname = "{}/weight-updates.pickle".format(directory)

    if not os.path.isfile(pickle_fname):
        # Print first models layers
        model = load_model(h5_fnames[0])
        for i, layer in enumerate(model.layers):
            print("{}: {}".format(i, layer))

        # Start analysing weights
        last_weights = {}
        recorded_epochs = 0
        for epoch, h5_fname in enumerate(h5_fnames):
            # if i < 80:
            #     continue
            print("Load {}".format(h5_fname))
            model = load_model(h5_fname)
            recorded_epochs += 1
            for layer_index, layer in enumerate(model.layers):
                if not isinstance(layer, keras.layers.convolutional.Conv2D):
                    continue
                weights = model.layers[layer_index].get_weights()[0].flatten()
                if layer_index not in last_weights:
                    last_weights[layer_index] = weights
                    changes[layer_index] = []
                    continue
                else:
                    change_epoch = []
                    for w1, w2 in zip(last_weights[layer_index], weights):
                        change_epoch.append(abs(w1 - w2))
                    changes[layer_index].append(change_epoch)
                    last_weights[layer_index] = weights
            # if recorded_epochs > 10:  # TODO: dev
                # break
        # serialize
        with open(pickle_fname, 'wb') as handle:
            pickle.dump(changes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_fname, 'rb') as handle:
            changes = pickle.load(handle)
        recorded_epochs = len(changes.items()[0][1])
        print("Recorded epochs={}".format(recorded_epochs))
    print("Done. plot stuff ({} lines)".format(len(changes.items())))
    f, ax1 = plt.subplots(1, 1)
    # ax1.set_xticklabels()
    # p = sns.violinplot(data=changes, orient="v",
    #                    palette=sns.color_palette(palette="RdBu", n_colors=1),
    #                    ax=ax1)
    # p.tick_params(labelsize=16)
    # p.set_xlabel('Epoch', fontsize=20)
    # p.set_ylabel('absolute weight update', fontsize=20)
    for layer_index, layer_changes in sorted(changes.items()):
        y = [np.array(epoch).mean() for epoch in layer_changes]  # quick fix
        # print(y[19])
        # y[19] = y[18]
        x = list(range(2, 2 + recorded_epochs))
        p = ax1.plot(x, y, label=layer_index)  # quick-fix
        color = p[0].get_color()
        for x in [40, 80, 100]:
            y2 = y[x]
            sns.plt.plot(x, y2, 'o', color='white', markersize=9)
            sns.plt.plot(x, y2, 'k', marker="$%s$" % layer_index, color=color,
                         markersize=7)
    plt.legend()
    ax1.set_xlabel('Epoch', fontsize=20)
    sns.plt.show()


def get_parser():
    """Get parser object for script compare_weights.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d",
                        dest="directory",
                        help="artifacts directory with .chk.h5 files",
                        metavar="DIR",
                        required=True)
    # parser.add_argument("-i",
    #                     dest="layer_i",
    #                     default=2,
    #                     type=int,
    #                     help="layer for which the weights are compared",
    #                     required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.directory)
