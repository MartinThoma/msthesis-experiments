#!/usr/bin/env python

"""Compare the weights of layer i."""

import glob
from keras.models import load_model
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt


def main(directory, layer_index):
    """Orchestrate."""
    h5_fnames = glob.glob("{}/*chk*.h5".format(directory))
    h5_fnames = sorted(h5_fnames)
    print("{} models loaded.".format(len(h5_fnames)))

    changes = []

    # Start analysing weights
    last_weights = None
    for i, h5_fname in enumerate(h5_fnames):
        print("Load {}".format(h5_fname))
        model = load_model(h5_fname)
        weights = model.layers[layer_index].get_weights()[0].flatten()
        if last_weights is None:
            last_weights = weights
            continue
        else:
            change_epoch = []
            for w1, w2 in zip(last_weights, weights):
                change_epoch.append(abs(w1 - w2))
            changes.append(change_epoch)
            last_weights = weights
        if i > 20:
            break
    f, ax1 = plt.subplots(1, 1)
    p = sns.violinplot(data=changes, orient="v",
                       palette=sns.color_palette(palette="RdBu", n_colors=1),
                       ax=ax1)
    p.tick_params(labelsize=16)
    p.set_xlabel('Epoch', fontsize=20)
    p.set_ylabel('absolute weight update', fontsize=20)
    ax1.set_xticklabels(list(range(2, 2 + len(changes))))
    sns.plt.show()


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d",
                        dest="directory",
                        help="artifacts directory with .chk.h5 files",
                        metavar="DIR",
                        required=True)
    parser.add_argument("-i",
                        dest="layer_i",
                        default=2,
                        type=int,
                        help="layer for which the weights are compared",
                        required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.directory, args.layer_i)
