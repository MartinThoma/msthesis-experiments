#!/usr/bin/env python

import glob
import json
import numpy as np


def main(directory):
    files_glob = "{}/*.json".format(directory)
    files = glob.glob(files_glob)
    train_metas = []
    for fname in files:
        with open(fname) as data_file:
            data = json.load(data_file)
        train_metas.append(data)

    training_times = []
    training_epochs = []
    for meta in train_metas:
        print(meta['HOST'])
        training_times.append(meta['training_time'])
        training_epochs.append(meta['epochs'])
    training_times = np.array(training_times)
    training_epochs = np.array(training_epochs)
    print("Epochs (mean={}, std={})".format(training_epochs.mean(),
                                            training_epochs.std()))
    print("Training time (mean={}, std={})".format(training_times.mean(),
                                                   training_times.std()))


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d",
                        dest="directory",
                        help="artifacts experiment directory with json files",
                        metavar="DIRECTORY")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.directory)
