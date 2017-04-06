#!/usr/bin/env python


"""Build an ensemble of CIFAR 100 Keras models."""

import logging
import sys
import yaml
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
from natsort import natsorted

import json
import imp
import os
train_keras = imp.load_source('train_keras', "train/train_keras.py")
from train_keras import get_level, flatten_completely, filter_by_class
from train_keras import update_labels

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def make_paths_absolute(dir_, conf):
    for i in range(len(conf["models"])):
        conf["models"][i] = os.path.join(dir_, conf["models"][i])
        conf["models"][i] = os.path.abspath(conf["models"][i])
    tmp = conf["dataset"]["script_path"]
    tmp = os.path.join(dir_, tmp)
    conf["dataset"]["script_path"] = os.path.abspath(tmp)

    if 'hierarchy_path' in conf["dataset"]:
        tmp = conf["dataset"]["hierarchy_path"]
        tmp = os.path.join(dir_, tmp)
        conf["dataset"]["hierarchy_path"] = os.path.abspath(tmp)
    return conf


def calculate_cm(y_true, y_pred, n_classes):
    """Calculate confusion matrix."""
    y_true_i = y_true.flatten()
    y_pred_i = y_pred.argmax(1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int)
    for i, j in zip(y_true_i, y_pred_i):
        cm[i][j] += 1
    return cm


def get_bin(x, n=0):
    """
    Get the binary representation of x.

    Parameters
    ----------
    x : int
    n : int
        Minimum number of digits. If x needs less digits in binary, the rest
        is filled with zeros.

    Returns
    -------
    str
    """
    return format(x, 'b').zfill(n)


def main(ensemble_fname):
    # Read YAML file
    with open(ensemble_fname) as data_file:
        config = yaml.load(data_file)
        config = make_paths_absolute(os.path.dirname(ensemble_fname),
                                     config)

    sys.path.insert(1, os.path.dirname(config['dataset']['script_path']))
    data_module = imp.load_source('data', config['dataset']['script_path'])

    # Load data
    data = data_module.load_data()
    X_train = data['x_train']
    X_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    X_train = data_module.preprocess(X_train)
    X_test = data_module.preprocess(X_test)

    # load hierarchy, if present
    if 'hierarchy_path' in config['dataset']:
        with open(config['dataset']['hierarchy_path']) as data_file:
            hierarchy = json.load(data_file)
        if 'subset' in config['dataset']:
            remaining_cls = get_level(hierarchy, config['dataset']['subset'])
            logging.info("Remaining classes: {}".format(remaining_cls))
            # Only do this if coarse is False:
            remaining_cls = flatten_completely(remaining_cls)
            data_module.n_classes = len(remaining_cls)
            X_train, y_train = filter_by_class(X_train, y_train, remaining_cls)
            X_test, y_test = filter_by_class(X_test, y_test, remaining_cls)
            old_cli2new_cli = {}
            for new_cli, old_cli in enumerate(remaining_cls):
                old_cli2new_cli[old_cli] = new_cli
            y_train = update_labels(y_train, old_cli2new_cli)
            y_test = update_labels(y_test, old_cli2new_cli)

    n_classes = data_module.n_classes
    logging.info("n_classes={}".format(n_classes))


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.10,
                                                      random_state=42)

    X_eval = X_test
    y_eval = y_test

    # Load models
    model_names = natsorted(config["models"])
    print("Ensemble of {} models ({})".format(len(model_names), model_names))
    models = []
    for model_path in model_names:
        print("Load model {}...".format(model_path))
        models.append(load_model(model_path))

    # Calculate confusion matrix
    # y_val_i = y_val.flatten()
    y_preds = []
    for model_path, model in zip(model_names, models):
        print("Evaluate model {}...".format(model_path))
        pred = model.predict(X_eval)
        y_preds.append(pred)

    for model_index, y_val_pred in enumerate(y_preds):
        cm = calculate_cm(y_eval, y_val_pred, n_classes)
        acc = sum([cm[i][i] for i in range(n_classes)]) / float(cm.sum())
        print("Cl #{} ({}): accuracy: {:0.2f}".format(model_index + 1,
                                                      model_names[model_index],
                                                      acc * 100))
    max_acc = 0.0
    for x in range(1, 2**len(y_preds) - 1):
        bitstring = get_bin(x, len(y_preds))
        y_preds_take = [p for p, i in zip(y_preds, bitstring) if i == "1"]
        y_val_pred = sum(y_preds_take) / bitstring.count("1")
        cm = calculate_cm(y_eval, y_val_pred, n_classes)
        acc = sum([cm[i][i] for i in range(n_classes)]) / float(cm.sum())
        if acc >= max_acc:
            print("Ensemble Accuracy: {:0.2f}% ({})".format(acc * 100,
                                                            bitstring))
            max_acc = acc

    Y_eval = np_utils.to_categorical(y_eval, n_classes)
    smoothed_lables = (y_val_pred + Y_eval) / 2
    np.save("smoothed_lables", smoothed_lables)


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="ensemble_fname",
                        help="File to describe an ensemble.",
                        metavar="FILE")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.ensemble_fname)
