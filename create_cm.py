#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the confusion matrices for this model."""

import logging
import sys
import yaml
from keras.models import load_model
import numpy as np
import io
import json
import imp
import pprint
import collections
import os
train_keras = imp.load_source('train_keras', "train/train_keras.py")
from train_keras import get_level, flatten_completely, filter_by_class
from train_keras import update_labels
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

from run_training import make_paths_absolute

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def _calculate_cm(model, X, y, n_classes):
    y_i = y.flatten()
    y_pred = model.predict(X)
    y_pred_i = y_pred.argmax(1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int)
    for i, j in zip(y_i, y_pred_i):
        cm[i][j] += 1
    return cm


def _write_cm(cm, path):
    # Serialize confusion matrix
    with io.open(path, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(cm.tolist(), sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False)
        outfile.write(to_unicode(str_))


def create_cm(model_path, data_module, artifacts_path, config):
    """Create confusion matrices."""
    # Load model
    model = load_model(model_path)
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

    # Calculate confusion matrix for training set
    cm = _calculate_cm(model, X_train, y_train, n_classes)
    correct_count = sum([cm[i][i] for i in range(n_classes)])
    acc = correct_count / float(cm.sum())
    print("Accuracy (Train): {:0.2f}% ({} of {} wrong)"
          .format(acc * 100, cm.sum() - correct_count, cm.sum()))
    _write_cm(cm, path=os.path.join(artifacts_path, 'cm-train.json'))

    # Calculate confusion matrix for test set
    cm = _calculate_cm(model, X_test, y_test, n_classes)
    correct_count = sum([cm[i][i] for i in range(n_classes)])
    acc = correct_count / float(cm.sum())
    print("Accuracy (Test): {:0.2f}% ({} of {} wrong)"
          .format(acc * 100, cm.sum() - correct_count, cm.sum()))
    _write_cm(cm, path=os.path.join(artifacts_path, 'cm-test.json'))

    # Calculate the accuracy for each sub-group
    if 'hierarchy_path' in config['dataset']:
        hierarchy = get_level(hierarchy, config['dataset']['subset'])
        for class_group in hierarchy:
            if isinstance(class_group, collections.Iterable):
                # calculate acc on this group
                print("Group: {}".format(class_group))
                correct = sum([cm[i][i] for i in class_group])
                all_ = sum(cm[i][j] for i in class_group for j in class_group)
                acc = correct / float(all_)
                print("\t{:0.2f}%".format(acc * 100))


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE.yaml",
                        required=True)
    return parser

if __name__ == '__main__':
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
    create_cm(model_path, data, artifacts_path, experiment_meta)
