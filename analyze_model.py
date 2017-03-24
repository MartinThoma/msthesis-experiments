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
import os
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


def create_cm(model_path, data_module):
    """Create confusion matrices."""
    # Load model
    model = load_model(model_path)
    data = data_module.load_data()
    X_train = data['x_train']
    X_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    n_classes = data_module.n_classes

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # Calculate confusion matrix for training set
    cm = _calculate_cm(model, X_train, y_train, n_classes)
    acc = sum([cm[i][i] for i in range(n_classes)]) / float(cm.sum())
    print("Accuracy (Train): {:0.2f}%".format(acc * 100))
    _write_cm(cm, path='cm-train.json')

    # Calculate confusion matrix for test set
    cm = _calculate_cm(model, X_test, y_test, n_classes)
    acc = sum([cm[i][i] for i in range(n_classes)]) / float(cm.sum())
    print("Accuracy (Test): {:0.2f}%".format(acc * 100))
    _write_cm(cm, path='cm-test.json')


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
    create_cm(experiment_meta['train']['model_output_path'], data)
