#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Analyze a cifar100 keras model."""

from keras.models import load_model
from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
import numpy as np
import json
import io
import matplotlib.pyplot as plt
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

n_classes = 100


def plot_cm(cm, zero_diagonal=False):
    """Plot a confusion matrix."""
    n = len(cm)
    size = int(n / 4.)
    fig = plt.figure(figsize=(size, size), dpi=80, )
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(cm), cmap=plt.cm.viridis,
                    interpolation='nearest')
    width, height = cm.shape
    fig.colorbar(res)
    plt.savefig('confusion_matrix.png', format='png')


def load_data():
    """Load data."""
    (X, y), (X_test, y_test) = cifar100.load_data()

    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.10,
                                                      random_state=42)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_val /= 255
    X_test /= 255
    return X_train, X_val, X_test, y_test, y_val, y_test


def main(model_path):
    # Load model
    model = load_model(model_path)
    X_train, X_val, X_test, y_test, y_val, y_test = load_data()

    # Calculate confusion matrix
    y_val_i = y_val.flatten()
    y_val_pred = model.predict(X_val)
    y_val_pred_i = y_val_pred.argmax(1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int)
    for i, j in zip(y_val_i, y_val_pred_i):
        cm[i][j] += 1

    acc = sum([cm[i][i] for i in range(100)]) / float(cm.sum())
    print("Accuracy: %0.4f" % acc)

    # Create plot
    plot_cm(cm)

    # Serialize confusion matrix
    with io.open('cm.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(cm.tolist(),
                          indent=4, sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False)
        outfile.write(to_unicode(str_))


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="model_path",
                        help="Path to a Keras model file",
                        metavar="model.h5",
                        required=True)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.model_path)
