#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility file for the MNIST dataset."""

from keras.utils.data_utils import get_file
import numpy as np
import os
import scipy.misc

n_classes = 10
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
WIDTH = 28
HEIGHT = 28
img_rows = 28
img_cols = 28
img_channels = 1

_mean_filename = "hasy-mean.npy"


def load_data(path='mnist.npz'):
    """
    Load the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz')
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    data = {'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'labels': labels
            }
    return data


def preprocess(x, subtact_mean=False):
    """Preprocess features."""
    x = x.astype('float32')
    x_new = np.zeros((len(x), 32, 32), dtype=np.uint8)
    for i, x_i in enumerate(x):
        x_new[i] = scipy.misc.imresize(x_i, (32, 32))
    x = x_new
    if not subtact_mean:
        x /= 255.0
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mean_path = os.path.join(dir_path, _mean_filename)
        mean_image = np.load(mean_path)
        x -= mean_image
        x /= 128.
    return x


if __name__ == '__main__':
    data = load_data()
    mean_image = np.mean(data['x_train'], axis=0)
    np.save(_mean_filename, mean_image)
    scipy.misc.imshow(mean_image.squeeze())
    x = preprocess(data['x_train'])
