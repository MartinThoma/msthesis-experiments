#!/usr/bin/env python

from __future__ import absolute_import
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import os
from sklearn.model_selection import train_test_split

n_classes = 10
img_rows = 32
img_cols = 32
img_channels = 3

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
          "horse", "ship", "truck"]

_mean_filename = "cifar-10-mean.npy"


def load_data(config):
    """
    Load CIFAR10 dataset.

    Returns
    -------
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.10,
                                                      random_state=42,
                                                      stratify=y_train)

    return {'x_train': x_train, 'y_train': y_train,
            'x_val': x_val, 'y_val': y_val,
            'x_test': x_test, 'y_test': y_test}


def preprocess(x, subtact_mean=False):
    """Preprocess features."""
    x = x.astype('float32')

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
    config = {'dataset': {}}
    data = load_data(config)
    print("Training data n={}".format(len(data['x_train'])))
    print("Validation data n={}".format(len(data['x_val'])))
    print("Test data n={}".format(len(data['x_test'])))
    mean_image = np.mean(data['x_train'], axis=0)
    np.save(_mean_filename, mean_image)
    import scipy.misc
    scipy.misc.imshow(mean_image)
    for img, label in zip(data['x_train'], data['y_train']):
        label = label[0]
        print(globals()['labels'][label])
        scipy.misc.imshow(img)
