#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility file for the STL-10 dataset.

See also
--------
Adam Coates, Honglak Lee, Andrew Y. Ng An Analysis of Single Layer Networks in
Unsupervised Feature Learning AISTATS, 2011.

https://cs.stanford.edu/~acoates/stl10/
"""

import numpy as np
from keras.utils.data_utils import get_file
import os
from sklearn.model_selection import train_test_split

n_classes = 10
img_rows = 96  # height
img_cols = 96  # width
img_channels = 3


_mean_filename = "stl10-mean.npy"

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH


def load_data():
    """
    Load STL10 dataset.

    Returns
    -------
    dict
    """
    # url of the binary data
    origin = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    dirname = 'stl10_binary'
    path = get_file(dirname, origin=origin, untar=True)

    # Load training data
    data_path = os.path.join(path, 'train_X.bin')
    label_path = os.path.join(path, 'train_y.bin')
    x_train = read_all_images(data_path)
    y_train = read_labels(label_path)

    # Load test data
    data_path = os.path.join(path, 'test_X.bin')
    label_path = os.path.join(path, 'test_y.bin')
    x_test = read_all_images(data_path)
    y_test = read_labels(label_path)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.10,
                                                      random_state=42,
                                                      stratify=y_train)

    return {'x_train': x_train, 'y_train': y_train,
            'x_val': x_val, 'y_val': y_val,
            'x_test': x_test, 'y_test': y_test}


def read_labels(path_to_labels):
    """
    Read labels from the STL-10 binary dataset.

    Parameters
    ----------
    path_to_labels : string
        path to the binary file containing labels from the STL-10 dataset

    Returns
    -------
    an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels - 1


def read_all_images(path_to_data):
    """
    Read images from the STL-10 binary dataset.

    Parameters
    ----------
    path_to_data : strubg
        The file containing the binary images from the STL-10 dataset

    Returns
    -------
    an array containing all the images
    """
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


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


if __name__ == "__main__":
    data = load_data()
    mean_image = np.mean(data['x_train'], axis=0)
    print("data['x_train'].shape={}".format(data['x_train'].shape))
    print("data['x_test'].shape={}".format(data['x_test'].shape))
    np.save(_mean_filename, mean_image)
    import scipy.misc
    scipy.misc.imshow(mean_image.squeeze())
