#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility file for the Caltech-101 dataset.

See also
--------
http://www.vision.caltech.edu/Image_Datasets/Caltech101/
"""

import numpy as np
import os
import glob
import scipy.misc
from sklearn.model_selection import train_test_split
from six.moves import cPickle as pickle
from keras.utils.data_utils import get_file
import logging
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

n_classes = 101
img_rows = 32
img_cols = 32
img_channels = 3

labels = None

_mean_filename = "caltech-101-mean.npy"


def prepreprocess(img_path, res_width, res_height):
    """
    Make image to size width x height.

    Parameters
    ----------
    img_path : string
    res_width : int
    res_height : int

    Returns
    -------
    numpy array
    """
    try:
        im = scipy.misc.imread(img_path, mode='RGB')
    except:
        logging.error("Failed to load {}.".format(img_path))
        return np.zeros((res_height, res_width, 3), dtype=np.uint8)
    channels = 3
    im_height, im_width, _ = im.shape

    # Put on bigger canvas
    new_height = max(res_height, im_height)
    new_width = max(res_width, im_width)
    im_new = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    pad_top = (new_height - im_height) / 2
    pad_left = (new_width - im_width) / 2
    for channel in range(channels):
        im_new[pad_top:im_height + pad_top,
               pad_left:im_width + pad_left, channel] = im[:, :, channel]

    # Crop to correct aspect ratio
    factor = min(new_width / res_width, new_height / res_height)
    cut_height = factor * res_height
    cut_width = factor * res_width
    pad_top = (new_height - cut_height) / 2
    pad_left = (new_width - cut_width) / 2
    im = im[pad_top:pad_top + cut_height,
            pad_left:pad_left + cut_width, :]

    # scale
    im = scipy.misc.imresize(im, (res_height, res_width), interp='bilinear')
    return im


def load_data():
    """
    Load the Caltech-101 dataset.

    Parameters
    ----------
    label_mode: one of "fine", "coarse".

    Returns
    -------
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    Raises
    ------
    ValueError: in case of invalid `label_mode`.
    """
    dirname = '101_ObjectCategories'
    origin = ('http://www.vision.caltech.edu/Image_Datasets/Caltech101/'
              '101_ObjectCategories.tar.gz')
    path = get_file(dirname, origin=origin, untar=True)

    pickle_fpath = os.path.join(path, "caltech-101-data.pickle")

    if not os.path.isfile(pickle_fpath):
        classes = sorted(glob.glob("{}/*".format(path)),
                         key=lambda n: n.lower())
        classes = [el for el in classes if "BACKGROUND" not in el]
        globals()["labels"] = [os.path.basename(el) for el in classes]
        x = []
        y = []
        for i, class_path in enumerate(classes):
            class_path_glob = "{}/*.jpg".format(class_path)
            class_fnames = glob.glob(class_path_glob)
            print("{} in {}".format(len(class_fnames), class_path_glob))
            print("Start reading {}".format(globals()["labels"][i]))
            for class_fname in class_fnames:
                x.append(prepreprocess(class_fname, img_cols, img_rows))
                y.append(i)
        x = np.array(x, dtype=np.uint8)
        y = np.array(y, dtype=np.int64)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)

        data = {'x_train': x_train, 'y_train': y_train,
                'x_test': x_test, 'y_test': y_test,
                'labels': globals()["labels"]}

        with open(pickle_fpath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)
        globals()["labels"] = data['labels']
    return data


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
    data = load_data()
    mean_image = np.mean(data['x_train'], axis=0)
    np.save(_mean_filename, mean_image)
    scipy.misc.imshow(mean_image)
    for img, label in zip(data['x_train'], data['y_train']):
        print(data['labels'][label])
        scipy.misc.imshow(img)
