#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility file for the Microsoft/Assira/ Kaggle Cats and Dogs dataset.

See also
--------
https://www.microsoft.com/en-us/download/details.aspx?id=54765
https://www.kaggle.com/c/dogs-vs-cats/leaderboard
"""

import numpy as np
import logging
import os
import sys
import glob
import scipy.misc
from sklearn.model_selection import train_test_split
from six.moves import cPickle as pickle

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

n_classes = 2
img_channels = 3

labels = ['cat', 'dog']

# Design decision
img_rows = None  # height
img_cols = None  # width
_mean_filename = None



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


def load_data(config):
    """
    Load Cat Dog dataset.

    Returns
    -------
    dict
    """
    globals()["img_rows"] = config['dataset']['img_rows']
    globals()["img_cols"] = config['dataset']['img_cols']
    globals()["_mean_filename"] = ("caltech-101-{}-{}-mean.npy"
                                   .format(img_rows, img_cols))
    # url of the binary data
    cache_dir = os.path.expanduser(os.path.join('~', '.keras/datasets'))
    path = os.path.join(cache_dir, 'kagglecatsanddogs_3367a')
    if not os.path.isdir(path):
        logging.info("Please download the Kaggle Cats and Dogs dataset from "
                     "Microsoft to {} and extract it there."
                     .format(path))
        sys.exit(-1)
    path = os.path.join(path, "PetImages")
    pickle_fpath = os.path.join(path,
                                "cat-dog-data-{}-{}.pickle"
                                .format(config['dataset']['img_rows'],
                                        config['dataset']['img_cols']))

    if not os.path.isfile(pickle_fpath):
        # Load data
        cat_path_glob = "{}/Cat/*.jpg".format(path)
        cats_fnames = glob.glob(cat_path_glob)
        dogs_path_glob = "{}/Dog/*.jpg".format(path)
        dogs_fnames = glob.glob(dogs_path_glob)
        print("{} in {}".format(len(cats_fnames), cat_path_glob))
        print("{} in {}".format(len(dogs_fnames), dogs_path_glob))

        # Make np arrays
        x = np.zeros((len(dogs_fnames) + len(cats_fnames),
                      img_rows, img_cols, 3), dtype=np.uint8)
        y = np.zeros((len(dogs_fnames) + len(cats_fnames), 1), dtype=np.uint64)
        print("Start reading dogs")
        for i, dog_fname in enumerate(dogs_fnames):
            x[i, :, :, :] = prepreprocess(dog_fname, img_cols, img_rows)
            y[i] = 1
        print("Start reading cats")
        for i, cat_fname in enumerate(cats_fnames, start=len(dogs_fnames)):
            x[i, :, :, :] = prepreprocess(cat_fname, img_cols, img_rows)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42,
                                                            stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=0.10,
                                                          random_state=42,
                                                          stratify=y_train)

        # both = cats_fnames + dogs_fnames
        # from random import shuffle
        # shuffle(both)
        # for el in both:
        #     prepreprocess(el, img_cols, img_rows)

        data = {'x_train': x_train, 'y_train': y_train,
                'x_val': x_val, 'y_val': y_val,
                'x_test': x_test, 'y_test': y_test}

        with open(pickle_fpath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)

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


if __name__ == "__main__":
    config = {'dataset': {'just_resize': False,
                          'img_cols': 32,
                          'img_rows': 32}}
    data = load_data(config)
    print("Training data n={}".format(len(data['x_train'])))
    print("Validation data n={}".format(len(data['x_val'])))
    print("Test data n={}".format(len(data['x_test'])))
    mean_image = np.mean(data['x_train'], axis=0)
    print("data['x_train'].shape={}".format(data['x_train'].shape))
    print("data['x_test'].shape={}".format(data['x_test'].shape))
    np.save(_mean_filename, mean_image)
    scipy.misc.imshow(mean_image.squeeze())
