#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility file for the Caltech-256 dataset.

See also
--------
http://www.vision.caltech.edu/Image_Datasets/Caltech256/
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

n_classes = 256
img_channels = 3
labels = None

# Design decision
img_rows = 224
img_cols = 224


_mean_filename = "caltech-256-{}-{}-mean.npy".format(img_rows, img_cols)


def serialize(filename, data):
    x_train_fname = "{}-x-train.npy".format(filename)
    x_test_fname = "{}-x-test.npy".format(filename)
    x_train = data['x_train']
    x_test = data['x_test']
    np.save(x_train_fname, data['x_train'])
    np.save(x_test_fname, data['x_test'])
    data['x_train'] = x_train_fname
    data['x_test'] = x_test_fname
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    data['x_train'] = x_train
    data['x_test'] = x_test


def deserialize(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    data['x_train'] = np.load(data['x_train'])
    data['x_test'] = np.load(data['x_test'])
    return data


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
    Load the Caltech-256 dataset.

    Returns
    -------
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = '256_ObjectCategories'
    origin = ('http://www.vision.caltech.edu/Image_Datasets/Caltech256/'
              '256_ObjectCategories.tar')
    path = get_file(dirname, origin=origin, untar=False)

    fname = "caltech-256-{}-{}-data.pickle".format(img_rows, img_cols)
    pickle_fpath = os.path.join(path, fname)

    if not os.path.isfile(pickle_fpath):
        classes = sorted(glob.glob("{}/*".format(path)),
                         key=lambda n: n.lower())
        classes = [el for el in classes if "257" not in el]
        classes = [el for el in classes if os.path.isdir(el)]
        globals()["labels"] = [os.path.basename(el) for el in classes]
        globals()["labels"] = [el.split(".")[1] for el in globals()["labels"]]
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
                                                            random_state=42,
                                                            stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=0.10,
                                                          random_state=42,
                                                          stratify=y_train)

        data = {'x_train': x_train, 'y_train': y_train,
                'x_val': x_val, 'y_val': y_val,
                'x_test': x_test, 'y_test': y_test,
                'labels': globals()["labels"]}
        serialize(pickle_fpath, data)
    else:
        data = deserialize(pickle_fpath)
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
    print(data)
    mean_image = np.mean(data['x_train'], axis=0)
    np.save(_mean_filename, mean_image)
    scipy.misc.imshow(mean_image)
    for img, label in zip(data['x_train'], data['y_train']):
        print(data['labels'][label])
        scipy.misc.imshow(img)
