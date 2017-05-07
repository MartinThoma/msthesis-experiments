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

n_classes = 102
img_channels = 3
labels = None

nb_train_samples = 0
train_data_dir = None

# Design decision
img_rows = None
img_cols = None
_mean_filename = None


def initialize(config):
    """Load the Caltech-101 dataset."""
    just_resize = config['dataset']['just_resize']
    globals()["img_rows"] = config['dataset']['img_rows']
    globals()["img_cols"] = config['dataset']['img_cols']
    globals()["_mean_filename"] = ("caltech-101-{}-{}-mean.npy"
                                   .format(img_rows, img_cols))

    dirname = '101_ObjectCategories'
    origin = ('http://www.vision.caltech.edu/Image_Datasets/Caltech101/'
              '101_ObjectCategories.tar.gz')
    path = get_file(dirname, origin=origin, untar=True)
    new_path = "{}/../Caltech-101/{}-{}-{}/".format(path, img_rows, img_cols, just_resize)
    new_path = os.path.abspath(new_path)
    globals()["train_data_dir"] = os.path.join(new_path, "train")
    globals()["validation_data_dir"] = os.path.join(new_path, "val")

    fname = ("caltech-101-{}-{}-{}-102-data.pickle"
             .format(img_rows, img_cols, just_resize))
    pickle_fpath = os.path.join(new_path, fname)
    print("Pickle: {}".format(pickle_fpath))

    i2dirname = {}

    if not os.path.isfile(pickle_fpath):
        classes = sorted(glob.glob("{}/*".format(path)),
                         key=lambda n: n.lower())
        if (("background" in config['dataset'] and
             not config['dataset']["background"])):
            classes = [el for el in classes if "BACKGROUND" not in el]
        classes = [el for el in classes if os.path.isdir(el)]
        assert len(classes) == globals()["n_classes"]
        globals()["labels"] = [os.path.basename(el) for el in classes]
        assert len(globals()["labels"]) == globals()["n_classes"]
        x_fnames = []
        y = []
        for i, class_path in enumerate(classes):
            i2dirname[i] = os.path.basename(class_path)
            class_path_glob = "{}/*.jpg".format(class_path)
            class_fnames = glob.glob(class_path_glob)
            print("{}: Reading {} of {} in {}..."
                  .format(i,
                          len(class_fnames),
                          globals()["labels"][i],
                          class_path_glob))
            for class_fname in class_fnames:
                x_fnames.append(class_fname)
                y.append(i)
        x_trainf, x_testf, y_trainf, y_testf = train_test_split(x_fnames, y,
                                                                test_size=0.33,
                                                                random_state=0,
                                                                stratify=y)
        tmp = train_test_split(x_trainf, y_trainf,
                               test_size=0.10,
                               random_state=0,
                               stratify=y_trainf)
        x_trainf, x_valf, y_trainf, y_valf = tmp
        # Test data
        x_test = []
        for class_fname in x_testf:
            gen = prepreprocess(class_fname, img_cols, img_rows,
                                just_resize, only_center=True)
            for gen_el in gen:
                x_test.append(gen_el)
        x_test = np.array(x_test, dtype=np.uint8)
        y_test = np.array(y_testf, dtype=np.int64)

        # Validation data
        x_val = []
        for class_fname in x_valf:
            gen = prepreprocess(class_fname, img_cols, img_rows,
                                just_resize, only_center=True)
            for gen_el in gen:
                x_val.append(gen_el)
        x_val = np.array(x_val, dtype=np.uint8)
        y_val = np.array(y_valf, dtype=np.int64)

        # Training data
        for class_fname, y in zip(x_trainf, y_trainf):
            class_name = os.path.basename(i2dirname[y])
            new_classpath = os.path.join(globals()["train_data_dir"],
                                         class_name)
            if not os.path.exists(new_classpath):
                print("create {}".format(new_classpath))
                os.makedirs(new_classpath)
            gen = prepreprocess(class_fname, img_cols, img_rows,
                                just_resize, only_center=False)
            for gen_el in gen:
                i = globals()["nb_train_samples"]
                scipy.misc.imsave('{}/{}.jpg'.format(new_classpath, i),
                                  gen_el)
                globals()["nb_train_samples"] += 1

        data = {'x_val': x_val, 'y_val': y_val,
                'x_test': x_test, 'y_test': y_test,
                'labels': globals()["labels"],
                'nb_train_samples': globals()["nb_train_samples"]}

        with open(pickle_fpath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)
        print(data['x_val'].shape)
        globals()["labels"] = data['labels']
        globals()["nb_train_samples"] = data['nb_train_samples']
    return data


def prepreprocess(img_path, res_width, res_height, just_resize=False,
                  only_center=True):
    """
    Make image to size width x height.

    Parameters
    ----------
    img_path : string
    res_width : int
    res_height : int
    just_resize : bool

    Returns
    -------
    numpy array
    """
    try:
        im = scipy.misc.imread(img_path, mode='RGB')
    except:
        logging.error("Failed to load {}.".format(img_path))
        yield np.zeros((res_height, res_width, 3), dtype=np.uint8)

    if not just_resize:
        channels = 3
        im_height, im_width, _ = im.shape

        # Put on bigger canvas if image is too small
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
        im_cut = im[pad_top:pad_top + cut_height,
                    pad_left:pad_left + cut_width, :]
        # scale
        im_cut = scipy.misc.imresize(im_cut, (res_height, res_width),
                                     interp='bilinear')
        yield im_cut
        if not only_center:
            x_pad_step = int((new_height - cut_height) / 2.)
            x_pad_step = max(x_pad_step, 5)
            y_pad_step = int((new_width - cut_width) / 2.)
            y_pad_step = max(y_pad_step, 5)
            for pad_top in range(0, new_height - cut_height, x_pad_step):
                for pad_left in range(0, new_width - cut_width, y_pad_step):
                    im_cut = im[pad_top:pad_top + cut_height,
                                pad_left:pad_left + cut_width, :]
                    # scale
                    im_cut = scipy.misc.imresize(im_cut, (res_height, res_width),
                                                 interp='bilinear')
                    yield im_cut
    else:
        # scale
        im = scipy.misc.imresize(im, (res_height, res_width),
                                 interp='bilinear')
        yield im


def load_data(config):
    """
    Load the Caltech-101 dataset.

    Returns
    -------
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    just_resize = config['dataset']['just_resize']
    globals()["img_rows"] = config['dataset']['img_rows']
    globals()["img_cols"] = config['dataset']['img_cols']
    globals()["_mean_filename"] = ("caltech-101-{}-{}-mean.npy"
                                   .format(img_rows, img_cols))

    dirname = '101_ObjectCategories'
    origin = ('http://www.vision.caltech.edu/Image_Datasets/Caltech101/'
              '101_ObjectCategories.tar.gz')
    path = get_file(dirname, origin=origin, untar=True)

    fname = ("caltech-101-{}-{}-{}-102-data.pickle"
             .format(img_rows, img_cols, just_resize))
    pickle_fpath = os.path.join(path, fname)

    if not os.path.isfile(pickle_fpath):
        classes = sorted(glob.glob("{}/*".format(path)),
                         key=lambda n: n.lower())
        if (("background" in config['dataset'] and
             not config['dataset']["background"])):
            classes = [el for el in classes if "BACKGROUND" not in el]
        classes = [el for el in classes if os.path.isdir(el)]
        assert len(classes) == globals()["n_classes"]
        globals()["labels"] = [os.path.basename(el) for el in classes]
        assert len(globals()["labels"]) == globals()["n_classes"]
        x = []
        y = []
        for i, class_path in enumerate(classes):
            class_path_glob = "{}/*.jpg".format(class_path)
            class_fnames = glob.glob(class_path_glob)
            print("{}: Reading {} of {} in {}..."
                  .format(i,
                          len(class_fnames),
                          globals()["labels"][i],
                          class_path_glob))
            for class_fname in class_fnames:
                gen = prepreprocess(class_fname, img_cols, img_rows,
                                    just_resize)
                for gen_el in gen:
                    x.append(gen_el)
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
    config = {'dataset': {'just_resize': False,
                          'img_cols': 240,
                          'img_rows': 180}}
    data = load_data(config)
    print("len(data['x_train'])={}".format(len(data['x_train'])))
    mean_image = np.mean(data['x_train'], axis=0)
    np.save(_mean_filename, mean_image)
    scipy.misc.imshow(mean_image)
    for img, label in zip(data['x_train'], data['y_train']):
        print(data['labels'][label])
        scipy.misc.imshow(img)
