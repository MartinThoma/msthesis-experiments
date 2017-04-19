#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility file for the Tiny Imagenet dataset."""

from __future__ import absolute_import
# from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import os
import zipfile
import shutil
import sys
import glob
import scipy.misc
import PIL
from PIL import Image
import csv
from six.moves import cPickle as pickle

n_classes = 200
img_rows = 64
img_cols = 64
img_channels = 3
labels = None

_mean_filename = "tiny-imagenet-mean.npy"
train_img_paths = []
test_img_paths = []
val_img_paths = []
wnid2word = {}


def _maybe_download(url, fname, md5_hash):
    origin = os.path.join(url, fname)
    fpath = get_file(fname, origin=origin, untar=False, md5_hash=md5_hash)
    return fpath


def _maybe_extract(fpath, dirname, descend=True):
    path = os.path.dirname(fpath)
    untar_fpath = os.path.join(path, dirname)
    if not os.path.exists(untar_fpath):
        print('Extracting contents of "{}"...'.format(dirname))
        tfile = zipfile.ZipFile(fpath, 'r')
        try:
            tfile.extractall(untar_fpath)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(untar_fpath):
                if os.path.isfile(untar_fpath):
                    os.remove(untar_fpath)
                else:
                    shutil.rmtree(untar_fpath)
            raise
        tfile.close()
    if descend:
        dirs = [os.path.join(untar_fpath, o)
                for o in os.listdir(untar_fpath)
                if os.path.isdir(os.path.join(untar_fpath, o))]
        if len(dirs) != 1:
            print("Error, found not exactly one dir: {}".format(dirs))
            sys.exit(-1)
        return dirs[0]
    else:
        return untar_fpath


def _get_train_data(train_dir, global_name):
    x_train = []
    y_train = []
    classes = sorted(glob.glob("{}/*".format(train_dir)))
    for class_path in classes:
        class_name = os.path.basename(class_path)
        class_path_i = os.path.join(class_path, 'images')
        files = sorted(glob.glob("{}/*.JPEG".format(class_path_i)))
        globals()[global_name] += files
        for file_ in files:
            img = scipy.misc.imread(file_, mode='RGB')
            img = Image.fromarray(img)
            img = img.resize((img_rows, img_cols), PIL.Image.ANTIALIAS)
            arr = np.array(img, dtype=np.uint8)
            x_train.append(arr)
            y_train.append(globals()['labels'].index(class_name))
    x_train = np.array(x_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.int64)
    return x_train, y_train


def _get_test_data(dir_, global_name):
    x_train = []
    files = sorted(glob.glob("{}/*.JPEG".format(dir_)))
    globals()[global_name] += files
    for file_ in files:
        img = scipy.misc.imread(file_, mode='RGB')
        img = Image.fromarray(img)
        img = img.resize((img_rows, img_cols), PIL.Image.ANTIALIAS)
        arr = np.array(img, dtype=np.uint8)
        x_train.append(arr)
    x_train = np.array(x_train, dtype=np.uint8)
    return x_train


def _get_val_data(train_dir, global_name):
    x_train = []
    y_train = []

    # Read CSV file
    with open(os.path.join(train_dir, "val_annotations.txt"), 'r') as fp:
        reader = csv.reader(fp, delimiter='\t', quotechar='"')
        # filename, class, bb, bb, bb, bb
        annotations = [row for row in reader]

    fname2cl = {}
    for ann in annotations:
        fname2cl[ann[0]] = globals()['labels'].index(ann[1])

    # Read images
    train_dir = os.path.join(train_dir, "images")
    files = sorted(glob.glob("{}/*.JPEG".format(train_dir)))
    globals()[global_name] += files
    for file_ in files:
        img = scipy.misc.imread(file_, mode='RGB')
        arr = np.array(img, dtype=np.uint8)
        x_train.append(arr)
        y_train.append(fname2cl[os.path.basename(file_)])
    x_train = np.array(x_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.int64)
    return x_train, y_train


def load_data(config):
    """
    Load Tiny Imagenet dataset.

    Returns
    -------
    dict
    """
    # Download if not already done
    url = 'http://cs231n.stanford.edu/'
    fname = 'tiny-imagenet-200.zip'
    md5_sum = '90528d7ca1a48142e341f4ef8d21d0de'
    fpath = _maybe_download(url, fname, md5_sum)

    # Extract content if not already done
    main_dir = _maybe_extract(fpath, "Tiny-Imagenet")
    train_dir = os.path.join(main_dir, "train")
    test_dir = os.path.join(main_dir, "test")
    val_dir = os.path.join(main_dir, "val")

    # Load wnids.txt
    with open(os.path.join(main_dir, "wnids.txt")) as f:
        globals()['labels'] = f.read().splitlines()

    with open(os.path.join(main_dir, "words.txt"), 'r') as fp:
        reader = csv.reader(fp, delimiter='\t', quotechar='"')
        # filename, class, bb, bb, bb, bb
        words = [row for row in reader]
    wnid2word = {}
    for wnid, word in words:
        wnid2word[wnid] = word

    globals()['wnid2word'] = wnid2word

    # Get labeled training data
    pickle_fpath = os.path.join(train_dir, "data.pickle")
    if not os.path.exists(pickle_fpath):
        # Get train data
        x_train, y_train = _get_train_data(train_dir, "train_img_paths")
        x_test = _get_test_data(test_dir, "test_img_paths")
        x_val, y_val = _get_val_data(val_dir, "val_img_paths")

        data = {'x_train': x_train, 'y_train': y_train,
                'x_val': x_val, 'y_val': y_val,
                'x_test': x_test,
                'train_img_paths': globals()['train_img_paths'],
                'test_img_paths': globals()['test_img_paths'],
                'val_img_paths': globals()['val_img_paths']}

        # Store data as pickle to speed up later calls
        with open(pickle_fpath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)

    perm = np.random.permutation(len(data['x_train']))
    data['x_train'] = data['x_train'][perm]
    data['y_train'] = data['y_train'][perm]

    if K.image_dim_ordering() == 'th':
        data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
        data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

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
    config = {'dataset': {}}
    data = load_data(config)
    n = 10
    for x, y in zip(data['x_train'][:n], data['y_train'][:n]):
        print(globals()['wnid2word'][labels[y]])
        scipy.misc.imshow(x)
    mean_image = np.mean(data['x_train'], axis=0)
    np.save(_mean_filename, mean_image)
    scipy.misc.imshow(mean_image)
