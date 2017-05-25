#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility file for the GTSDB dataset.

See http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads for
details.
"""

from __future__ import absolute_import
from keras.utils.data_utils import get_file
from keras import backend as K
import PIL
from PIL import Image
import numpy as np
import os
import glob
import zipfile
import shutil
import csv
import sys
import random
random.seed(0)
from six.moves import cPickle as pickle
from sklearn.model_selection import train_test_split

_mean_filename = "gtsrb-mean.npy"


labels = ['speed limit 20 (prohibitory)',
          'speed limit 30 (prohibitory)',
          'speed limit 50 (prohibitory)',
          'speed limit 60 (prohibitory)',
          'speed limit 70 (prohibitory)',
          'speed limit 80 (prohibitory)',
          'restriction ends 80 (other)',
          'speed limit 100 (prohibitory)',
          'speed limit 120 (prohibitory)',
          'no overtaking (prohibitory)',
          'no overtaking (trucks) (prohibitory)',
          'priority at next intersection (danger)',
          'priority road (other)',
          'give way (other)',
          'stop (other)',
          'no traffic both ways (prohibitory)',
          'no trucks (prohibitory)',
          'no entry (other)',
          'danger (danger)',
          'bend left (danger)',
          'bend right (danger)',
          'bend (danger)',
          'uneven road (danger)',
          'slippery road (danger)',
          'road narrows (danger)',
          'construction (danger)',
          'traffic signal (danger)',
          'pedestrian crossing (danger)',
          'school crossing (danger)',
          'cycles crossing (danger)',
          'snow (danger)',
          'animals (danger)',
          'restriction ends (other)',
          'go right (mandatory)',
          'go left (mandatory)',
          'go straight (mandatory)',
          'go right or straight (mandatory)',
          'go left or straight (mandatory)',
          'keep right (mandatory)',
          'keep left (mandatory)',
          'roundabout (mandatory)',
          'restriction ends (overtaking) (other)',
          'restriction ends (overtaking (trucks)) (other)']
labels_short = ['20',
                '30',
                '50',
                '60',
                '70',
                '80',
                'restriction ends 80',
                '100',
                '120',
                'no overtaking',
                'no overtaking (tr)',
                'priority at next int',
                'priority road',
                'give way',
                'stop',
                'no traffic both ways',
                'no trucks',
                'no entry',
                'danger',
                'bend left',
                'bend right',
                'bend',
                'uneven road',
                'slippery road',
                'road narrows',
                'construction',
                'traffic signal',
                'pedestrian crossing',
                'school crossing',
                'cycles crossing',
                'snow',
                'animals',
                'restriction ends',
                'go right',
                'go left',
                'go straight',
                'go right or straight',
                'go left or straight',
                'keep right',
                'keep left',
                'roundabout',
                'restriction ends (overt)',
                'restriction ends (overt tr)']
n_classes = len(labels)
img_rows = 32
img_cols = 32
img_channels = 3
train_img_paths = []
train_img_rois = {}


def _load_csv(filepath, delimiter=',', quotechar="'"):
    """
    Load a CSV file.

    Parameters
    ----------
    filepath : str
        Path to a CSV file
    delimiter : str, optional
    quotechar : str, optional

    Returns
    -------
    list of dicts : Each line of the CSV file is one element of the list.
    """
    data = []
    csv_dir = os.path.dirname(filepath)
    with open(filepath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile,
                                delimiter=delimiter,
                                quotechar=quotechar)
        for row in reader:
            for el in ['path', 'path1', 'path2']:
                if el in row:
                    row[el] = os.path.abspath(os.path.join(csv_dir, row[el]))
            data.append(row)
    return data


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


def load_data(config):
    """
    Load GTSDB dataset.

    If Tensorflow backend is used: (index, height, width, channels)

    Returns
    -------
    dict
    """
    # Download if not already done
    url = 'http://benchmark.ini.rub.de/Dataset/'
    fname_train = 'GTSRB_Final_Training_Images.zip'
    fname_test = 'GTSRB_Final_Test_Images.zip'
    fname_test_gt = 'GTSRB_Final_Test_GT.zip'
    md5_train = 'f33fd80ac59bff73c82d25ab499e03a3'
    md5_test = 'c7e4e6327067d32654124b0fe9e82185'
    md5_test_gt = 'fe31e9c9270bbcd7b84b7f21a9d9d9e5'
    fpath_train = _maybe_download(url, fname_train, md5_train)
    fpath_test = _maybe_download(url, fname_test, md5_test)
    fpath_test_gt = _maybe_download(url, fname_test_gt, md5_test_gt)

    # Extract content if not already done
    train_dir = _maybe_extract(fpath_train, "GTSRB_Train")
    train_dir = os.path.join(train_dir, "Final_Training/Images")
    test_diro = _maybe_extract(fpath_test, "GTSRB_Test")
    test_dir = os.path.join(test_diro, "Final_Test/Images")
    test_dir_gt = _maybe_extract(fpath_test_gt, "GTSRB_Test_GT", descend=False)

    # Get labeled training data
    pickle_fpath = os.path.join(test_diro, "data.pickle")
    if not os.path.exists(pickle_fpath):
        # Get train data
        x_train = []
        y_train = []
        classes = sorted(glob.glob("{}/*".format(train_dir)))
        for class_path in classes:
            classi = int(os.path.basename(class_path))
            files = sorted(glob.glob("{}/*.ppm".format(class_path)))
            globals()["train_img_paths"] += files
            for file_ in files:
                with Image.open(file_) as img:
                    img = img.resize((32, 32), PIL.Image.ANTIALIAS)
                    arr = np.array(img)
                    x_train.append(arr)
                y_train.append(classi)
        x_train = np.array(x_train, dtype=np.uint8)
        y_train = np.array(y_train, dtype=np.int64)

        # Get test data
        x_test = []
        onlyfiles = sorted(glob.glob("{}/*.ppm".format(test_dir)))
        for f in onlyfiles:
            with Image.open(f) as img:
                img = img.resize((32, 32), PIL.Image.ANTIALIAS)
                arr = np.array(img)
                x_test.append(arr)
        x_test = np.array(x_test, dtype=np.uint8)

        # Get test GT
        y_test = []
        print(test_dir_gt)
        filepath_csv = os.path.join(test_dir_gt, "GT-final_test.csv")
        with open(filepath_csv, 'r') as fp:
            reader = csv.reader(fp, delimiter=';', quotechar='"')
            next(reader, None)  # skip the headers
            y_test = [int(row[7]) for row in reader]
        y_test = np.array(y_test, dtype=np.int64)

        data = {'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test}

        # Store data as pickle to speed up later calls
        with open(pickle_fpath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)

    if K.image_dim_ordering() == 'th':
        data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
        data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

    x_train, x_val, y_train, y_val = train_test_split(data['x_train'],
                                                      data['y_train'],
                                                      test_size=0.10,
                                                      random_state=42,
                                                      stratify=data['y_train'])

    return {'x_train': x_train, 'y_train': y_train,
            'x_val': x_val, 'y_val': y_val,
            'x_test': data['x_test'], 'y_test': data['y_test']}

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
    print("Training data n={}".format(len(data['x_train'])))
    print("Validation data n={}".format(len(data['x_val'])))
    print("Test data n={}".format(len(data['x_test'])))
    mean_image = np.mean(data['x_train'], axis=0)
    np.save(_mean_filename, mean_image)
    import scipy.misc
    scipy.misc.imshow(mean_image)
    for img, label in zip(data['x_train'], data['y_train']):
        print(globals()['labels'][label])
        scipy.misc.imshow(img)
