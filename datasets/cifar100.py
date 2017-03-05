#!/usr/bin/env python

from utils import maybe_download_and_extract, DataSet
import os
import cPickle
import scipy.misc
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'cifar100')
IMAGE_SIZE = 32
meta = {'n_classes': 100,
        'image_width': 32,
        'image_height': 32,
        'image_depth': 3}
train_data = []
test_data = []


def _unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = cPickle.load(fo)
    return dict_


def prepare(data_dir):
    maybe_download_and_extract(dest_directory=DATA_DIR, data_url=DATA_URL)
    directory = os.path.join(DATA_DIR, 'cifar-100-python')
    files = ['meta', 'test', 'train']
    files = [os.path.abspath(os.path.join(directory, f))
             for f in files]
    for f in files:
        print(f)
        if f.endswith("meta"):
            unpickled = _unpickle(f)
            # unpickled['coarse_label_names']  -- 20 labels
            meta['label_names'] = unpickled['fine_label_names']
            meta['labelname2index'] = {}
            for index, labelname in enumerate(meta['label_names']):
                meta['labelname2index'][labelname] = index
        elif f.endswith("test"):
            unpickled = _unpickle(f)
            tu = (unpickled['data'],
                  unpickled['fine_labels'],
                  unpickled['filenames'])
            for img, label, filename in zip(tu[0], tu[1], tu[2]):
                img = img.reshape((3, 32, 32)).transpose((1, 2, 0)).reshape(-1)
                test_data.append({'img': img,
                                  'label': label,
                                  'filename': filename})
        else:
            # ['data', 'labels', 'batch_label', 'filenames']
            unpickled = _unpickle(f)
            tu = (unpickled['data'],
                  unpickled['fine_labels'],
                  unpickled['filenames'])
            for img, label, filename in zip(tu[0], tu[1], tu[2]):
                img = img.reshape((3, 32, 32)).transpose((1, 2, 0)).reshape(-1)
                train_data.append({'img': img,
                                   'label': label,
                                   'filename': filename})


def inputs(eval_data, batch_size):
    """
    Construct input for CIFAR evaluation using the Reader ops.

    Parameters
    ----------
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-100 data directory.
    batch_size: Number of images per batch.

    Returns
    -------
    Generator with

    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    images = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3),
                      dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int32)
    if eval_data:
        dataset = test_data
    else:
        dataset = train_data
    i = 0
    while i + batch_size < len(dataset):
        for j in range(batch_size):
            images[j, :, :, :] = dataset[i + j]['img']
            labels[j] = dataset[i + j]['label']
        i += batch_size
        yield images, labels


def read_data_sets(validation_size=0.1):
    test_images = np.array([el['img'] for el in test_data])
    test_labels = np.array([el['label'] for el in test_data])
    test_labels = np.eye(meta['n_classes'])[test_labels]
    train_images = np.array([el['img'] for el in train_data])
    train_labels = np.array([el['label'] for el in train_data])
    train_labels = np.eye(meta['n_classes'])[train_labels]

    if 0 <= validation_size < 1.0:
        validation_size = int(validation_size * len(train_images))

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))
    # Shuffle data
    perm = np.arange(len(train_labels))
    np.random.shuffle(perm)
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    print(train_labels.shape)

    # Split training set in training and validation set
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, dtype=np.float32)
    validation = DataSet(validation_images,
                         validation_labels,
                         dtype=np.float32)
    test = DataSet(test_images, test_labels, dtype=np.float32)

    return base.Datasets(train=train, validation=validation, test=test)


def visualize(identifier):
    if identifier < len(train_data):
        img = train_data[identifier]['img']
        scipy.misc.imshow(img)
    else:
        identifier -= len(train_data)
        img = test_data[identifier]['img']
    img = img.reshape((meta['image_width'],
                       meta['image_height'],
                       meta['image_depth']))
    img = img.transpose((2, 0, 1))
    scipy.misc.imshow(img)


if __name__ == '__main__':
    prepare()
