#!/usr/bin/env python

"""Utility functions for loading Computer Vision datasets."""

import os
import sys
import tarfile
from six.moves import urllib
# from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import numpy as np


def maybe_download_and_extract(dest_directory, data_url):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print(('Successfully downloaded {filename} '
               '({bytes} bytes)').format(filename=filename,
                                         bytes=statinfo.st_size))
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    return dest_directory


class DataSet(object):
    """DataSet from tensorflow.contrib.learn.python.learn.datasets.mnist."""

    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=dtypes.float32):
        """
        Construct a DataSet.

        `dtype` can be either `uint8` to leave the input as `[0, 255]`, or
        `float32` to rescale into `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(('Invalid image dtype %r, expected uint8 or '
                             'float32') % dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   labels.shape))
        self._num_examples = images.shape[0]
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
