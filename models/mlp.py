#!/usr/bin/env python

"""MLP model."""

import tflearn
from tflearn.layers.core import fully_connected


def inference(images, dataset_meta):
    """
    Build the CIFAR-10 model.

    Parameters
    ----------
    images : tensor
        Images returned from distorted_inputs() or inputs().
    dataset_meta : dict
        Has key 'n_classes'

    Returns
    -------
    logits
    """
    net = tflearn.layers.core.flatten(images, name='Flatten')
    y_conv = fully_connected(net, dataset_meta['n_classes'],
                             activation='softmax',
                             weights_init='truncated_normal',
                             bias_init='zeros',
                             regularizer=None,
                             weight_decay=0)
    return y_conv
