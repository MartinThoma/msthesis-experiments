#!/usr/bin/env python

"""MLP model."""

import tensorflow as tf
import tflearn
from tflearn.layers.core import fully_connected
batch_normalization = tf.layers.batch_normalization


def inference(images, dataset_meta):
    """
    Build a tiny CNN model.

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
    net = tf.reshape(images, [-1,
                              dataset_meta['image_width'],
                              dataset_meta['image_height'],
                              dataset_meta['image_depth']])
    net = batch_normalization(net,
                              beta_initializer=tf.zeros_initializer(),
                              moving_variance_initializer=tf.ones_initializer())
    net = tflearn.layers.conv.conv_2d(net,
                                      nb_filter=64,
                                      filter_size=5,
                                      activation='prelu',
                                      strides=1,
                                      weight_decay=0.0)
    net = tflearn.layers.conv.max_pool_2d(net,
                                          kernel_size=3,
                                          strides=2,
                                          padding='same',
                                          name='MaxPool2D')
    net = batch_normalization(net,
                              beta_initializer=tf.zeros_initializer(),
                              moving_variance_initializer=tf.ones_initializer())
    net = tflearn.layers.conv.conv_2d(net,
                                      nb_filter=64,
                                      filter_size=5,
                                      activation='prelu',
                                      strides=1,
                                      weight_decay=0.0)
    net = tflearn.layers.conv.max_pool_2d(net,
                                          kernel_size=3,
                                          strides=2,
                                          padding='same',
                                          name='MaxPool2D')
    net = batch_normalization(net,
                              beta_initializer=tf.zeros_initializer(),
                              moving_variance_initializer=tf.ones_initializer())
    net = tflearn.layers.conv.conv_2d(net,
                                      nb_filter=8,
                                      filter_size=3,
                                      activation='prelu',
                                      strides=1,
                                      weight_decay=0.0)
    net = fully_connected(net, 1024,
                          activation='tanh',
                          weights_init='truncated_normal',
                          bias_init='zeros',
                          regularizer=None,
                          weight_decay=0)
    net = tflearn.layers.core.dropout(net, keep_prob=0.5)
    net = batch_normalization(net,
                              beta_initializer=tf.zeros_initializer(),
                              moving_variance_initializer=tf.ones_initializer())
    y_conv = fully_connected(net, dataset_meta['n_classes'],
                             activation='softmax',
                             weights_init='truncated_normal',
                             bias_init='zeros',
                             regularizer=None,
                             weight_decay=0)
    return y_conv
