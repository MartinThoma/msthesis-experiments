#!/usr/bin/env python

from keras.optimizers import SGD


def get_optimizer(config):
    lr = config['optimizer']['initial_lr']
    optimizer = SGD(lr=lr)  # Using Adam instead of SGD to speed up training
    return optimizer
