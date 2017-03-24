#!/usr/bin/env python

from keras.optimizers import Adam


def get_optimizer(config):
    lr = config['optimizer']['initial_lr']
    optimizer = Adam(lr=lr)  # Using Adam instead of SGD to speed up training
    return optimizer
