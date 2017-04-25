#!/usr/bin/env python

from keras.optimizers import SGD


def get_optimizer(config):
    optimizer = SGD(lr=config['optimizer']['initial_lr'],
                    momentum=config['optimizer']['momentum'],
                    nesterov=True)
    return optimizer
