#!/usr/bin/env python

"""Create a sequential model."""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, Dense
from keras.layers import Convolution2D, MaxPooling2D


def create_model(nb_classes, input_shape):
    """Create a VGG-16 like model."""
    model = Sequential()
    model.add(Convolution2D(4, (3, 3), padding='same', activation='relu',
                            input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(4, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    create_model(100, (32, 32, 3))
