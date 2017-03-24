#!/usr/bin/env python

"""Create a sequential model."""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D


def create_model(nb_classes, input_shape):
    """Create a VGG-16 like model."""
    model = Sequential()
    print("input_shape: %s" % str(input_shape))
    # input_shape = (None, None, 3)  # for fcn
    model.add(Convolution2D(32, (3, 3), padding='same', activation='relu',
                            input_shape=input_shape))
    model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(2048, (8, 8),
                            padding='valid', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2048, (1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(nb_classes, (1, 1), padding='same'))
    model.add(Activation('softmax'))
    model.add(Flatten())  # Remove for FCN
    return model

if __name__ == '__main__':
    create_model(100, (32, 32, 3))
