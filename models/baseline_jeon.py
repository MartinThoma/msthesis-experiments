#!/usr/bin/env python

"""Create a sequential model."""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.regularizers import l2


def create_model(nb_classes, input_shape):
    """Create a VGG-16 like model."""
    print("input_shape: %s" % str(input_shape))
    model = Sequential()
    model.add(Convolution2D(16, (1, 1), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001),
                            input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 1
    model.add(Convolution2D(48, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(48, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 2
    model.add(Convolution2D(96, (3, 3), padding='same', strides=2,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(96, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 3
    model.add(Convolution2D(192, (3, 3), padding='same', strides=2,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(192, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 
    model.add(Convolution2D(1000, (1, 1), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(nb_classes,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    model = create_model(100, (32, 32, 3))
    model.summary()
