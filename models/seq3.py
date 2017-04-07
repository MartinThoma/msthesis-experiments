#!/usr/bin/env python

"""Create a sequential model."""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def create_model(nb_classes, input_shape):
    """Create a VGG-16 like model."""
    model = Sequential()
    print("input_shape: %s" % str(input_shape))
    # input_shape = (None, None, 3)  # for fcn
    model.add(Convolution2D(32, (3, 3), padding='same',
                            input_shape=input_shape,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (4, 4),
                            padding='valid',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(512, (1, 1), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(nb_classes, (1, 1), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.add(Flatten())  # Remove for FCN
    return model

if __name__ == '__main__':
    model = create_model(100, (32, 32, 3))
    model.summary()
    from keras.utils import plot_model
    plot_model(model, to_file='seq3.png',
               show_layer_names=False, show_shapes=True)
