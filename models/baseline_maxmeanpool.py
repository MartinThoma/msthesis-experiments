#!/usr/bin/env python

"""Create a sequential model."""

from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers.merge import Add


def max_avg_pool2d(x):
    max_x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    mean_x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return Add()([max_x, mean_x])  # concatenate on channel


def max_avg_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    # shape[1] *= 2
    shape[1] /= 2
    shape[2] /= 2
    return tuple(shape)


def create_model(nb_classes, input_shape, config=None):
    """Create a VGG-16 like model."""
    if config is None:
        config = {'model': {}}
    model = Sequential()
    # input_shape = (None, None, 3)  # for fcn
    first = True
    min_feature_map_dimension = min(input_shape[:2])
    if min_feature_map_dimension < 32:
        print("ERROR: Please upsample the feature maps to have at least "
              "a size of 32 x 32. Currently, it has {}".format(input_shape))
    tmp = min_feature_map_dimension / 32.
    while tmp >= 2.:
        if first:
            model.add(Convolution2D(32, (3, 3), padding='same',
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=l2(0.0001),
                                    input_shape=input_shape))
            first = False
        else:
            model.add(Convolution2D(32, (3, 3), padding='same',
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=l2(0.0001)))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(Convolution2D(32, (3, 3), padding='same',
                                kernel_initializer='he_uniform',
                                kernel_regularizer=l2(0.0001)))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(Lambda(max_avg_pool2d,
                         output_shape=max_avg_pool2d_output_shape))
        tmp /= 2

    if 'c1' in config['model']:
        c1_nb_filter = config['model']['c1']['nb_filter']
    else:
        c1_nb_filter = 32
    if 'c8' in config['model']:
        c8_nb_filter = config['model']['c8']['nb_filter']
    else:
        c8_nb_filter = 512

    if first:
        model.add(Convolution2D(c1_nb_filter, (3, 3), padding='same',
                                kernel_initializer='he_uniform',
                                kernel_regularizer=l2(0.0001),
                                input_shape=input_shape))
        first = False
    else:
        model.add(Convolution2D(c1_nb_filter, (3, 3), padding='same',
                                kernel_initializer='he_uniform',
                                kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Convolution2D(32, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Lambda(max_avg_pool2d,
                         output_shape=max_avg_pool2d_output_shape))

    model.add(Convolution2D(64, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Convolution2D(64, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Lambda(max_avg_pool2d,
                         output_shape=max_avg_pool2d_output_shape))

    model.add(Convolution2D(64, (3, 3), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Lambda(max_avg_pool2d,
                         output_shape=max_avg_pool2d_output_shape))

    model.add(Convolution2D(512, (4, 4),
                            padding='valid',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(c8_nb_filter, (1, 1), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(nb_classes, (1, 1), padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(0.0001)))
    model.add(GlobalAveragePooling2D())  # Adjust for FCN
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    model = create_model(100, (32, 32, 3))
    model.summary()
    from keras.utils import plot_model
    plot_model(model, to_file='seq3.png',
               show_layer_names=False, show_shapes=True)
