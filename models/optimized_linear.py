#!/usr/bin/env python

"""Create a sequential model."""

from keras.layers import Dropout, Activation, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.regularizers import l2
from keras.models import Model


def create_model(nb_classes, input_shape, config=None):
    """Create a VGG-16 like model."""
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_channels, nb_rows, "
                        "nb_cols) or (nb_rows, nb_cols, nb_channels), "
                        "depending on your backend.")
    if config is None:
        config = {'model': {}}

    min_feature_map_dimension = min(input_shape[:2])
    if min_feature_map_dimension < 32:
        print("ERROR: Please upsample the feature maps to have at least "
              "a size of 32 x 32. Currently, it has {}".format(input_shape))
    nb_filter = 32

    if 'l2' not in config['model']:
        config['model']['l2'] = 0.0001

    # Network definition
    # input_shape = (None, None, 3)  # for fcn
    input_ = Input(shape=input_shape)
    x = input_

    # Scale feature maps down to [63, 32] x [63, 32]
    tmp = min_feature_map_dimension / 32.
    if tmp >= 2:
        while tmp >= 2.:
            for _ in range(2):
                x = Convolution2D(nb_filter, (3, 3), padding='same',
                                  kernel_initializer='he_uniform',
                                  kernel_regularizer=l2(config['model']['l2']))(x)
                x = BatchNormalization()(x)
                x = Activation('linear')(x)
            x = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(x)
            nb_filter *= 2
            tmp /= 2

    # 32x32
    x = Convolution2D(nb_filter + 37, (3, 3), padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(config['model']['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('linear')(x)
    x = Convolution2D(nb_filter + 37, (3, 3), padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(config['model']['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('linear')(x)

    # 16x16
    x = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(x)
    x = Convolution2D(2 * nb_filter, (3, 3), padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(config['model']['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('linear')(x)
    x = Convolution2D(2 * nb_filter, (3, 3), padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(config['model']['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('linear')(x)

    # 8x8
    x = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(x)
    x = Convolution2D(2 * nb_filter, (3, 3), padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(config['model']['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('linear')(x)

    # 4x4
    x = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(x)
    x = Convolution2D(512, (4, 4),
                      padding='valid',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(config['model']['l2']),
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('linear')(x)
    x = Dropout(0.5)(x)

    # 1x1
    x = Convolution2D(512, (1, 1), padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(config['model']['l2']),
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('linear')(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(nb_classes, (1, 1), padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(config['model']['l2']),
                      use_bias=False)(x)
    x = GlobalAveragePooling2D()(x)  # Adjust for FCN
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)
    model = Model(inputs=input_, outputs=x)
    return model

if __name__ == '__main__':
    model = create_model(100, (32, 32, 3))
    model.summary()
    from keras.utils import plot_model
    plot_model(model, to_file='baseline.png',
               show_layer_names=False, show_shapes=True)
