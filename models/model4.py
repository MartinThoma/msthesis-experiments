#!/usr/bin/env python

"""Create a sequential model."""

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Flatten, Activation, concatenate
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def create_model(nb_classes, input_shape):
    """Create a model with skip connection."""
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    model_input = Input(shape=input_shape)
    x1 = Convolution2D(32, (3, 3),
                       kernel_initializer="he_uniform", padding="same",
                       use_bias=True,
                       kernel_regularizer=l2(1E-4))(model_input)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x2 = Convolution2D(29, (3, 3),
                       kernel_initializer="he_uniform", padding="same",
                       use_bias=True,
                       kernel_regularizer=l2(1E-4))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = concatenate([model_input, x2], axis=concat_axis)
    m3 = MaxPooling2D(pool_size=(2, 2))(x2)
    x4 = Convolution2D(64, (3, 3),
                       kernel_initializer="he_uniform", padding="same",
                       use_bias=True,
                       kernel_regularizer=l2(1E-4))(m3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x5 = Convolution2D(64, (3, 3),
                       kernel_initializer="he_uniform", padding="same",
                       use_bias=True,
                       kernel_regularizer=l2(1E-4))(x4)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = concatenate([m3, x5], axis=concat_axis)
    m6 = MaxPooling2D(pool_size=(2, 2))(x5)

    x7 = Convolution2D(64, (3, 3),
                       kernel_initializer="he_uniform", padding="same",
                       use_bias=True,
                       kernel_regularizer=l2(1E-4))(m6)
    x7 = BatchNormalization()(x7)
    x7 = Activation('relu')(x7)
    m8 = MaxPooling2D(pool_size=(2, 2))(x7)

    x9 = Convolution2D(512, (4, 4),
                       kernel_initializer="he_uniform", padding="valid",
                       use_bias=True,
                       kernel_regularizer=l2(1E-4))(m8)
    x9 = BatchNormalization()(x9)
    x9 = Activation('relu')(x9)
    x10 = Dropout(0.5)(x9)
    x11 = Convolution2D(512, (1, 1),
                        kernel_initializer="he_uniform", padding="same",
                        use_bias=True,
                        kernel_regularizer=l2(1E-4))(x10)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)
    x12 = Dropout(0.5)(x11)
    x13 = Convolution2D(nb_classes, (1, 1),
                        kernel_initializer="he_uniform", padding="same",
                        use_bias=True,
                        kernel_regularizer=l2(1E-4))(x12)
    x13 = BatchNormalization()(x13)
    x13 = Activation('softmax')(x13)
    x13 = Flatten()(x13)
    model = Model(inputs=model_input, outputs=x13)
    return model

if __name__ == '__main__':
    model = create_model(100, (32, 32, 3))
    model.summary()
    from keras.utils import plot_model
    plot_model(model, to_file='seq3.png',
               show_layer_names=False, show_shapes=True)
