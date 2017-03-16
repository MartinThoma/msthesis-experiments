#!/usr/bin/env python


"""Create a CIFAR model found by reinforcment learning."""

from keras.models import Model
from keras.layers import Input, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
import keras.backend as K


def create_rflearn_net(nb_classes, img_dim):
    """
    Build the CIFAR model described in the Zoph and Le paper.

    Neural Archtecture Search With Reinforcment Learning

    Parameters
    ----------
    nb_classes: number of classes
    img_dim: tuple of shape (channels, rows, columns) or
                            (rows, columns, channels)

    Returns
    -------
    keras tensor
    """
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    model_input = Input(shape=img_dim)
    x1 = Convolution2D(36, 3, 3,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(model_input)
    x2 = Convolution2D(48, 3, 3,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(x1)
    m3 = merge([x1, x2], mode='concat', concat_axis=concat_axis)
    x3 = Convolution2D(36, 3, 3,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(m3)
    m4 = merge([x1, x2, x3], mode='concat', concat_axis=concat_axis)
    x4 = Convolution2D(36, 5, 5,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(m4)
    m5 = merge([x3, x4], mode='concat', concat_axis=concat_axis)
    x5 = Convolution2D(48, 7, 3,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(m5)
    m6 = merge([x2, x3, x4, x5], mode='concat', concat_axis=concat_axis)
    x6 = Convolution2D(48, 7, 7,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(m6)
    m7 = merge([x2, x3, x4, x5, x6], mode='concat', concat_axis=concat_axis)
    x7 = Convolution2D(48, 7, 7,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(m7)
    m8 = merge([x1, x6, x7], mode='concat', concat_axis=concat_axis)
    x8 = Convolution2D(36, 3, 7,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(m8)
    m9 = merge([x1, x5, x6, x8], mode='concat', concat_axis=concat_axis)
    x9 = Convolution2D(36, 1, 7,
                       init="he_uniform", border_mode="same", bias=True,
                       W_regularizer=l2(1E-4))(m9)
    m10 = merge([x1, x3, x4, x5, x6, x7, x8, x9],
                mode='concat', concat_axis=concat_axis)
    x10 = Convolution2D(36, 7, 7,
                        init="he_uniform", border_mode="same", bias=True,
                        W_regularizer=l2(1E-4))(m10)
    m11 = merge([x1, x2, x5, x6, x7, x8, x9, x10],
                mode='concat', concat_axis=concat_axis)
    x11 = Convolution2D(36, 7, 5,
                        init="he_uniform", border_mode="same", bias=True,
                        W_regularizer=l2(1E-4))(m11)
    m12 = merge([x1, x2, x3, x4, x6, x11],
                mode='concat', concat_axis=concat_axis)
    x12 = Convolution2D(36, 7, 5,
                        init="he_uniform", border_mode="same", bias=True,
                        W_regularizer=l2(1E-4))(m12)
    m13 = merge([x1, x3, x6, x7, x8, x9, x10, x11, x12],
                mode='concat', concat_axis=concat_axis)
    x13 = Convolution2D(48, 5, 7,
                        init="he_uniform", border_mode="same", bias=True,
                        W_regularizer=l2(1E-4))(m13)
    m14 = merge([x3, x7, x12, x13],
                mode='concat', concat_axis=concat_axis)
    x14 = Convolution2D(48, 5, 7,
                        init="he_uniform", border_mode="same", bias=True,
                        W_regularizer=l2(1E-4))(m14)
    m15 = merge([x2, x7, x11, x12, x13, x14],
                mode='concat', concat_axis=concat_axis)
    x15 = Convolution2D(48, 5, 7,
                        init="he_uniform", border_mode="same", bias=True,
                        W_regularizer=l2(1E-4))(m15)
    x = Flatten(name='flatten')(x15)
    x16 = Dense(nb_classes, activation='softmax', name='predictions')(x)
    cifar_rl_net = Model(input=model_input, output=x16)
    return cifar_rl_net


if __name__ == '__main__':
    model = create_rflearn_net(nb_classes=10, img_dim=(32, 32, 3))
    model.summary()
