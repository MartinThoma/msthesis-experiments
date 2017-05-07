#!/usr/bin/env python

"""Create a sequential model."""

from keras.layers import Activation, Input
from keras.layers import Convolution2D, MaxPooling2D
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

    # Network definition:
    # 3x48x48-100C7-MP2-150C4-150MP2-250C4-250MP2-300N-43N
    input_ = Input(shape=input_shape)
    x = input_

    # 48x48
    x = Convolution2D(100, (7, 7), padding='valid',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(0.0001))(x)
    # 42x42
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 21x21
    x = Convolution2D(150, (4, 4), padding='valid',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(0.0001))(x)
    # 18x18
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 9x9
    x = Convolution2D(250, (4, 4), padding='valid',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(0.0001))(x)
    # 6x6
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 3x3
    x = Convolution2D(300, (1, 1), padding='valid',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(0.0001))(x)
    x = Activation('tanh')(x)
    x = Convolution2D(nb_classes, (1, 1), padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(0.0001))(x)
    x = GlobalAveragePooling2D()(x)  # Adjust for FCN
    x = Activation('softmax')(x)
    model = Model(inputs=input_, outputs=x)
    return model

if __name__ == '__main__':
    model = create_model(43, (48, 48, 3))
    model.summary()
    from keras.utils import plot_model
    plot_model(model, to_file='baseline.png',
               show_layer_names=False, show_shapes=True)
