#!/usr/bin/env python

"""Create a resnet model."""

from keras.models import Model
from keras.layers import Input
from keras.layers.merge import add
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, AveragePooling2D
from keras.regularizers import l2


def residual_block(input_tensor, stage,
                   kernel_initializer='he_uniform', l2reg=0.0,
                   use_shortcuts=True):
    """Create a ResNet pre-activation bottleneck layer."""
    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = 'add' + str(stage)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1
    # conv
    if stage > 1:  # first activation is just after conv1
        x = BatchNormalization(axis=1, name=bn_name + 'a')(input_tensor)
        x = Activation('relu', name=relu_name + 'a')(x)
    else:
        x = input_tensor

    # x = Convolution2D(nb_bottleneck_filters, (1, 1),
    #                   kernel_initializer=kernel_initializer,
    #                   kernel_regularizer=l2(l2reg),
    #                   use_bias=False,
    #                   name=conv_name + 'a')(x)

    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters
    # via FxF conv
    # x = BatchNormalization(axis=1, name=bn_name + 'b')(x)
    # x = Activation('relu', name=relu_name + 'b')(x)
    x = Convolution2D(3, (3, 3),
                      padding='same',
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2(l2reg),
                      use_bias=False,
                      name=conv_name + 'b')(x)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1
    # conv
    x = BatchNormalization(axis=1, name=bn_name + 'c')(x)
    x = Activation('relu', name=relu_name + 'c')(x)
    x = Convolution2D(3, (1, 1),
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2(l2reg),
                      name=conv_name + 'c')(x)

    # merge
    if use_shortcuts:
        x = add([x, input_tensor], name=merge_name)

    return x


def create_model(nb_classes=10, input_shape=(3, 32, 32),
                 res_blocks=30,
                 final_layer_params=None,
                 kernel_initializer='he_uniform',
                 l2reg=0.0,
                 use_shortcuts=True):
    """
    Create ResNet using pre-activation.

    based on the work in
    "Identity Mappings in Deep Residual Networks"  by He et al
    http://arxiv.org/abs/1603.05027

    The following network definition achieves 92.0% accuracy on CIFAR-10 test
    using `adam` optimizer, 100 epochs, learning rate schedule of 1e.-3 / 1.e-4
    / 1.e-5 with transitions at 50 and 75 epochs:
    ResNetPreAct(layer1_params=(3,128,2),res_layer_params=(3,32,25),reg=reg)

    Removed max pooling and using just stride in first convolutional layer.
    Motivated by "Striving for Simplicity: The All Convolutional Net"  by
    Springenberg et al (https://arxiv.org/abs/1412.6806) and my own experiments
    where I observed about 0.5% improvement by replacing the max pool
    operations in the VGG-like cifar10_cnn.py example in the Keras
    distribution.

    Parameters
    ----------
    input_dim : tuple
        (C, H, W) or (H, W, C), depending on the backend
    nb_classes : int
    layer1_params : tuple
        (filter size, num filters, stride for conv)
    res_layer_params : tuple
        (filter size, num res layer filters, num res stages)
    final_layer_params : None or tuple
        (filter size, num filters, stride for conv)
    kernel_initializer : string, optional (default: 'he_uniform')
        type of kernel initialization to use
    l2reg : float
        L2 weight regularization (or weight decay)
    use_shortcuts : bool
        to evaluate difference between residual and non-residual network
    """
    img_input = Input(shape=input_shape, name='input')
    x = Convolution2D(3, (3, 3),
                      padding='same',
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2(l2reg),
                      use_bias=False,
                      name='conv0')(img_input)
    x = BatchNormalization(axis=1, name='bn0')(x)
    x = Activation('relu', name='relu0')(x)

    for stage in range(1, res_blocks + 1):
        x = residual_block(x, stage,
                           kernel_initializer=kernel_initializer,
                           l2reg=l2reg,
                           use_shortcuts=use_shortcuts)

    x = BatchNormalization(axis=1, name='bnF')(x)
    x = Activation('relu', name='reluF')(x)
    # x = AveragePooling2D((2, 2), name='avg_pool')(x)
    x = Flatten(name='flat')(x)
    x = Dense(nb_classes, activation='softmax', name='fc10')(x)
    return Model(img_input, x, name='rnpa')


if __name__ == '__main__':
    model = create_model(100, (32, 32, 3))
    model.summary()
    from keras.utils import plot_model
    plot_model(model, to_file='resnetpa.png',
               show_layer_names=False, show_shapes=True)
