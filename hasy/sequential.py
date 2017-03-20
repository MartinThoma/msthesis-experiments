#!/usr/bin/env python

"""
Train a CNN on HASY.

Takes about 100 minutes.
"""

from __future__ import print_function
import numpy as np
np.random.seed(0)
import hasy
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.initializers import TruncatedNormal
import csv
import densenet

batch_size = 128
nb_classes = hasy.n_classes
nb_epoch = 42
data_augmentation = False
model_type = 'sequential'

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
data = hasy.load_data()
X_train, y_train = data['x_train'], data['y_train']
X_val, y_val = X_train, y_train
X_test, y_test = data['x_test'], data['y_test']
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
#                                                   test_size=0.10,
#                                                   random_state=42)


# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

if model_type == 'sequential':
    # keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    model = Sequential()
    print("input_shape: %s" % str(X_train.shape[1:]))
    model.add(Convolution2D(32, (3, 3), padding='same', activation='relu',
                            input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dense(1024, activation='tanh'))
    model.add(Convolution2D(1024, (16, 16), padding='valid',
                            activation='tanh',
                            kernel_initializer='Orthogonal'))
    model.add(Dropout(0.5))
    # model.add(Dense(1024, activation='tanh'))
    model.add(Convolution2D(1024, (1, 1), padding='valid',
                            activation='tanh',
                            kernel_initializer='Orthogonal'))
    model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))
    model.add(Convolution2D(nb_classes, (1, 1), padding='same',
                            kernel_initializer='Orthogonal'))
    model.add(Flatten())
    model.add(Activation('softmax'))
elif model_type == 'dense':
    if K.image_dim_ordering() == "th":
        img_dim = (img_channels, img_rows, img_cols)
    else:
        img_dim = (img_rows, img_cols, img_channels)

    depth = 100
    nb_dense_block = 3
    growth_rate = 12
    nb_filter = -1
    bottleneck = True
    reduction = 0.5
    dropout_rate = 0.0  # 0.0 for data augmentation

    model = densenet.create_dense_net(nb_classes, img_dim, depth, nb_dense_block,
                                      growth_rate, nb_filter,
                                      bottleneck=bottleneck, reduction=reduction,
                                      dropout_rate=dropout_rate)
else:
    print("Not implemented model '{}'".format(model_type))
print("Model created")

model.summary()
optimizer = Adam(lr=1e-4)  # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_val /= 255.0
X_test /= 255.0

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)
model.save('hasy.h5')
