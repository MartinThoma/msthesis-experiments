#!/usr/bin/env python

"""
Train a CNN on CIFAR 100.

Takes about 100 minutes.
"""

from __future__ import print_function
import numpy as np
np.random.seed(0)
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# from sklearn.model_selection import train_test_split
import csv
import json
import logging
import os
import sys
import collections

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def flatten_completely(iterable):
    """Make a flat list out of an interable of iterables of iterables ..."""
    flattened = []
    queue = [iterable]
    while len(queue) > 0:
        el = queue.pop(0)
        if isinstance(el, collections.Iterable):
            for subel in el:
                queue.append(subel)
        else:
            flattened.append(el)
    return flattened


def get_oldi2newi(hierarchy):
    """Get a dict mapping from the old class idx to the new class indices."""
    oldi2newi = {}
    n = len(flatten_completely(hierarchy))
    for i in range(n):
        if i in hierarchy:  # is it in the first level?
            oldi2newi[i] = hierarchy.index(i)
        else:
            for j, group in enumerate(hierarchy):
                if isinstance(group, (int, long)):
                    continue
                if i in group:
                    oldi2newi[i] = j
    return oldi2newi


def apply_hierarchy(hierarchy, y):
    """Apply a hierarchy to a label vector."""
    oldi2newi = get_oldi2newi(hierarchy)
    for i in range(len(y)):
        y[i] = oldi2newi[y[i]]
    return y


def main(data_module, model_module, optimizer_module, filename, config):
    """Patch everything together."""
    batch_size = config['train']['batch_size']
    nb_epoch = config['train']['epochs']

    # The data, shuffled and split between train and test sets:
    data = data_module.load_data()
    print("Data loaded.")

    X_train, y_train = data['x_train'], data['y_train']
    X_train = data_module.preprocess(X_train)
    X_test, y_test = data['x_test'], data['y_test']
    X_test = data_module.preprocess(X_test)

    # apply hierarchy, if present
    if 'hierarchy_path' in config['dataset']:
        with open(config['dataset']['hierarchy_path']) as data_file:
            hierarchy = json.load(data_file)
        logging.info("Loaded hierarchy: {}".format(hierarchy))
        data_module.n_classes = len(hierarchy)
        logging.info("New model has {} classes".format(data_module.n_classes))
        y_train = apply_hierarchy(hierarchy, y_train)
        y_test = apply_hierarchy(hierarchy, y_test)
    nb_classes = data_module.n_classes
    img_rows = data_module.img_rows
    img_cols = data_module.img_cols
    img_channels = data_module.img_channels
    data_augmentation = config['train']['data_augmentation']

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
    #                                                   test_size=0.10,
    #                                                   random_state=42)

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_train = Y_train.reshape((-1, 1, 1, nb_classes))  # For fcn

    # Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    # Y_test = Y_test.reshape((-1, 1, 1, nb_classes))

    # Input shape depends on the backend
    if K.image_dim_ordering() == "th":
        input_shape = (img_channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, img_channels)

    model = model_module.create_model(nb_classes, input_shape)
    print("Model created")

    model.summary()
    optimizer = optimizer_module.get_optimizer(config)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            # divide inputs by std of the dataset
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=15,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=5. / 32,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=5. / 32,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            hsv_augmentation=(0.2, 0.5, 0.2, 0.5, 0.2))

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train, seed=0)

        # Fit the model on the batches generated by datagen.flow().
        model_chk_path = os.path.join(config['train']['artifacts_path'],
                                      config['train']['checkpoint_fname'])
        cb = ModelCheckpoint(model_chk_path,
                             monitor="val_acc",
                             save_best_only=True,
                             save_weights_only=False)
        history_cb = model.fit_generator(datagen.flow(X_train, Y_train,
                                         batch_size=batch_size),
                                         steps_per_epoch=X_train.shape[0]  // batch_size,
                                         epochs=nb_epoch,
                                         validation_data=(X_test, Y_test),
                                         callbacks=[cb])
        loss_history = history_cb.history["loss"]
        acc_history = history_cb.history["acc"]
        val_acc_history = history_cb.history["val_acc"]
        np_loss_history = np.array(loss_history)
        np_acc_history = np.array(acc_history)
        np_val_acc_history = np.array(val_acc_history)
        data = zip(np_loss_history, np_acc_history, np_val_acc_history)
        data = [("%0.4f" % el[0],
                 "%0.4f" % el[1],
                 "%0.4f" % el[2]) for el in data]
        with open(config['train']['history_fname'], 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerows([("loss", "acc", "val_acc")])
            writer.writerows(data)
    model_fn = os.path.join(config['train']['artifacts_path'],
                            config['train']['model_output_fname'])
    model.save(model_fn)
