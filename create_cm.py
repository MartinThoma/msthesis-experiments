#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the confusion matrices for this model."""

import logging
import sys
import yaml
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import io
import json
import imp
import pprint
import collections
import os
train_keras = imp.load_source('train_keras', "train/train_keras.py")
from train_keras import get_level, flatten_completely, filter_by_class
from train_keras import update_labels
from analyze_model import make_mosaic
from run_training import make_paths_absolute
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def run_model_prediction(model, config, X_train, X, n_classes):
    if config['evaluate']['augmentation_factor'] > 1:
        # Test time augmentation
        da = config['evaluate']['data_augmentation']
        if 'hue_shift' in da:
            hsv_augmentation = (da['hue_shift'],
                                da['saturation_scale'],
                                da['saturation_shift'],
                                da['value_scale'],
                                da['value_shift'])
        else:
            hsv_augmentation = None

        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=da['featurewise_center'],
            # set each sample mean to 0
            samplewise_center=da['samplewise_center'],
            # divide inputs by std of the dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=da['samplewise_std_normalization'],
            zca_whitening=da['zca_whitening'],
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=da['rotation_range'],
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=da['width_shift_range'],
            # randomly shift images vertically (fraction of total height)
            height_shift_range=da['height_shift_range'],
            horizontal_flip=da['horizontal_flip'],
            vertical_flip=da['vertical_flip'],
            hsv_augmentation=hsv_augmentation,
            zoom_range=da['zoom_range'],
            shear_range=da['shear_range'],
            channel_shift_range=da['channel_shift_range'])

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train, seed=0)

        # Apply normalization to test data
        # for i in range(len(X)):
        #     X[i] = datagen.standardize(X[i])

        y_pred = np.zeros((X.shape[0], n_classes))

        a_factor = config['evaluate']['augmentation_factor']
        samples = config['evaluate']['batch_size']
        batch_arr = np.zeros([a_factor * samples] +
                             list(X[0].shape))
        print("batch_arr.shape={}".format(batch_arr.shape))
        if len(X) % samples != 0:
            logging.warning(("len(X) % config['evaluate']['batch_size'] != 0 "
                             "(len(X)={})").format(len(X)))
        run_through_X = False
        for index_sample in range(0, len(X), samples):
            for subi in range(samples):
                if index_sample + subi == len(X):
                    run_through_X = True
                    break
                batch = datagen.flow(np.array([X[index_sample + subi]]),
                                     np.array([np.zeros(n_classes)]),
                                     batch_size=a_factor)
                for i, el in enumerate(batch):
                    if i == 0:
                        batch_arr[subi * a_factor + i] = \
                            X[index_sample + subi]
                        continue
                    x, label = el
                    batch_arr[subi * a_factor + i] = x
                    if i == a_factor - 1:
                        break
            if run_through_X:
                break
                # import scipy.misc
                # mosaic = make_mosaic(batch_arr, 2, 2)
                # scipy.misc.imshow(mosaic)
            y_pred_single = model.predict(batch_arr)
            for subi in range(samples):
                y_pred_s = (y_pred_single[subi * a_factor:
                                          (subi + 1) * a_factor].sum(axis=0) /
                            float(a_factor))
                y_pred[index_sample + subi] = y_pred_s
            print("\t{:>7} of {}".format(index_sample, len(X)))
    else:
        y_pred = model.predict(X)
    return y_pred


def _calculate_cm(config, model, X_train, X, y, n_classes, smooth):
    y_i = y.flatten()
    y_pred = run_model_prediction(model, config, X_train, X, n_classes)

    if smooth:
        cm = np.zeros((n_classes, n_classes), dtype=np.float64)
        class_count = np.zeros(n_classes, dtype=np.int)
        for i, pred in zip(y_i, y_pred):
            cm[i] += pred
            class_count[i] += 1
        for i in range(n_classes):
            cm[i] /= class_count[i]
    else:
        cm = np.zeros((n_classes, n_classes), dtype=np.int)
        y_pred_i = y_pred.argmax(1)
        for i, j in zip(y_i, y_pred_i):
            cm[i][j] += 1
    return cm


def _write_cm(cm, path):
    # Serialize confusion matrix
    with io.open(path, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(cm.tolist(), sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False)
        outfile.write(to_unicode(str_))


def create_cm(data_module, config, smooth, model_path):
    """
    Create confusion matrices.

    Parameters
    ----------
    data_module : Python module
    config : dict
    smooth : boolean
    model_path : string
    """
    artifacts_path = config['train']['artifacts_path']
    if model_path is None:
        model_path = os.path.join(artifacts_path,
                                  config['train']['model_output_fname'])
    # Load model
    if not os.path.isfile(model_path):
        logging.error("File {} does not exist. You might need to train it."
                      .format(model_path))
        sys.exit(-1)
    logging.info("Load model {}".format(model_path))
    model = load_model(model_path)

    # The data, shuffled and split between train and test sets:
    data = data_module.load_data()
    print("Data loaded.")

    X_train, y_train = data['x_train'], data['y_train']
    X_train = data_module.preprocess(X_train)
    X_test, y_test = data['x_test'], data['y_test']
    X_test = data_module.preprocess(X_test)

    # load hierarchy, if present
    if 'hierarchy_path' in config['dataset']:
        with open(config['dataset']['hierarchy_path']) as data_file:
            hierarchy = json.load(data_file)
        if 'subset' in config['dataset']:
            remaining_cls = get_level(hierarchy, config['dataset']['subset'])
            logging.info("Remaining classes: {}".format(remaining_cls))
            # Only do this if coarse is False:
            remaining_cls = flatten_completely(remaining_cls)
            data_module.n_classes = len(remaining_cls)
            X_train, y_train = filter_by_class(X_train, y_train, remaining_cls)
            X_test, y_test = filter_by_class(X_test, y_test, remaining_cls)
            old_cli2new_cli = {}
            for new_cli, old_cli in enumerate(remaining_cls):
                old_cli2new_cli[old_cli] = new_cli
            y_train = update_labels(y_train, old_cli2new_cli)
            y_test = update_labels(y_test, old_cli2new_cli)

    nb_classes = data_module.n_classes
    logging.info("# classes = {}".format(data_module.n_classes))

    # Calculate confusion matrix for training set
    cm = _calculate_cm(config, model, X_train, X_train, y_train, nb_classes,
                       smooth)
    correct_count = sum([cm[i][i] for i in range(nb_classes)])
    acc = correct_count / float(cm.sum())
    print("Accuracy (Train): {:0.2f}% ({} of {} wrong)"
          .format(acc * 100, cm.sum() - correct_count, cm.sum()))
    _write_cm(cm, path=os.path.join(artifacts_path, 'cm-train.json'))

    # Calculate confusion matrix for test set
    cm = _calculate_cm(config, model, X_train, X_test, y_test, nb_classes,
                       smooth)
    correct_count = sum([cm[i][i] for i in range(nb_classes)])
    acc = correct_count / float(cm.sum())
    print("Accuracy (Test): {:0.2f}% ({} of {} wrong)"
          .format(acc * 100, cm.sum() - correct_count, cm.sum()))
    _write_cm(cm, path=os.path.join(artifacts_path, 'cm-test.json'))

    # Calculate the accuracy for each sub-group
    if 'hierarchy_path' in config['dataset']:
        hierarchy = get_level(hierarchy, config['dataset']['subset'])
        for class_group in hierarchy:
            if isinstance(class_group, collections.Iterable):
                # calculate acc on this group
                print("Group: {}".format(class_group))
                correct = sum([cm[i][i] for i in class_group])
                all_ = sum(cm[i][j] for i in class_group for j in class_group)
                acc = correct / float(all_)
                print("\t{:0.2f}%".format(acc * 100))


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE.yaml",
                        required=True)
    parser.add_argument("--model",
                        dest="model_fname",
                        help="path to a h5 keras model file",
                        default=None)
    parser.add_argument("--smooth",
                        action="store_true",
                        dest="smooth",
                        default=False,
                        help="Use prediction probability instead of argmax")
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    # Read YAML experiment definition file
    with open(args.filename, 'r') as stream:
        experiment_meta = yaml.load(stream)
    # Make paths absolute
    experiment_meta = make_paths_absolute(os.path.dirname(args.filename),
                                          experiment_meta)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(experiment_meta)
    dpath = experiment_meta['dataset']['script_path']
    sys.path.insert(1, os.path.dirname(dpath))
    data = imp.load_source('data', experiment_meta['dataset']['script_path'])
    create_cm(data, experiment_meta, args.smooth, args.model_fname)
