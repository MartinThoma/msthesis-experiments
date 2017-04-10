#!/usr/bin/env python

import imp
import os
import sys
import pprint
import yaml
import random
import numpy as np
import time
from keras.models import load_model
train_keras = imp.load_source('train_keras', "train/train_keras.py")
from run_training import make_paths_absolute
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def inference_timing(data_module, config, model_path, batch_size):
    artifacts_path = config['train']['artifacts_path']
    if model_path is None:
        model_path = os.path.basename(config['train']['artifacts_path'])
        model_path = "{}.h5".format(model_path)
        model_path = os.path.join(artifacts_path, model_path)
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
    times = []
    x_train = data['x_train']
    for i in range(1000):
        start = random.randint(0, len(x_train) - batch_size)
        X = x_train[start:start + batch_size]
        t0 = time.time()
        model.predict(X)
        t1 = time.time()
        times.append(t1 - t0)
    times = np.array(times)
    print("mean time={} (std={}, Batch size={})"
          .format(np.mean(times), np.std(times), batch_size))


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
    parser.add_argument("-n",
                        dest="n",
                        default=1,
                        type=int,
                        help="batch size")
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
    inference_timing(data, experiment_meta, args.model_fname, args.n)
