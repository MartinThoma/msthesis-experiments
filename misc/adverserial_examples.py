#!/usr/bin/env python

from cleverhans.attacks import fgsm
from keras.models import load_model
import scipy.misc
import yaml
from run_training import make_paths_absolute
import os
import imp
import sys
import glob
import logging
import pprint

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def main(config, data_module, model_path):
    """Run analysis."""
    model = load_model(model_path)
    data = data_module.load_data(config)
    x = data['x_train'][:64]
    predictions = model.predict(x)
    adv_x = fgsm(x, predictions, eps=0.3)
    scipy.misc.imshow(x[0])
    scipy.misc.imshow(adv_x[0])


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="Experiment yaml file",
                        required=True,
                        metavar="FILE")
    parser.add_argument("--model",
                        dest="model_path",
                        help="Model h5 file",
                        metavar="FILE")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Read YAML experiment definition file
    with open(args.filename, 'r') as stream:
        config = yaml.load(stream)
    # Make paths absolute
    config = make_paths_absolute(os.path.dirname(args.filename),
                                 config)

    # Print experiment file
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    # Load data module
    dpath = config['dataset']['script_path']
    sys.path.insert(1, os.path.dirname(dpath))
    data = imp.load_source('data', config['dataset']['script_path'])

    # Load model
    if args.model_path is not None:
        model_path = args.model_path
    else:
        artifacts_path = config['train']['artifacts_path']
        model_path = os.path.basename(config['train']['artifacts_path'])
        model_paths = glob.glob("{}/*.h5".format(artifacts_path))
        model_paths = [m for m in model_paths if "_chk.h5" not in m]
        model_path = model_paths[0]
    logging.info("Take {}".format(model_path))
    main(config, data, model_path)
