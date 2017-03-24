#!/usr/bin/env python

"""Run an experiment."""

import logging
import sys
import os
import yaml
import imp
import pprint

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def is_valid_file(parser, arg):
    """
    Check if arg is a valid file that already exists on the file system.

    Parameters
    ----------
    parser : argparse object
    arg : str

    Returns
    -------
    arg
    """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        type=lambda x: is_valid_file(parser, x),
                        help="write report to FILE",
                        metavar="FILE",
                        required=True)
    return parser


def make_paths_absolute(experiment_meta):
    for key in experiment_meta.keys():
        if key.endswith("_path"):
            experiment_meta[key] = os.path.abspath(experiment_meta[key])
            # if not os.path.isfile(experiment_meta[key]):
            #     logging.error("%s does not exist.", experiment_meta[key])
            #     sys.exit(-1)
        if type(experiment_meta[key]) is dict:
            experiment_meta[key] = make_paths_absolute(experiment_meta[key])
    return experiment_meta


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Read YAML experiment definition file
    with open(args.filename, 'r') as stream:
        experiment_meta = yaml.load(stream)
    # Make paths absolute
    experiment_meta = make_paths_absolute(experiment_meta)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(experiment_meta)
    dpath = experiment_meta['dataset']['script_path']
    sys.path.insert(1, os.path.dirname(dpath))
    data = imp.load_source('data', experiment_meta['dataset']['script_path'])
    model = imp.load_source('model', experiment_meta['model']['script_path'])
    optimizer = imp.load_source('optimizer',
                                experiment_meta['optimizer']['script_path'])
    train = imp.load_source('train', experiment_meta['train']['script_path'])
    train.main(data, model, optimizer, os.path.abspath(args.filename),
               config=experiment_meta)
