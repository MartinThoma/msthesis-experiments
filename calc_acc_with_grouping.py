#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculate the accuracy if grouping is applied."""

import logging
import sys
import json
import numpy as np
import collections

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def get_flat_classes(hierarchy):
    """Get a flat list of classes of a hierarchy."""
    classes_in_h = []
    for c in hierarchy:
        if isinstance(c, (int, long)):
            classes_in_h.append(c)
        else:
            classes_in_h += get_flat_classes(c)
    return classes_in_h


def get_recall(cm, class_index):
    """Calculate the recall of a class."""
    n = len(cm)
    tp = cm[class_index][class_index]
    total = 0.0
    for i in range(n):
        total += cm[class_index][i]
    return {'total': total, 'recall': float(tp) / total}


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


def get_grouped_cm(cm, hierarchy):
    """Make the cm of the root classifier by hierarchy."""
    cm_new = np.zeros((len(hierarchy), len(hierarchy)))
    oldi2newi = get_oldi2newi(hierarchy)
    n = len(flatten_completely(hierarchy))
    for i in range(n):
        for j in range(n):
            cm_new[oldi2newi[i]][oldi2newi[j]] += cm[i][j]
    return cm_new


def get_accuracy(cm):
    """Get the accuracy of a classifier by its confusion matrix."""
    return sum([cm[i][i] for i in range(len(cm))]) / float(cm.sum())


def main(cm_fname, hierarchy_fname):
    """
    Orchestrate.

    Parameters
    ----------
    cm_fname : string
        Path to a confusion matrix in JSON format.
    hierachy_fname : string
        Path to a hierarchy of classes.
    """
    # Read data
    with open(cm_fname) as data_file:
        cm = json.load(data_file)
    with open(hierarchy_fname) as data_file:
        hierarchy = json.load(data_file)
    cm = np.array(cm)

    # Sanity check of hierarchy
    flat_classes = get_flat_classes(hierarchy)
    assert len(flat_classes) == len(set(flat_classes))
    assert len(flat_classes) == len(cm)
    # recalls = [get_recall(cm, i) for i in range(len(cm))]
    # acc = (sum([el['recall'] * el['total'] for el in recalls]) /
    #        sum([el['total'] for el in recalls]))
    print("Total acc={:0.2f}".format(get_accuracy(cm) * 100))
    cm_root = get_grouped_cm(cm, hierarchy)
    print("Root acc={:0.2f}".format(get_accuracy(cm_root) * 100))
    # print(recalls)
    print(cm)
    print(hierarchy)


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cm",
                        dest="confusion_matrix_fname",
                        help="Confusion matrix",
                        metavar="CM.json",
                        required=True)
    parser.add_argument("--hierarchy",
                        dest="hierarchy_fname",
                        help="Hierarchy how to group data",
                        metavar="hierarchy.json",
                        required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.confusion_matrix_fname, args.hierarchy_fname)
