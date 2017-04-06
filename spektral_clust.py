#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimize confusion matrix.

For more information, see

* http://cs.stackexchange.com/q/70627/2914
* http://datascience.stackexchange.com/q/17079/8820
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)
import logging
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import silhouette_score

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def main(cm_file, perm_file, steps, labels_file, limit_classes=None):
    """Run optimization and generate output."""
    # Load confusion matrix
    with open(cm_file) as f:
        cm = json.load(f)
        cm = np.array(cm)

    # Load labels
    if os.path.isfile(labels_file):
        with open(labels_file, "r") as f:
            labels = json.load(f)
    else:
        labels = list(range(len(cm)))

    n_clusters = 14

    from sklearn.cluster import SpectralClustering
    spectral = SpectralClustering(n_clusters=n_clusters,
                                  eigen_solver='arpack',
                                  affinity="nearest_neighbors")
    spectral.fit(cm)
    if hasattr(spectral, 'labels_'):
        y_pred = spectral.labels_.astype(np.int)
    else:
        y_pred = spectral.predict(cm)
    print("silhouette_score={}".format(silhouette_score(cm, y_pred)))
    grouping = [[] for _ in range(n_clusters)]
    for label, y in zip(labels, y_pred):
        grouping[y].append(label)
    for group in grouping:
        print(group)


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cm",
                        dest="cm_file",
                        help=("path of a json file with a confusion matrix"),
                        metavar="cm.json",
                        required=True)
    parser.add_argument("--perm",
                        dest="perm_file",
                        help=("path of a json file with a permutation to "
                              "start with"),
                        metavar="perm.json",
                        default="")
    parser.add_argument("--labels",
                        dest="labels_file",
                        help=("path of a json file with a list of label "
                              "names"),
                        metavar="labels.json",
                        default="")
    parser.add_argument("-n",
                        dest="n",
                        default=4 * 10**5,
                        type=int,
                        help="number of steps to iterate")
    parser.add_argument("--limit",
                        dest="limit_classes",
                        type=int,
                        help="Limit the number of classes in the output")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    import doctest
    doctest.testmod()
    main(args.cm_file, args.perm_file, args.n, args.labels_file,
         args.limit_classes)
