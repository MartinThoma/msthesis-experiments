#!/usr/bin/env python

"""Create a confusion matrix with some controlled properties."""

import numpy as np
import json
import random

import logging
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def create_cm(nb_classes, nb_data, accuracy):
    cm = np.zeros((nb_classes, nb_classes), dtype=np.uint)

    # Make diagonal for accuracy
    diagonal = np.random.random(nb_classes)
    diagonal = diagonal / sum(diagonal) * (nb_classes * nb_data * accuracy)
    diagonal = [int(el) for el in diagonal]

    actual_acc = sum(diagonal) / float(nb_data * nb_classes)
    while actual_acc < accuracy:
        i = random.randint(0, nb_classes - 1)
        diagonal[i] += 1
        actual_acc = sum(diagonal) / float(nb_data * nb_classes)

    assert abs(actual_acc - accuracy) < 0.0001, actual_acc
    while any([True for el in diagonal if el > nb_data]):
        redistribute = 0
        for i, el in enumerate(diagonal):
            if el > nb_data:
                redistribute += diagonal[i] - nb_data
                diagonal[i] = nb_data
        while redistribute > 0:
            i = random.randint(0, nb_classes - 1)
            max_red = min(nb_data - diagonal[i], redistribute)
            r = random.randint(0, max_red)
            diagonal[i] += r
            redistribute -= r

    for i in range(nb_classes):
        # Set diagonal
        cm[i][i] = diagonal[i]

        # Set rest
        r = nb_data - cm[i][i]
        while r > 0:
            j = random.randint(0, nb_classes - 1)
            if j == i:
                continue
            amount = random.randint(0, r)
            cm[i][j] += amount
            r -= amount

    return cm.tolist()


def main(nb_classes, nb_data, accuracy):
    cm = create_cm(nb_classes, nb_data, accuracy)
    # Write JSON file
    with open('random-cm.json', 'w') as outfile:
        str_ = json.dumps(cm,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c",
                        dest="classes",
                        default=10,
                        type=int,
                        help="how many classes should the CM have?")
    parser.add_argument("-d",
                        dest="nb_data",
                        default=10,
                        type=int,
                        help="how many datapoints per class?")
    parser.add_argument("-a",
                        dest="accuracy",
                        default=10,
                        type=float,
                        help="how high should the accuracy be (0 to 1.0)")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.classes, args.nb_data, args.accuracy)
